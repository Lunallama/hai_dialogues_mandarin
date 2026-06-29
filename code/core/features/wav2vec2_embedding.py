"""wav2vec2 嵌入提取 + 说话人质心距离 (entrainment 用)

设计目标
--------
把高维 wav2vec2 表征塌缩成一个**标量序列** ``embedding_cosine_dist``,
使其能无缝进入现有的 synchrony / convergence / EI 标量管线 (与
pitch / intensity / mfcc 完全同框)。

核心做法 (沿用 "个人质心 + cosine 距离" 思路, 并修正信息泄漏):
  1. 逐 IPU 提取 wav2vec2 某一层的隐藏态, 在时间维做均值池化 -> 每句一个向量;
  2. 按说话人计算 "风格质心", 再算每个 IPU 到质心的 cosine 距离 (标量);
  3. 质心提供两种口径, 便于对照实验:
       - global   : 全局质心 (该说话人**全部** IPU 的均值)  —— 原论文做法, 会"看未来";
       - expanding: 扩张窗口因果质心 (仅用该说话人**当前 IPU 之前**的历史均值) —— 不泄漏未来。

层选说明 (偏声学)
-----------------
``hidden_states`` 是长度 (num_layers + 1) 的元组:
  index 0      = 特征投影输出 (最接近 CNN/原始波形, 最"声学");
  index 1..N   = 各 Transformer 层输出。
逐层探测文献 (Pasad et al. 2021/2023, CCA) 的一致结论:
  低层 (~1-4)  偏**声学** (频谱包络 / mel / f0 相关), 与 MFCC 最冗余;
  中层 (~6-9)  偏**音素/语音学**;
  高层 (未微调) 因自编码器效应回落, 可解释性最差 —— **应避开最后一层**。
本模块默认取低层 (DEFAULT_LAYER), 符合"主要考虑偏声学特征"的需求;
做层选实验时直接改 ``layer`` 参数扫一遍即可。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 默认中文预训练模型 (对应论文引用的 chinese-speech-pretrain)
DEFAULT_MODEL = "TencentGameMate/chinese-wav2vec2-base"
# 偏声学的默认层: 低层 (1-4) 最声学。取 4 = 仍偏声学但带一点上下文。
# 纯声学/与 MFCC 对照可设 1; 偏音素可设 6-9。
DEFAULT_LAYER = 4
# 扩张窗口质心的预热句数: 历史不足该值时距离记为 NaN (基准尚未稳定)
DEFAULT_WARMUP = 5
# wav2vec2 要求的输入采样率
TARGET_SR = 16000

# 模型单例缓存: {(model_name, device): (model, device)}
_MODEL_CACHE = {}


def _get_model(model_name=DEFAULT_MODEL, use_gpu=False):
    """惰性加载 wav2vec2 模型 (单例)。失败时抛异常, 由调用方降级处理。"""
    import torch
    from transformers import Wav2Vec2Model

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    key = (model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)
    _MODEL_CACHE[key] = (model, device)
    return model, device


def _read_segment(audio_path, start, end):
    """读取 [start, end] 片段, 转单声道 + 重采样到 16kHz float32。"""
    import soundfile as sf

    duration = end - start
    if duration <= 0:
        return None

    info = sf.info(audio_path)
    file_sr = info.samplerate
    start_frame = int(start * file_sr)
    frames_to_read = int(duration * file_sr)
    if frames_to_read <= 0:
        return None

    with sf.SoundFile(audio_path) as f:
        f.seek(start_frame)
        signal = f.read(frames_to_read, dtype="float32")

    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if len(signal) == 0:
        return None

    if file_sr != TARGET_SR:
        from scipy.signal import resample
        num_samples = int(len(signal) * TARGET_SR / file_sr)
        if num_samples <= 0:
            return None
        signal = resample(signal, num_samples).astype(np.float32)

    return signal


def extract_embedding(audio_path, start, end, layer=DEFAULT_LAYER,
                      model_name=DEFAULT_MODEL, use_gpu=False):
    """提取单个 IPU 的 wav2vec2 嵌入 (指定层, 时间维均值池化)。

    Args:
        audio_path: 音频文件路径。
        start, end: 片段起止时间 (秒)。
        layer: hidden_states 索引 (0=特征投影, 1..N=各 Transformer 层)。偏声学取低层。
        model_name: HuggingFace 模型名或本地路径。
        use_gpu: 是否使用 GPU。

    Returns:
        一维 np.ndarray (hidden_size,); 提取失败返回 None。
    """
    try:
        import torch

        signal = _read_segment(audio_path, start, end)
        if signal is None:
            return None

        model, device = _get_model(model_name, use_gpu)

        # wav2vec2-base 帧率 ~50Hz, 过短片段没有有效帧
        if len(signal) < TARGET_SR * 0.04:  # < 40ms
            return None

        input_values = torch.from_numpy(signal).float().unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_values)

        hidden_states = outputs.hidden_states  # tuple, len = num_layers + 1
        n_layers = len(hidden_states)
        layer_idx = max(0, min(layer, n_layers - 1))
        if layer_idx != layer:
            logger.warning("layer=%d 超出范围 [0,%d], 已截断为 %d",
                           layer, n_layers - 1, layer_idx)

        # (1, T, H) -> 时间维均值池化 -> (H,)
        emb = hidden_states[layer_idx].squeeze(0).mean(dim=0)
        return emb.cpu().numpy().astype(np.float64)

    except Exception as e:
        logger.warning("wav2vec2 嵌入提取失败 (%.2f-%.2f): %s", start, end, e)
        return None


def extract_embeddings_for_segments(audio_path, segments, layer=DEFAULT_LAYER,
                                    model_name=DEFAULT_MODEL, use_gpu=False,
                                    callback=None):
    """批量提取一组 IPU 的嵌入。

    Args:
        audio_path: 音频文件路径。
        segments: 片段列表, 每项含 'start' / 'end'。
        layer / model_name / use_gpu: 见 extract_embedding。
        callback: 可选 callable(done, total)。

    Returns:
        np.ndarray, shape (n_seg, hidden_size); 某句提取失败则该行全为 NaN。
        若全部失败或模型不可用, 返回 shape (n_seg, 0) 的空数组。
    """
    embs = []
    hidden = None
    total = len(segments)
    for i, seg in enumerate(segments):
        emb = extract_embedding(audio_path, seg["start"], seg["end"],
                                layer=layer, model_name=model_name, use_gpu=use_gpu)
        if emb is not None and hidden is None:
            hidden = emb.shape[0]
        embs.append(emb)
        if callback:
            callback(i + 1, total)

    if hidden is None:
        # 全失败 (模型缺失 / 依赖缺失等)
        return np.full((total, 0), np.nan)

    out = np.full((total, hidden), np.nan, dtype=np.float64)
    for i, emb in enumerate(embs):
        if emb is not None and emb.shape[0] == hidden:
            out[i] = emb
    return out


def _cosine_distance(vec, centroid):
    """cosine 距离 = 1 - cosine 相似度; 任一为零向量/NaN 返回 NaN。"""
    if vec is None or centroid is None:
        return np.nan
    if np.any(np.isnan(vec)) or np.any(np.isnan(centroid)):
        return np.nan
    nv = np.linalg.norm(vec)
    nc = np.linalg.norm(centroid)
    if nv < 1e-12 or nc < 1e-12:
        return np.nan
    return float(1.0 - np.dot(vec, centroid) / (nv * nc))


def compute_centroid_distance(embeddings, speakers, mode="global",
                              warmup=DEFAULT_WARMUP):
    """计算每个 IPU 到其说话人质心的 cosine 距离 (标量序列)。

    embeddings 的行顺序应已按时间排序 (expanding 模式依赖此顺序)。

    Args:
        embeddings: np.ndarray (n, hidden); 失败行可为全 NaN。
        speakers: 长度 n 的说话人标签序列 (list / np.ndarray / Series)。
        mode:
            'global'    全局质心 (该说话人全部有效 IPU 的均值, 会"看未来");
            'expanding' 扩张窗口因果质心 (仅用当前 IPU **之前**的历史均值, 不泄漏)。
        warmup: 仅 expanding 模式生效; 历史有效句数 < warmup 时距离记为 NaN。

    Returns:
        np.ndarray (n,) 的 cosine 距离, 含 NaN。
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    speakers = np.asarray(speakers)
    n = len(speakers)
    dist = np.full(n, np.nan, dtype=np.float64)

    if embeddings.ndim != 2 or embeddings.shape[1] == 0 or embeddings.shape[0] != n:
        return dist  # 无有效嵌入

    valid = ~np.any(np.isnan(embeddings), axis=1)  # 该行嵌入是否有效

    if mode == "global":
        for spk in np.unique(speakers):
            idx = np.where((speakers == spk) & valid)[0]
            if len(idx) == 0:
                continue
            centroid = embeddings[idx].mean(axis=0)
            for i in idx:
                dist[i] = _cosine_distance(embeddings[i], centroid)
        return dist

    if mode == "expanding":
        # 按当前 (时间) 顺序逐说话人维护历史; 仅用过去的有效嵌入
        history = {}  # spk -> {'sum': vec, 'count': int}
        for i in range(n):
            spk = speakers[i]
            h = history.get(spk)
            if h is not None and h["count"] >= warmup:
                centroid = h["sum"] / h["count"]
                dist[i] = _cosine_distance(embeddings[i], centroid)
            # 用完当前句后再并入历史 (保证因果: 不含当前句)
            if valid[i]:
                if h is None:
                    history[spk] = {"sum": embeddings[i].copy(), "count": 1}
                else:
                    h["sum"] += embeddings[i]
                    h["count"] += 1
        return dist

    raise ValueError(f"未知 mode: {mode!r} (应为 'global' 或 'expanding')")


def add_embedding_features(df, audio_path, layer=DEFAULT_LAYER,
                           model_name=DEFAULT_MODEL, use_gpu=False,
                           speaker_col="speaker", time_col="start",
                           warmup=DEFAULT_WARMUP, embeddings=None,
                           callback=None):
    """为单个对话的特征 DataFrame 追加两个质心距离列 (就地返回新 df)。

    新增列:
        embedding_cosine_dist_global    : 全局质心距离 (原论文口径)
        embedding_cosine_dist_expanding : 扩张窗口因果质心距离 (推荐主分析)

    Args:
        df: 单文件特征 DataFrame, 至少含 speaker_col / time_col / 'start' / 'end'。
        audio_path: 对应音频路径。
        layer / model_name / use_gpu: 见 extract_embedding。
        warmup: 扩张窗口预热句数。
        embeddings: 可选, 预先算好的 (n, hidden) 嵌入 (跳过重复提取, 利于对照实验)。
        callback: 可选 callable(done, total) 提取进度。

    Returns:
        新的 DataFrame (按 time_col 排序), 含上述两列; 提取不可用时两列为 NaN。
    """
    import pandas as pd

    df = df.sort_values(time_col).reset_index(drop=True)

    if embeddings is None:
        segments = df[["start", "end"]].to_dict("records")
        embeddings = extract_embeddings_for_segments(
            audio_path, segments, layer=layer, model_name=model_name,
            use_gpu=use_gpu, callback=callback)

    speakers = df[speaker_col].values
    df["embedding_cosine_dist_global"] = compute_centroid_distance(
        embeddings, speakers, mode="global")
    df["embedding_cosine_dist_expanding"] = compute_centroid_distance(
        embeddings, speakers, mode="expanding", warmup=warmup)
    return df
