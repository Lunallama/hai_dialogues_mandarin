"""Whisper ASR引擎 - 使用faster-whisper实现"""

import os
import sys
import logging
from typing import List, Dict, Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


def get_bundled_model_path() -> Optional[str]:
    """获取打包内置的 whisper 模型路径

    PyInstaller 打包后, 资源解压到 sys._MEIPASS 目录;
    开发模式下, 查找项目根目录的 whisper/ 文件夹。

    Returns:
        模型目录的绝对路径, 若不存在则返回 None
    """
    candidates = []

    # 1) PyInstaller 打包环境
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(os.path.join(meipass, 'whisper'))

    # 2) 开发环境: 项目根目录 / whisper
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.append(os.path.join(project_root, 'whisper'))

    for path in candidates:
        model_bin = os.path.join(path, 'model.bin')
        if os.path.isfile(model_bin):
            logger.info(f"检测到内置 whisper 模型: {path}")
            return path

    return None

# 在导入 torch 之前设置环境变量，避免 Windows DLL 加载问题
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')


class WhisperEngine:
    """基于faster-whisper的ASR引擎"""

    # 用于断句的标点符号
    SENTENCE_ENDINGS = set('。！？.!?')

    def __init__(self, model_size: str = 'base', language: Optional[str] = None,
                 hf_token: Optional[str] = None, use_gpu: bool = False,
                 model_path: Optional[str] = None,
                 silence_threshold_ms: int = 100,
                 vad_min_silence_ms: int = 200):
        """初始化Whisper引擎

        Args:
            model_size: 模型大小, 可选 tiny/base/small/medium/large
            language: 语言代码, 'zh'/'en'/None(自动检测)
            hf_token: HuggingFace token (用于下载模型)
            use_gpu: 是否使用GPU加速 (需要NVIDIA GPU + CUDA)
            model_path: 本地模型文件夹路径 (若提供则不从网络下载)
            silence_threshold_ms: 断句静音阈值(毫秒), 词间静音超过此值则断句
            vad_min_silence_ms: VAD最小静音段长度(毫秒)
        """
        self.model_size = model_size
        self.language = language
        self.hf_token = hf_token
        self.use_gpu = use_gpu
        self.model_path = model_path
        self.silence_threshold = silence_threshold_ms / 1000.0  # 转为秒
        self.vad_min_silence_ms = vad_min_silence_ms
        self.model = None

    def _ensure_model(self):
        """延迟加载模型"""
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "请安装faster-whisper: pip install faster-whisper"
                )

            # 根据用户选择决定设备
            if self.use_gpu:
                device, compute_type = "cuda", "float16"
            else:
                device, compute_type = "cpu", "int8"

            # 确定模型来源：用户指定路径 > 内置模型 > 在线模型名
            model_id = self.model_path
            if not model_id:
                bundled = get_bundled_model_path()
                if bundled:
                    model_id = bundled
                    logger.info(f"使用内置 whisper 模型: {bundled}")
            if not model_id:
                model_id = self.model_size

            logger.info(f"加载Whisper模型: {model_id} (device={device}, compute={compute_type})")

            try:
                self.model = WhisperModel(
                    model_id,
                    device=device,
                    compute_type=compute_type
                )
            except Exception as e:
                if device != "cpu":
                    logger.warning(f"GPU加载失败({e}), 回退到CPU模式")
                    self.model = WhisperModel(
                        model_id,
                        device="cpu",
                        compute_type="int8"
                    )
                else:
                    raise

    @staticmethod
    def _detect_device() -> tuple:
        """检测可用设备，返回 (device, compute_type)"""
        return "cpu", "int8"

    def transcribe(self, audio_path: str,
                   callback: Optional[Callable[[float, str], None]] = None) -> List[Dict]:
        """转录音频文件

        Args:
            audio_path: 音频文件路径
            callback: 进度回调函数 callback(progress_pct, message)

        Returns:
            分段列表: [{start: float, end: float, text: str}, ...]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        self._ensure_model()

        if callback:
            callback(0.0, "开始转录...")

        # 执行转录
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=self.vad_min_silence_ms,
                speech_pad_ms=100
            )
        )

        if callback:
            callback(10.0, f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")

        # 收集所有word级别的时间戳
        all_words = []
        segment_list = list(segments_gen)
        total_segments = len(segment_list) if segment_list else 1

        for idx, segment in enumerate(segment_list):
            if segment.words:
                for word in segment.words:
                    all_words.append({
                        'start': word.start,
                        'end': word.end,
                        'text': word.word.strip()
                    })
            else:
                # 没有word级别时间戳时，使用segment级别
                all_words.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })

            if callback:
                progress = 10.0 + 80.0 * (idx + 1) / total_segments
                callback(progress, f"处理段落 {idx + 1}/{total_segments}")

        if not all_words:
            logger.warning("未检测到任何语音内容")
            return []

        # 将words组合为句子
        sentences = self._group_words_to_sentences(all_words)

        if callback:
            callback(100.0, f"转录完成, 共 {len(sentences)} 个句子")

        return sentences

    def _group_words_to_sentences(self, words: List[Dict]) -> List[Dict]:
        """将词组合为句子

        使用标点符号作为断句依据，若无标点则使用静音间隔断句。

        Args:
            words: word级别分段列表

        Returns:
            句子级别分段列表
        """
        if not words:
            return []

        sentences = []
        current_sentence_words = []
        current_start = words[0]['start']

        for i, word in enumerate(words):
            current_sentence_words.append(word)

            # 检查是否应该在此处断句
            should_break = False

            # 条件1: 文本以标点符号结尾
            text = word['text']
            if text and text[-1] in self.SENTENCE_ENDINGS:
                should_break = True

            # 条件2: 与下一个词之间存在较长静音
            if not should_break and i < len(words) - 1:
                gap = words[i + 1]['start'] - word['end']
                if gap > self.silence_threshold:
                    should_break = True

            # 条件3: 最后一个词
            if i == len(words) - 1:
                should_break = True

            if should_break and current_sentence_words:
                sentence_text = ''.join(
                    w['text'] for w in current_sentence_words
                )
                sentences.append({
                    'start': current_start,
                    'end': word['end'],
                    'text': sentence_text
                })
                current_sentence_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1]['start']

        return sentences


class SpeakerDiarizer:
    """基于pyannote.audio的说话人分离"""

    def __init__(self, hf_token: str):
        """初始化说话人分离模型

        Args:
            hf_token: HuggingFace token (pyannote需要授权)
        """
        self.hf_token = hf_token
        self.pipeline = None

    def _ensure_pipeline(self):
        """延迟加载diarization pipeline"""
        if self.pipeline is None:
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*torchcodec.*')
                    from pyannote.audio import Pipeline
            except ImportError:
                raise ImportError(
                    "请安装pyannote.audio: pip install pyannote.audio"
                )

            logger.info("加载pyannote diarization pipeline...")
            # 设置 HF token 环境变量，避免编码问题
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )

            # 尝试使用GPU（安全方式）
            try:
                import torch
                if torch.cuda.is_available():
                    torch.zeros(1, device='cuda')  # 测试CUDA真正可用
                    self.pipeline.to(torch.device("cuda"))
                    logger.info("pyannote 使用 CUDA")
            except Exception as e:
                logger.info(f"pyannote 使用 CPU ({e})")

    def diarize(self, audio_path: str, num_speakers: int = 2) -> List[Dict]:
        """执行说话人分离

        Args:
            audio_path: 音频文件路径
            num_speakers: 说话人数量

        Returns:
            分段列表: [{start: float, end: float, speaker_label: str}, ...]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        self._ensure_pipeline()

        # 用 soundfile 预加载音频为 waveform，彻底绕过 torchcodec/torchaudio 解码问题
        import torch
        import soundfile as sf
        logger.info(f"预加载音频: {audio_path}")
        data, sample_rate = sf.read(audio_path, dtype='float32')
        # soundfile 返回 (samples,) 或 (samples, channels)，转为 (channel, time)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        logger.info(f"开始说话人分离: {audio_path} (speakers={num_speakers})")
        diarization = self.pipeline(
            audio_input,
            num_speakers=num_speakers
        )

        segments = []
        # 兼容新版 pyannote (DiarizeOutput) 和旧版 (Annotation)
        annotation = diarization
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker_label': speaker
            })

        logger.info(f"说话人分离完成, 共 {len(segments)} 个片段")
        return segments

    def assign_speakers(self, transcription_segments: List[Dict],
                        diarization_segments: List[Dict]) -> List[Dict]:
        """将说话人标签分配给转录结果

        基于时间重叠比例进行匹配。

        Args:
            transcription_segments: 转录分段 [{start, end, text}, ...]
            diarization_segments: 分离分段 [{start, end, speaker_label}, ...]

        Returns:
            更新后的转录分段，增加 'speaker' 字段
        """
        if not diarization_segments:
            logger.warning("无说话人分离结果, 所有段落标记为 SPEAKER_00")
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_00'
            return transcription_segments

        result = []
        for t_seg in transcription_segments:
            t_start = t_seg['start']
            t_end = t_seg['end']
            t_duration = t_end - t_start

            if t_duration <= 0:
                t_seg['speaker'] = 'SPEAKER_00'
                result.append(t_seg)
                continue

            # 计算与每个diarization segment的重叠
            speaker_overlaps = {}
            for d_seg in diarization_segments:
                overlap_start = max(t_start, d_seg['start'])
                overlap_end = min(t_end, d_seg['end'])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > 0:
                    speaker = d_seg['speaker_label']
                    speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap

            # 选择重叠最多的说话人
            if speaker_overlaps:
                best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            else:
                # 没有重叠，选最近的说话人段落
                best_speaker = self._find_nearest_speaker(t_start, t_end, diarization_segments)

            t_seg_copy = dict(t_seg)
            t_seg_copy['speaker'] = best_speaker
            result.append(t_seg_copy)

        return result

    @staticmethod
    def _find_nearest_speaker(t_start: float, t_end: float,
                              diarization_segments: List[Dict]) -> str:
        """找到距离最近的说话人段落"""
        t_center = (t_start + t_end) / 2
        min_dist = float('inf')
        nearest_speaker = 'SPEAKER_00'

        for d_seg in diarization_segments:
            d_center = (d_seg['start'] + d_seg['end']) / 2
            dist = abs(t_center - d_center)
            if dist < min_dist:
                min_dist = dist
                nearest_speaker = d_seg['speaker_label']

        return nearest_speaker
