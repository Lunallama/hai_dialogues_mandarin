"""独立特征提取脚本（与论文 3.2 节严格对齐）

输入
----
- 标注 CSV：<annotation-dir>/{comfort,default,quarrel}/*.csv
  列：start_time,end_time,speaker,text,sentence_type
  speaker：真人为 "序号+性别"（如 01f / 10m），AI 为 "AI"
- 音频 WAV：<wav-dir>/<同名>.wav

输出（逐 IPU 一行，与 3.2 节描述一一对应）
-----------------------------------------
韵律（Parselmouth/Praat）：
    pitch_mean/max/min/std, intensity_mean/max/min/std, duration, speech_rate
音色 / 风格（派生为"到本人质心"的标量，口径与 wav2vec2 一致）：
    mfcc_centroid_dist          —— 逐 IPU 13 维 MFCC 均值 → 说话人全局质心 → cosine 距离
    embedding_cosine_dist       —— wav2vec2 低层嵌入 → 说话人全局质心 → cosine 距离

成功判据：最终 CSV 在上述特征列上无空值（提取失败的 IPU 已剔除）。

用法
----
    python analysis/extract_features.py \
        --annotation-dir "D:/HAI Entrainment/annotation" \
        --wav-dir "E:/人机对话实验被试录音录频/wavfile" \
        --out "D:/HAI Entrainment/analysis/features_all.csv"
    # 调试：--limit 2 仅处理前 2 个文件
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.features.pitch_intensity import extract_pitch_features      # noqa: E402
from core.features.mfcc import extract_mfcc                           # noqa: E402
from core.features.speech_rate import calculate_speech_rate          # noqa: E402
from core.features.wav2vec2_embedding import (                       # noqa: E402
    extract_embeddings_for_segments, compute_centroid_distance, DEFAULT_LAYER)

SCENE_MAP = {"comfort": "comforting", "default": "neutral", "quarrel": "arguing"}
FNAME_SCENE = {"安慰": "comforting", "中性": "neutral", "吵架": "arguing"}

PROSODY_FEATURES = [
    "pitch_mean", "pitch_max", "pitch_min", "pitch_std",
    "intensity_mean", "intensity_max", "intensity_min", "intensity_std",
    "speech_rate",
]
FEATURE_COLS = PROSODY_FEATURES + ["mfcc_centroid_dist", "embedding_cosine_dist"]
MFCC_RAW = [f"mfcc_{i + 1}" for i in range(13)]


def extract_intensity_with_std(audio_path, start, end):
    """强度统计（含标准差）——core 模块缺 std，此处补齐以对齐正文。"""
    nan = {"intensity_mean": np.nan, "intensity_max": np.nan,
           "intensity_min": np.nan, "intensity_std": np.nan}
    try:
        snd = parselmouth.Sound(audio_path).extract_part(start, end)
        it = snd.to_intensity()
        return {
            "intensity_mean": float(call(it, "Get mean", 0, 0, "dB")),
            "intensity_max": float(call(it, "Get maximum", 0, 0, "Parabolic")),
            "intensity_min": float(call(it, "Get minimum", 0, 0, "Parabolic")),
            "intensity_std": float(call(it, "Get standard deviation", 0, 0)),
        }
    except Exception:
        return nan


def parse_meta(csv_name, speaker_raw):
    """从文件名（场景-性别-序号-姓名）与 speaker 码解析元信息。"""
    base = os.path.splitext(csv_name)[0]
    parts = base.split("-")
    fname_scene = FNAME_SCENE.get(parts[0], "")
    gender = parts[1] if len(parts) > 1 else ""
    subject_id = parts[2] if len(parts) > 2 else ""
    role = "AI" if str(speaker_raw).strip().upper() == "AI" else "human"
    return fname_scene, gender, subject_id, role


def process_file(csv_path, wav_path, scene, layer=DEFAULT_LAYER, verbose=True):
    csv_name = os.path.basename(csv_path)
    df = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"start_time": "start", "end_time": "end"})
    df = df.dropna(subset=["start", "end", "speaker"]).reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        start, end = float(r["start"]), float(r["end"])
        spk = str(r["speaker"]).strip()
        text = str(r["text"]) if pd.notna(r.get("text")) else ""
        fname_scene, gender, sid, role = parse_meta(csv_name, spk)
        row = {
            "file": csv_name, "scene": scene, "fname_scene": fname_scene,
            "subject_id": sid, "gender": gender, "role": role, "speaker": spk,
            "start": start, "end": end, "duration": end - start, "text": text,
        }
        row.update(extract_pitch_features(wav_path, start, end))
        row.update(extract_intensity_with_std(wav_path, start, end))
        row.update(extract_mfcc(wav_path, start, end))             # mfcc_1..13
        row["speech_rate"] = calculate_speech_rate(text, start, end, language="zh")
        rows.append(row)

    fdf = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

    # —— wav2vec2 嵌入 → 说话人全局质心 → cosine 距离 ——
    segs = fdf[["start", "end"]].to_dict("records")
    emb = extract_embeddings_for_segments(
        wav_path, segs, layer=layer,
        callback=(lambda d, t: (d % 25 == 0 or d == t) and
                  print(f"    embedding {d}/{t}", flush=True)) if verbose else None)
    fdf["embedding_cosine_dist"] = compute_centroid_distance(
        emb, fdf["speaker"].values, mode="global")

    # —— MFCC 13 维 → 说话人全局质心 → cosine 距离（与嵌入同法）——
    fdf["mfcc_centroid_dist"] = compute_centroid_distance(
        fdf[MFCC_RAW].values, fdf["speaker"].values, mode="global")

    return fdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-dir", default=os.path.join(_ROOT, "annotation"))
    ap.add_argument("--wav-dir", default=r"E:/人机对话实验被试录音录频/wavfile")
    ap.add_argument("--out", default=os.path.join(_ROOT, "analysis", "features_all.csv"))
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件（调试）")
    ap.add_argument("--shard", default="", help="并行分片 'i/n'：仅处理 index%%n==i 的文件")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    args = ap.parse_args()

    csvs = []
    for folder, scene in SCENE_MAP.items():
        for c in sorted(glob.glob(os.path.join(args.annotation_dir, folder, "*.csv"))):
            csvs.append((c, scene, folder))
    if args.limit:
        csvs = csvs[:args.limit]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        csvs = [c for k, c in enumerate(csvs) if k % n == i]
    print(f"待处理标注文件: {len(csvs)}", flush=True)

    all_parts, missing_wav, mismatches = [], [], []
    for k, (csv_path, scene, folder) in enumerate(csvs, 1):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        wav = os.path.join(args.wav_dir, name + ".wav")
        print(f"[{k}/{len(csvs)}] {name}  (scene={scene})", flush=True)
        if not os.path.exists(wav):
            missing_wav.append(name)
            print("    !! 缺少对应 wav，跳过", flush=True)
            continue
        fdf = process_file(csv_path, wav, scene)
        # 文件名场景 vs 文件夹场景 不一致预警
        fs = fdf["fname_scene"].iloc[0] if len(fdf) else ""
        if fs and fs != scene:
            mismatches.append((name, folder, fs))
        all_parts.append(fdf)

    feat = pd.concat(all_parts, ignore_index=True)
    n_raw = len(feat)

    # —— 剔除特征提取失败（任一特征列为空）的 IPU，保证无空值 ——
    feat_clean = feat.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    keep = ["file", "scene", "subject_id", "gender", "role", "speaker",
            "start", "end", "duration", "text"] + FEATURE_COLS
    feat_clean = feat_clean[keep]
    feat_clean.to_csv(args.out, index=False, encoding="utf-8-sig")

    # —— 报告 ——
    print("\n========== 汇总 ==========", flush=True)
    print(f"原始 IPU 行数         : {n_raw}")
    print(f"剔除空值后行数        : {len(feat_clean)} (剔除 {n_raw - len(feat_clean)})")
    print(f"特征列空值检查        : {int(feat_clean[FEATURE_COLS].isna().sum().sum())} (应为 0)")
    print(f"按场景计数            :\n{feat_clean['scene'].value_counts().to_string()}")
    print(f"按角色计数            :\n{feat_clean['role'].value_counts().to_string()}")
    print(f"对话文件数            : {feat_clean['file'].nunique()}")
    if missing_wav:
        print(f"缺少 wav 的标注({len(missing_wav)}): {missing_wav}")
    if mismatches:
        print(f"文件名/文件夹场景不一致: {mismatches}")
    print(f"输出: {args.out}", flush=True)


if __name__ == "__main__":
    main()
