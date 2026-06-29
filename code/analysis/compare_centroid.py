# -*- coding: utf-8 -*-
"""对比实验：global vs expanding 质心对趋同测量是否有显著差异。
不改动 extract_features.py / compute_entrainment.py / 正文；结果单独存 analysis/centroid_compare/。

仅影响两个风格特征（embedding、mfcc 的到本人质心距离）；韵律特征与质心口径无关，故不纳入。
在子集对话上重提嵌入，分别用 global 与 expanding(因果, warmup=5) 计算质心距离，
再用现有 synchrony_one / convergence_one(sym) 计算趋同，做配对检验（Wilcoxon）。
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "analysis"))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from core.features.mfcc import extract_mfcc
from core.features.wav2vec2_embedding import (
    extract_embeddings_for_segments, compute_centroid_distance, DEFAULT_LAYER)
from compute_entrainment import synchrony_one, convergence_one  # 复用，不修改

SCENE_MAP = {"comfort": "comforting", "default": "neutral", "quarrel": "arguing"}
MFCC_RAW = [f"mfcc_{i+1}" for i in range(13)]
OUT = os.path.join(_ROOT, "analysis", "centroid_compare")
os.makedirs(OUT, exist_ok=True)


def read_anno(p):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try: return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, UnicodeError): continue
    return pd.read_csv(p, encoding="utf-8", encoding_errors="replace")


def build_file(csv_path, wav, scene):
    df = read_anno(csv_path).rename(columns={"start_time": "start", "end_time": "end"})
    df = df.dropna(subset=["start", "end", "speaker"]).sort_values("start").reset_index(drop=True)
    spk = df["speaker"].astype(str).str.strip()
    role = np.where(spk.str.upper() == "AI", "AI", "human")
    segs = df[["start", "end"]].to_dict("records")
    emb = extract_embeddings_for_segments(wav, segs, layer=DEFAULT_LAYER)
    mfcc = np.full((len(df), 13), np.nan)
    for i, r in df.iterrows():
        m = extract_mfcc(wav, float(r["start"]), float(r["end"]))
        mfcc[i] = [m[c] for c in MFCC_RAW]
    out = pd.DataFrame({
        "file": os.path.basename(csv_path), "scene": scene,
        "speaker": spk.values, "role": role, "start": df["start"].values,
        "emb_global":  compute_centroid_distance(emb,  spk.values, mode="global"),
        "emb_expand":  compute_centroid_distance(emb,  spk.values, mode="expanding"),
        "mfcc_global": compute_centroid_distance(mfcc, spk.values, mode="global"),
        "mfcc_expand": compute_centroid_distance(mfcc, spk.values, mode="expanding"),
    })
    return out


def entrain_one(g, feat):
    h = g[g.role == "human"][feat].dropna().values
    a = g[g.role == "AI"][feat].dropna().values
    s = synchrony_one(h, a, np.random.default_rng(42))
    c = convergence_one(g, feat, "sym", np.random.default_rng(42))
    return (s["sync_strength"] if s else np.nan,
            c["slope"] if c else np.nan,
            c["direction"] if c else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-dir", default=os.path.join(_ROOT, "annotation"))
    ap.add_argument("--wav-dir", default=r"E:/人机对话实验被试录音录频/wavfile")
    ap.add_argument("--stride", type=int, default=3, help="每隔几个文件取一个（控制子集规模）")
    args = ap.parse_args()

    csvs = []
    for folder, scene in SCENE_MAP.items():
        for c in sorted(glob.glob(os.path.join(args.annotation_dir, folder, "*.csv"))):
            csvs.append((c, scene))
    csvs = csvs[::args.stride]
    print(f"子集对话数: {len(csvs)}", flush=True)

    rows = []
    for k, (csv_path, scene) in enumerate(csvs, 1):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        wav = os.path.join(args.wav_dir, name + ".wav")
        if not os.path.exists(wav):
            continue
        print(f"[{k}/{len(csvs)}] {name}", flush=True)
        g = build_file(csv_path, wav, scene)
        for base in ["emb", "mfcc"]:
            sg, cg, dg = entrain_one(g, f"{base}_global")
            se, ce, de = entrain_one(g, f"{base}_expand")
            rows.append({"file": g["file"].iloc[0], "scene": scene, "feature": base,
                         "sync_global": sg, "sync_expand": se,
                         "conv_global": cg, "conv_expand": ce,
                         "dir_global": dg, "dir_expand": de})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "centroid_compare_results.csv"), index=False, encoding="utf-8-sig")

    # ---- 配对检验 ----
    lines = [f"对比子集: {res['file'].nunique()} 段对话；每段 2 个风格特征(emb/mfcc)\n"]
    for base in ["emb", "mfcc"]:
        sub = res[res.feature == base]
        for metric in ["sync", "conv"]:
            g_, e_ = sub[f"{metric}_global"].dropna(), sub[f"{metric}_expand"].dropna()
            both = sub.dropna(subset=[f"{metric}_global", f"{metric}_expand"])
            x, y = both[f"{metric}_global"], both[f"{metric}_expand"]
            try:
                w, p = stats.wilcoxon(x, y)
            except Exception:
                w, p = np.nan, np.nan
            r = x.corr(y)
            lines.append(f"[{base} · {metric}] n={len(both)}  "
                         f"global μ={x.mean():.4f}  expand μ={y.mean():.4f}  "
                         f"Δμ={x.mean()-y.mean():+.4f}  Wilcoxon p={p:.4f}  r(global,expand)={r:.3f}")
        # 方向一致性
        sub2 = sub.dropna(subset=["dir_global", "dir_expand"])
        agree = (sub2["dir_global"] == sub2["dir_expand"]).mean() if len(sub2) else np.nan
        lines.append(f"[{base}] 趋近方向(趋近/发散/无)判定一致率 = {agree:.2%}\n")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "centroid_compare_summary.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("结果存于", OUT)


if __name__ == "__main__":
    main()
