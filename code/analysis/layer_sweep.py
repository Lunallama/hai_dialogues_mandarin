# -*- coding: utf-8 -*-
"""层扫描稳健性检验：wav2vec2 不同层(1/4/7/10)对趋同测量是否敏感。
目的：回答"为什么取第 4 层"——证明趋同结论不依赖具体层号。
不改动 extract_features.py / compute_entrainment.py / wav2vec2_embedding.py / 正文；
结果单独存 analysis/layer_sweep/。

效率：每个 IPU 只做一次前向(output_hidden_states)，一次性取所有待测层，避免 4× 前向。
仅影响风格层(embedding 到本人质心距离)；韵律/MFCC 与层无关，不纳入。
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "analysis"))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from core.features.wav2vec2_embedding import (
    _read_segment, _get_model, compute_centroid_distance, DEFAULT_LAYER, DEFAULT_MODEL)
from compute_entrainment import synchrony_one, convergence_one  # 复用，不修改

SCENE_MAP = {"comfort": "comforting", "default": "neutral", "quarrel": "arguing"}
LAYERS = [1, 4, 7, 10]   # 低声学 / 本研究取值 / 中音素 / 偏高
OUT = os.path.join(_ROOT, "analysis", "layer_sweep")
os.makedirs(OUT, exist_ok=True)


def read_anno(p):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try: return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, UnicodeError): continue
    return pd.read_csv(p, encoding="utf-8", encoding_errors="replace")


def embeddings_multi_layer(wav, segs, layers):
    """单次前向取多层：返回 {layer: ndarray(n,H)}，失败行 NaN。"""
    import torch
    model, device = _get_model(DEFAULT_MODEL, use_gpu=False)
    n = len(segs)
    out = {L: None for L in layers}
    H = None
    rows = {L: [] for L in layers}
    for seg in segs:
        sig = _read_segment(wav, seg["start"], seg["end"])
        emb_by_layer = None
        if sig is not None and len(sig) >= 16000 * 0.04:
            iv = torch.from_numpy(sig).float().unsqueeze(0).to(device)
            with torch.no_grad():
                hs = model(iv).hidden_states   # tuple len = n_layers+1
            nl = len(hs)
            emb_by_layer = {}
            for L in layers:
                li = max(0, min(L, nl - 1))
                v = hs[li].squeeze(0).mean(dim=0).cpu().numpy().astype(np.float64)
                emb_by_layer[L] = v
                if H is None: H = v.shape[0]
        for L in layers:
            rows[L].append(emb_by_layer[L] if emb_by_layer is not None else None)
    if H is None:
        return {L: np.full((n, 0), np.nan) for L in layers}
    for L in layers:
        m = np.full((n, H), np.nan)
        for i, v in enumerate(rows[L]):
            if v is not None and v.shape[0] == H:
                m[i] = v
        out[L] = m
    return out


def entrain_one(g, feat):
    s = synchrony_one(g[g.role == "human"][feat].dropna().values,
                      g[g.role == "AI"][feat].dropna().values,
                      np.random.default_rng(42))
    c = convergence_one(g, feat, "sym", np.random.default_rng(42))
    return (s["sync_strength"] if s else np.nan,
            c["slope"] if c else np.nan,
            c["direction"] if c else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-dir", default=os.path.join(_ROOT, "annotation"))
    ap.add_argument("--wav-dir", default=r"E:/人机对话实验被试录音录频/wavfile")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    csvs = []
    for folder, scene in SCENE_MAP.items():
        for c in sorted(glob.glob(os.path.join(args.annotation_dir, folder, "*.csv"))):
            csvs.append((c, scene))
    csvs = csvs[::args.stride]
    print(f"子集对话数: {len(csvs)}  层: {LAYERS}", flush=True)

    rows, dist_corr = [], []
    for k, (csv_path, scene) in enumerate(csvs, 1):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        wav = os.path.join(args.wav_dir, name + ".wav")
        if not os.path.exists(wav):
            continue
        print(f"[{k}/{len(csvs)}] {name}", flush=True)
        df = read_anno(csv_path).rename(columns={"start_time": "start", "end_time": "end"})
        df = df.dropna(subset=["start", "end", "speaker"]).sort_values("start").reset_index(drop=True)
        spk = df["speaker"].astype(str).str.strip()
        role = np.where(spk.str.upper() == "AI", "AI", "human")
        segs = df[["start", "end"]].to_dict("records")
        embs = embeddings_multi_layer(wav, segs, LAYERS)

        base = pd.DataFrame({"file": os.path.basename(csv_path), "scene": scene,
                             "speaker": spk.values, "role": role, "start": df["start"].values})
        dist = {}
        for L in LAYERS:
            d = compute_centroid_distance(embs[L], spk.values, mode="global")
            dist[L] = d
            g = base.copy(); g[f"emb_L{L}"] = d
            sy, cv, dr = entrain_one(g, f"emb_L{L}")
            rows.append({"file": base["file"].iloc[0], "scene": scene, "layer": L,
                         "sync": sy, "conv": cv, "direction": dr})
        # 同段内各层距离序列与第 4 层的相关
        ref = dist[DEFAULT_LAYER]
        for L in LAYERS:
            if L == DEFAULT_LAYER: continue
            m = ~(np.isnan(ref) | np.isnan(dist[L]))
            if m.sum() > 3:
                dist_corr.append({"file": base["file"].iloc[0], "layer": L,
                                  "r_with_L4": np.corrcoef(ref[m], dist[L][m])[0, 1]})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "layer_sweep_results.csv"), index=False, encoding="utf-8-sig")
    dc = pd.DataFrame(dist_corr)
    dc.to_csv(os.path.join(OUT, "layer_dist_corr.csv"), index=False, encoding="utf-8-sig")

    lines = [f"层扫描子集: {res['file'].nunique()} 段对话；风格层(wav2vec2 到本人质心距离)\n",
             "各层趋同指标(均值跨对话)："]
    agg = res.groupby("layer").agg(
        sync_mean=("sync", "mean"), conv_mean=("conv", "mean"),
        conv_neg_rate=("conv", lambda x: (x < 0).mean()), n=("file", "nunique"))
    for L, r in agg.iterrows():
        tag = " ← 本研究" if L == DEFAULT_LAYER else ""
        lines.append(f"  L{L:<2d}  sync μ={r.sync_mean:.4f}  conv slope μ={r.conv_mean:+.5f}  "
                     f"趋近(slope<0)占比={r.conv_neg_rate:.0%}  n={int(r.n)}{tag}")
    # 跨层趋同指标相关(以对话为样本，L4 vs 其它)
    lines.append("\n各层与第 4 层的一致性：")
    piv_s = res.pivot_table(index="file", columns="layer", values="sync")
    piv_c = res.pivot_table(index="file", columns="layer", values="conv")
    for L in LAYERS:
        if L == DEFAULT_LAYER: continue
        rs = piv_s[DEFAULT_LAYER].corr(piv_s[L])
        rc = piv_c[DEFAULT_LAYER].corr(piv_c[L])
        rd = dc[dc.layer == L]["r_with_L4"].mean() if len(dc) else np.nan
        lines.append(f"  L4↔L{L}:  距离序列 r̄={rd:.3f}  |  sync r={rs:.3f}  |  conv r={rc:.3f}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "layer_sweep_summary.txt"), "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\n结果存于", OUT)


if __name__ == "__main__":
    main()
