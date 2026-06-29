"""独立趋同计算脚本（3.3 同步 + 3.4 趋近）

对齐 81 号论文（Li et al., 2026）的同步/趋近测量，并做两处关键扩展：
  · 不把 AI 当"无趋同基线"——用 role 列区分 human / AI；
  · 同步：滞后相关用带符号 lag（>0 人跟随 AI=人→AI；<0 AI 跟随人=AI→人），
          报告两方向占比，回答"谁在跟随"；
  · 趋近：除 81 号的对称口径 |AI−human|（整体差距随时间缩小）外，
          另算【有向】趋近——
            人→AI：人当前 IPU 到"AI 上一句"的距离随时间斜率；
            AI→人：AI 当前 IPU 到"人上一句"的距离随时间斜率；
          斜率越负＝该方向趋近越强，据此比较人→AI 与 AI→人谁更强。

输入：特征 CSV（或含多场景的文件夹），需含列 file/role/speaker/start + 特征列。
输出：synchrony_results.csv / convergence_results.csv（逐 file×feature）。

用法：
    python analysis/compute_entrainment.py --features analysis/features_sample.csv --outdir analysis
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

FEATURES_DEFAULT = [
    "pitch_mean", "pitch_max", "pitch_min", "pitch_std",
    "intensity_mean", "intensity_max", "intensity_min", "intensity_std",
    "speech_rate", "mfcc_centroid_dist", "embedding_cosine_dist",
]
WIN = 10          # 同步窗口 IPU 数（81 号）
MAX_LAG = 5       # 滞后范围 ±5（81 号取 0..5；此处用带符号以判断方向）
N_CTRL = 10       # 随机对照段数（81 号）
N_PERM = 1000     # 趋近置换次数（81 号）
SMOOTH = 5        # 趋近差值序列平滑点数（81 号）


# ----------------------------- 同步 (3.3) -----------------------------
def lagged_peak(a, b, max_lag=MAX_LAG):
    """对两等长序列在 lag∈[-max_lag,max_lag] 求绝对值最大的 Pearson r。
    返回 (peak_abs_r, peak_lag_signed)。lag>0: b 领先 a（a 跟随 b）。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    best_abs, best_lag, best_signed = 0.0, 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = a[lag:n]; y = b[:n - lag] if lag > 0 else b[:n]
        else:
            x = a[:n + lag]; y = b[-lag:n]
        m = min(len(x), len(y))
        if m < 3:
            continue
        x, y = x[:m], y[:m]
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            continue
        r, _ = stats.pearsonr(x, y)
        if np.isfinite(r) and abs(r) > best_abs:
            best_abs, best_lag, best_signed = abs(r), lag, r
    return best_abs, best_lag, best_signed


def window_peaks(h, a, win=WIN, max_lag=MAX_LAG):
    """把人/AI 序列对齐到等长后切 win-IPU 窗口，逐窗求峰值。
    返回 list[(abs_r, lag_signed, r_signed)]。"""
    n = min(len(h), len(a))
    h, a = h[:n], a[:n]
    out = []
    for s in range(0, n - win + 1, win):
        ar, lg, sg = lagged_peak(h[s:s + win], a[s:s + win], max_lag)
        if ar > 0:
            out.append((ar, lg, sg))
    return out


def synchrony_one(h, a, rng):
    """单 file×feature 的同步分析：真实窗口峰值 vs 10 段随机对照；方向占比。"""
    real = window_peaks(h, a)
    if len(real) < 2:
        return None
    real_r = np.array([x[0] for x in real])
    lags = np.array([x[1] for x in real])
    signs = np.array([x[2] for x in real])
    s_sync_real = real_r / (1.0 + np.abs(lags))        # 滞后惩罚同步分数（EI 用）

    # 随机对照：打乱人类序列顺序，每段 1 组窗口峰值，合并 N_CTRL 段
    ctrl_r, ctrl_s = [], []
    for _ in range(N_CTRL):
        hp = rng.permutation(h)
        for ar, lg, sg in window_peaks(hp, a):
            ctrl_r.append(ar)
            ctrl_s.append(ar / (1.0 + abs(lg)))
    ctrl_r = np.array(ctrl_r) if ctrl_r else np.array([0.0])
    ctrl_s = np.array(ctrl_s) if ctrl_s else np.array([0.0])

    # Shapiro–Wilk → t 检验 / Mann–Whitney
    try:
        norm_ok = (len(real_r) >= 3 and len(ctrl_r) >= 3 and
                   stats.shapiro(real_r)[1] > 0.05 and stats.shapiro(ctrl_r)[1] > 0.05)
    except Exception:
        norm_ok = False
    if norm_ok:
        _, p = stats.ttest_ind(real_r, ctrl_r, equal_var=False); test = "t"
    else:
        try:
            _, p = stats.mannwhitneyu(real_r, ctrl_r, alternative="greater"); test = "MW"
        except Exception:
            p, test = np.nan, "NA"

    return {
        "n_windows": len(real),
        "sync_strength": float(np.mean(real_r)),       # 平均峰值|r|
        "ctrl_strength": float(np.mean(ctrl_r)),
        "s_sync_real": float(np.mean(s_sync_real)),    # 滞后惩罚同步分数（EI 用）
        "s_sync_rand_mean": float(np.mean(ctrl_s)),
        "s_sync_rand_std": float(np.std(ctrl_s)),
        "p_value": float(p) if p == p else np.nan,
        "test": test,
        "significant": bool(p < 0.05) if p == p else False,
        "pos_sync_ratio": float(np.mean(signs > 0)),   # 正向同步占比
        "neg_sync_ratio": float(np.mean(signs < 0)),   # 反向同步（去同化）占比
        "human_follow_ratio": float(np.mean(lags > 0)),  # 人→AI（人跟随）
        "ai_follow_ratio": float(np.mean(lags < 0)),     # AI→人（AI 跟随）
        "simultaneous_ratio": float(np.mean(lags == 0)),
        "mean_abs_lag": float(np.mean(np.abs(lags))),
    }


# ----------------------------- 趋近 (3.4) -----------------------------
def smooth(x, k=SMOOTH):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def slope_of(times, diffs):
    times = np.asarray(times, float); diffs = np.asarray(diffs, float)
    if len(times) < 3 or np.std(times) < 1e-10 or np.std(diffs) < 1e-10:
        return np.nan
    return stats.linregress(times, diffs).slope


def _mode_pairs(roles, times, mode):
    """预计算某口径下的(当前,前一)跨说话人配对索引与时间中点（不依赖特征值，置换间复用）。"""
    cur, prev = [], []
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            continue
        if mode == "h2a" and not (roles[i] == "human" and roles[i - 1] == "AI"):
            continue
        if mode == "a2h" and not (roles[i] == "AI" and roles[i - 1] == "human"):
            continue
        cur.append(i); prev.append(i - 1)
    cur = np.array(cur, int); prev = np.array(prev, int)
    T = (times[cur] + times[prev]) / 2.0 if len(cur) else np.array([])
    return cur, prev, T


def convergence_one(g, feat, mode, rng):
    """单 file×feature×方向的趋近：5点平滑→回归斜率→置换检验（纯 numpy，置换仅打乱目标说话人特征值）。"""
    roles = g["role"].values.astype(str)
    vals = g[feat].values.astype(float)
    times = g["start"].values.astype(float)
    cur, prev, T = _mode_pairs(roles, times, mode)
    if len(cur) < SMOOTH + 2:
        return None

    def slope_from(v):
        D = np.abs(v[cur] - v[prev])
        ok = ~np.isnan(D)
        if ok.sum() < SMOOTH + 2:
            return np.nan
        return slope_of(smooth(T[ok]), smooth(D[ok]))

    obs = slope_from(vals)
    if np.isnan(obs):
        return None

    shuffle_role = {"sym": "human", "h2a": "human", "a2h": "AI"}[mode]
    mask = roles == shuffle_role
    base = vals.copy()
    sub = base[mask]
    perm = np.empty(N_PERM)
    for k in range(N_PERM):
        v = base.copy()
        v[mask] = rng.permutation(sub)
        perm[k] = slope_from(v)
    perm = perm[np.isfinite(perm)]
    p_conv = float(np.mean(perm <= obs)) if len(perm) else np.nan   # 趋近=负斜率
    if obs < 0 and p_conv < 0.05:
        direction = "convergence"
    elif obs > 0 and len(perm) and float(np.mean(perm >= obs)) < 0.05:
        direction = "divergence"
    else:
        direction = "none"
    return {"slope": float(obs), "p_value": p_conv, "direction": direction,
            "n_pairs": int(np.sum(~np.isnan(np.abs(vals[cur] - vals[prev])))),
            "slope_rand_mean": float(np.mean(perm)) if len(perm) else np.nan,
            "slope_rand_std": float(np.std(perm)) if len(perm) else np.nan}


# ----------------------------- 主流程 -----------------------------
def run(features_csv, outdir, features, seed=42):
    df = pd.read_csv(features_csv)
    rng = np.random.default_rng(seed)
    feats = [f for f in features if f in df.columns]

    sync_rows, conv_rows = [], []
    for file_id, g in df.groupby("file"):
        g = g.sort_values("start").reset_index(drop=True)
        if g["role"].nunique() < 2:
            continue
        h = g[g["role"] == "human"].sort_values("start")
        a = g[g["role"] == "AI"].sort_values("start")
        scene = g["scene"].iloc[0]
        for feat in feats:
            # 同步
            sh = h[feat].dropna().values
            sa = a[feat].dropna().values
            sres = synchrony_one(sh, sa, np.random.default_rng(seed))
            if sres:
                sync_rows.append({"file": file_id, "scene": scene, "feature": feat, **sres})
            # 趋近（三口径）
            for mode in ("sym", "h2a", "a2h"):
                cres = convergence_one(g, feat, mode, np.random.default_rng(seed))
                if cres:
                    conv_rows.append({"file": file_id, "scene": scene, "feature": feat,
                                      "mode": mode, **cres})
        print(f"  done: {file_id}", flush=True)

    sync_df = pd.DataFrame(sync_rows)
    conv_df = pd.DataFrame(conv_rows)
    os.makedirs(outdir, exist_ok=True)
    sync_df.to_csv(os.path.join(outdir, "synchrony_results.csv"), index=False, encoding="utf-8-sig")
    conv_df.to_csv(os.path.join(outdir, "convergence_results.csv"), index=False, encoding="utf-8-sig")

    # ---- 概览：人→AI vs AI→人 趋近强度比较 ----
    print("\n===== 同步（均值）=====")
    if len(sync_df):
        print(sync_df[["sync_strength", "ctrl_strength", "pos_sync_ratio",
                       "neg_sync_ratio", "human_follow_ratio", "ai_follow_ratio"]].mean().round(3).to_string())
    print("\n===== 趋近：各方向平均斜率（负=趋近）=====")
    if len(conv_df):
        piv = conv_df.groupby("mode")["slope"].mean()
        print(piv.round(5).to_string())
        h2a = conv_df[conv_df["mode"] == "h2a"]["slope"].mean()
        a2h = conv_df[conv_df["mode"] == "a2h"]["slope"].mean()
        who = "人→AI" if h2a < a2h else "AI→人"
        print(f"\n  人→AI 平均斜率={h2a:.5f}, AI→人 平均斜率={a2h:.5f} → 趋近更强的一方: {who}")
        print(f"  显著趋近占比: " + conv_df.groupby('mode').apply(
            lambda d: f"{(d.direction=='convergence').mean():.0%}", include_groups=False).to_string().replace('\n', ' | '))
    print(f"\n输出: {os.path.join(outdir, 'synchrony_results.csv')} / convergence_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="analysis/features_sample.csv")
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--features-list", nargs="*", default=FEATURES_DEFAULT)
    args = ap.parse_args()
    run(args.features, args.outdir, args.features_list)
