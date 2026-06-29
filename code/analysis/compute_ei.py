"""独立 EI（Entrainment Index）计算脚本（3.5）

对齐 81 号论文（Li et al., 2026）公式：
    EI = 0.5 · Normalized_Sync + 0.5 · Normalized_Conv
其中
    Normalized_Sync = ( S_sync_real − μ_sync,rand ) / σ_sync,rand
        S_sync = r_max /(1+ℓ)   —— 滞后惩罚同步分数（ℓ 为最佳滞后绝对值）
    Normalized_Conv = ( slope_real − μ_slope,rand ) / σ_slope,rand
        趋近为负斜率 → 趋近越强，Normalized_Conv 越负；为使"趋近=正贡献"，
        对趋近项取负号后再合成（slope 越负→ −z 越大→ EI 越高）。
    最终 EI 在群体内转为 z 分数。

输入：compute_entrainment.py 产出的 synchrony_results.csv / convergence_results.csv
      （convergence 取 mode=='sym' 的对称口径，与 81 号一致）
输出：ei_results.csv（逐 file×feature 的 EI 及其两个分量）

用法：
    python analysis/compute_ei.py --indir analysis --outdir analysis
"""
import os
import argparse
import numpy as np
import pandas as pd


def zsafe(x, mu, sd):
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return (x - mu) / sd


def run(indir, outdir):
    sync = pd.read_csv(os.path.join(indir, "synchrony_results.csv"))
    conv = pd.read_csv(os.path.join(indir, "convergence_results.csv"))
    conv_sym = conv[conv["mode"] == "sym"].copy()

    # 逐 file×feature 合成 EI 两分量
    m = pd.merge(
        sync[["file", "scene", "feature", "s_sync_real", "s_sync_rand_mean", "s_sync_rand_std"]],
        conv_sym[["file", "feature", "slope", "slope_rand_mean", "slope_rand_std"]],
        on=["file", "feature"], how="inner")

    m["norm_sync"] = m.apply(
        lambda r: zsafe(r["s_sync_real"], r["s_sync_rand_mean"], r["s_sync_rand_std"]), axis=1)
    # 趋近：负斜率=趋近。标准化后取负，使"趋近"为正贡献。
    m["norm_conv"] = m.apply(
        lambda r: -zsafe(r["slope"], r["slope_rand_mean"], r["slope_rand_std"]), axis=1)

    m["EI_raw"] = 0.5 * m["norm_sync"] + 0.5 * m["norm_conv"]
    # 群体内 z 标准化（81 号：最终 EI 转 z 分数）
    valid = m["EI_raw"].notna()
    mu, sd = m.loc[valid, "EI_raw"].mean(), m.loc[valid, "EI_raw"].std()
    m["EI"] = (m["EI_raw"] - mu) / sd if sd > 1e-12 else np.nan

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "ei_results.csv")
    m.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"EI 计算完成：{valid.sum()} 个 file×feature 有效（共 {len(m)}）")
    print(f"空值检查：norm_sync NaN={m['norm_sync'].isna().sum()}, "
          f"norm_conv NaN={m['norm_conv'].isna().sum()}")
    print("\n===== 各特征平均 EI（z 分数）=====")
    print(m.groupby("feature")["EI"].mean().round(3).to_string())
    if "scene" in m.columns:
        print("\n===== 各场景平均 EI =====")
        print(m.groupby("scene")["EI"].mean().round(3).to_string())
    print(f"\n输出: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="analysis")
    ap.add_argument("--outdir", default="analysis")
    args = ap.parse_args()
    run(args.indir, args.outdir)
