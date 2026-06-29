"""按 (角色, 性别) 分组剔除特征奇异值（3.2 后处理）。

动机：基频等特征强烈依赖性别（男声 ~100 Hz，女声 ~200 Hz），统一阈值会误判。
做法：human 行按性别（男/女）分组、AI 行单独成组；对每个特征用稳健的
      中位数 ± k·(1.4826·MAD) 规则把奇异值置为 NaN（保留行，下游按特征跳过 NaN，
      不破坏 IPU 序列）。默认 k=3.5。

输入：features_all.csv  输出：features_clean.csv
用法：python analysis/clean_features.py --in analysis/features_all.csv --out analysis/features_clean.csv
"""
import argparse
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "pitch_mean", "pitch_max", "pitch_min", "pitch_std",
    "intensity_mean", "intensity_max", "intensity_min", "intensity_std",
    "speech_rate", "mfcc_centroid_dist", "embedding_cosine_dist",
]


def group_key(row):
    return "AI" if str(row["role"]).strip().upper() == "AI" else f"human-{row['gender']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="analysis/features_all.csv")
    ap.add_argument("--out", default="analysis/features_clean.csv")
    ap.add_argument("--k", type=float, default=3.5, help="MAD 倍数阈值")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df["_grp"] = df.apply(group_key, axis=1)
    report = []
    total_out = 0

    for grp, idx in df.groupby("_grp").groups.items():
        sub = df.loc[idx]
        for col in FEATURE_COLS:
            x = sub[col].astype(float)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            if mad < 1e-9:
                continue
            thr = args.k * 1.4826 * mad
            out_mask = (np.abs(x - med) > thr)
            n_out = int(out_mask.sum())
            if n_out:
                df.loc[x.index[out_mask], col] = np.nan
                total_out += n_out
                report.append((grp, col, n_out, round(med, 2), round(thr, 2)))

    df = df.drop(columns=["_grp"])
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"总行数: {len(df)}; 置为 NaN 的奇异值: {total_out}")
    print("分组×特征 奇异值计数 (grp, feat, n, median, 阈值):")
    for r in sorted(report, key=lambda t: -t[2])[:25]:
        print("  ", r)
    print("\n各特征剩余 NaN 占比:")
    print((df[FEATURE_COLS].isna().mean() * 100).round(2).to_string())
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
