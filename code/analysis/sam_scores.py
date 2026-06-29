"""导出每 (被试, 情境) 的 SAM PAD 均值（两名评定员均值），供 LME 固定效应使用。

数据源：SAM标注可靠性分析.py 内嵌的 data_str（避免重复维护，直接正则提取）。
被试编号：SAM id(1-40) 即文件名中的序号（如 男-10 = id 10）。
输出：sam_scores.csv  列：subject_id, scene, pleasure, arousal, dominance
"""
import io
import re
import os
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sam(src_py):
    txt = open(src_py, encoding="utf-8").read()
    m = re.search(r'data_str\s*=\s*"""(.*?)"""', txt, re.S)
    if not m:
        raise RuntimeError("未在源文件中找到 data_str")
    df = pd.read_csv(io.StringIO(m.group(1)))
    df["pleasure"] = (df["p_1"] + df["p_2"]) / 2.0
    df["arousal"] = (df["a_1"] + df["a_2"]) / 2.0
    df["dominance"] = (df["d_1"] + df["d_2"]) / 2.0
    return df[["id", "scene", "pleasure", "arousal", "dominance"]].rename(
        columns={"id": "subject_id"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "SAM标注可靠性分析.py"))
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "sam_scores.csv"))
    args = ap.parse_args()
    sam = load_sam(args.src)
    sam.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"SAM 分数: {len(sam)} 行 (被试×情境)")
    print(sam.groupby("scene")[["pleasure", "arousal", "dominance"]].mean().round(2).to_string())
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
