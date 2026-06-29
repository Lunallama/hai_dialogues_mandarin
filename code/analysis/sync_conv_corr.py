# -*- coding: utf-8 -*-
"""同步(synchrony)与趋近(convergence)在同一(说话人×场景, 特征)单元上是否相关。
若近乎不相关 → 二者是相对独立的过程，支持"同步偏自启动、趋近更受社会/情感驱动"。
读现成 synchrony_results.csv / convergence_results.csv，不重算。结果存 analysis/sync_conv_corr/。
"""
import os
import numpy as np, pandas as pd
from scipy import stats

A = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(A, "sync_conv_corr"); os.makedirs(OUT, exist_ok=True)

PROS = ["pitch_mean","pitch_max","pitch_min","pitch_std",
        "intensity_mean","intensity_max","intensity_min","intensity_std","speech_rate"]
def layer_of(f):
    return "Prosodic" if f in PROS else ("Spectral(MFCC)" if f=="mfcc_centroid_dist" else "High-level(wav2vec2)")

sync = pd.read_csv(os.path.join(A,"synchrony_results.csv"))
conv = pd.read_csv(os.path.join(A,"convergence_results.csv"))
conv = conv[conv["mode"]=="sym"].copy()

m = sync.merge(conv, on=["file","scene","feature"], suffixes=("_s","_c"))
m["conv_score"] = -m["slope"]          # 正=趋近，与 EI 口径一致
m["sync_score"] = m["sync_strength"]
m = m.dropna(subset=["sync_score","conv_score"])
m["layer"] = m["feature"].map(layer_of)

def corr(x,y):
    x,y = np.asarray(x,float), np.asarray(y,float)
    ok = ~(np.isnan(x)|np.isnan(y))
    if ok.sum()<5: return (np.nan,np.nan,np.nan,int(ok.sum()))
    r,pr = stats.pearsonr(x[ok],y[ok]); rho,prho = stats.spearmanr(x[ok],y[ok])
    return (r,pr,rho,int(ok.sum()))

lines=[]
# 1) 整体（原始）
r,pr,rho,n = corr(m.sync_score, m.conv_score)
lines.append(f"【整体·原始】sync vs conv(=-slope)  n={n}  Pearson r={r:+.3f} (p={pr:.3g})  Spearman ρ={rho:+.3f}")

# 2) 特征内标准化后合并（去除特征级均值差异的混淆）
mm = m.copy()
for col in ["sync_score","conv_score"]:
    mm[col+"_z"] = mm.groupby("feature")[col].transform(lambda v:(v-v.mean())/v.std(ddof=0) if v.std(ddof=0)>0 else v*0)
rz,pz,rhoz,nz = corr(mm.sync_score_z, mm.conv_score_z)
lines.append(f"【特征内标准化后合并】 n={nz}  Pearson r={rz:+.3f} (p={pz:.3g})  Spearman ρ={rhoz:+.3f}")

# 3) 逐特征
lines.append("\n逐特征（同一特征跨对话）：")
rows=[]
for f,g in m.groupby("feature"):
    r,pr,rho,n = corr(g.sync_score,g.conv_score)
    rows.append((f,layer_of(f),r,pr,n))
rows.sort(key=lambda t:(t[1],t[0]))
for f,L,r,pr,n in rows:
    star = "*" if (pr==pr and pr<.05) else " "
    lines.append(f"  {L:<22s} {f:<18s} r={r:+.3f}{star} (p={pr:.3g}, n={n})")

# 4) 分层（特征内标准化后，层内合并）
lines.append("\n分层（特征内标准化后合并）：")
for L,g in mm.groupby("layer"):
    r,pr,rho,n = corr(g.sync_score_z,g.conv_score_z)
    lines.append(f"  {L:<22s} r={r:+.3f} (p={pr:.3g}, ρ={rho:+.3f}, n={n})")

# 5) 分场景（特征内标准化后）
lines.append("\n分场景（特征内标准化后合并）：")
for sc,g in mm.groupby("scene"):
    r,pr,rho,n = corr(g.sync_score_z,g.conv_score_z)
    lines.append(f"  {sc:<12s} r={r:+.3f} (p={pr:.3g}, n={n})")

# 6) 显著特征占比（|r|>.1 且 p<.05）
sig = [t for t in rows if t[3]==t[3] and t[3]<.05]
lines.append(f"\n逐特征中达 p<.05 的：{len(sig)}/{len(rows)}")

txt="\n".join(lines)
open(os.path.join(OUT,"sync_conv_corr_summary.txt"),"w",encoding="utf-8").write(txt)
m.to_csv(os.path.join(OUT,"sync_conv_merged.csv"),index=False,encoding="utf-8-sig")
print(txt); print("\n存于",OUT)
