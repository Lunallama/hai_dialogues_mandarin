# -*- coding: utf-8 -*-
"""Results 扩写所需的全部数值：数据量/场景可比性、逐特征趋同、与前人(同一度量)对比+单样本检验、
EI 分布/正态性、趋近发散比。结果存 analysis/tab2_*.csv。不改动既有脚本与正文。"""
import os, sys, re
import numpy as np, pandas as pd
from scipy import stats
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
IND = "analysis"

feat = pd.read_csv(f"{IND}/features_clean.csv")
sync = pd.read_csv(f"{IND}/synchrony_results.csv")
conv = pd.read_csv(f"{IND}/convergence_results.csv")
ei   = pd.read_csv(f"{IND}/ei_results.csv")

PROS = ["pitch_mean","pitch_max","pitch_min","pitch_std",
        "intensity_mean","intensity_max","intensity_min","intensity_std","speech_rate"]
STYLE = ["mfcc_centroid_dist","embedding_cosine_dist"]
ALLF = PROS + STYLE

# ---------- 1. 数据量 & 场景可比性 ----------
g = feat.groupby("file")
dur = (g["end"].max() - g["start"].min())
meta = pd.DataFrame({"scene": g["scene"].first(), "ipu": g.size(), "dur": dur}).reset_index()
print("=== 数据量 ===")
print(f"对话数={len(meta)}  IPU总数={int(meta.ipu.sum())}  总时长={meta.dur.sum()/60:.1f} min ({meta.dur.sum()/3600:.2f} h)")
vol = meta.groupby("scene").agg(dialogues=("file","size"), IPU=("ipu","sum"),
        dur_min=("dur", lambda x: x.sum()/60), dur_mean_s=("dur","mean"), dur_sd_s=("dur","std")).round(2)
print(vol.to_string())
# 场景间每段时长差异：Kruskal-Wallis（非正态稳健）
groups=[meta[meta.scene==s]["dur"].values for s in ["neutral","arguing","comforting"]]
H,p = stats.kruskal(*groups)
print(f"场景×每段时长 Kruskal-Wallis: H={H:.2f}, p={p:.3f}  (p>0.05 表示时长可比)")
vol.to_csv(f"{IND}/tab2_volume.csv", encoding="utf-8-sig")

# ---------- 2. 逐特征 同步 & 趋近 ----------
sf = sync.groupby("feature").agg(sync=("sync_strength","mean"), ctrl=("ctrl_strength","mean"),
        sig=("significant","mean"), pos=("pos_sync_ratio","mean"), neg=("neg_sync_ratio","mean")).round(3)
cf = conv[conv["mode"]=="sym"].groupby("feature").agg(slope=("slope","mean"),
        convr=("direction",lambda x:(x=="convergence").mean()),
        divr=("direction",lambda x:(x=="divergence").mean())).round(4)
perfeat = sf.join(cf).reindex(ALLF)
print("\n=== 逐特征 同步/趋近 ===")
print(perfeat.to_string())
perfeat.to_csv(f"{IND}/tab2_perfeature.csv", encoding="utf-8-sig")

# ---------- 3. 同一度量重算 + 与前人单样本检验 ----------
# Levitan 同步 = 相邻跨说话人 IPU 的带符号 Pearson r；趋近 = |相邻差| 对时间的 Pearson r
def adjacent_pairs(gdf, fcol):
    gdf = gdf.sort_values("start"); r = gdf["role"].values; v = gdf[fcol].values; t = gdf["start"].values
    A,B,T=[],[],[]
    for i in range(1,len(gdf)):
        if r[i]!=r[i-1] and not (np.isnan(v[i]) or np.isnan(v[i-1])):
            A.append(v[i]); B.append(v[i-1]); T.append((t[i]+t[i-1])/2)
    return np.array(A),np.array(B),np.array(T)

LEV_SYNC = {"intensity_max":0.50,"intensity_mean":0.47,"pitch_mean":0.28,"pitch_max":0.18,"speech_rate":0.15}
LEV_CONV = {"pitch_mean":-0.06,"pitch_max":-0.05}
rows=[]
for fcol in ["intensity_max","intensity_mean","pitch_mean","pitch_max","speech_rate"]:
    rsync=[]; rconv=[]
    for fid,gdf in feat.groupby("file"):
        A,B,T = adjacent_pairs(gdf,fcol)
        if len(A)>=5 and np.std(A)>0 and np.std(B)>0:
            rsync.append(np.corrcoef(A,B)[0,1])
            d=np.abs(A-B)
            if np.std(d)>0 and np.std(T)>0: rconv.append(np.corrcoef(d,T)[0,1])
    rsync=np.array(rsync); rconv=np.array(rconv)
    lev=LEV_SYNC[fcol]
    t1,p1=stats.ttest_1samp(rsync, lev)
    row={"feature":fcol,"my_sync_r":round(rsync.mean(),3),"my_sync_sd":round(rsync.std(),3),
         "Levitan_sync_r":lev,"t_vs_Lev":round(t1,2),"p_vs_Lev":f"{p1:.1e}","n":len(rsync)}
    if fcol in LEV_CONV:
        t2,p2=stats.ttest_1samp(rconv, LEV_CONV[fcol])
        row.update({"my_conv_r":round(rconv.mean(),3),"Levitan_conv_r":LEV_CONV[fcol],
                    "conv_t":round(t2,2),"conv_p":f"{p2:.1e}"})
    else:
        row.update({"my_conv_r":round(rconv.mean(),3),"Levitan_conv_r":"-","conv_t":"-","conv_p":"-"})
    rows.append(row)
cmp=pd.DataFrame(rows)
print("\n=== 同一度量：本研究(HAI) vs Levitan&Hirschberg 2011(HHI)，单样本t检验 ===")
print(cmp.to_string(index=False))
cmp.to_csv(f"{IND}/tab2_vs_levitan.csv",index=False,encoding="utf-8-sig")

# ---------- 4. EI 分布 / 正态性 / 趋近发散比 ----------
print("\n=== EI 分布 ===")
W,pn=stats.shapiro(ei["EI"].dropna())
print(f"EI(z) Shapiro–Wilk W={W:.3f}, p={pn:.2e}  (p<0.05 偏离正态)")
print("分场景 EI 偏度/峰度:", {s: (round(stats.skew(ei[ei.scene==s].EI),2), round(stats.kurtosis(ei[ei.scene==s].EI),2)) for s in ei.scene.unique()})
sym=conv[conv["mode"]=="sym"]
nconv=(sym.direction=="convergence").sum(); ndiv=(sym.direction=="divergence").sum()
print(f"趋近:发散 = {nconv}:{ndiv} = {nconv/ndiv:.2f}:1 (对称口径全特征)")
for lay,cols in [("prosody",PROS),("mfcc",["mfcc_centroid_dist"]),("emb",["embedding_cosine_dist"])]:
    s2=sym[sym.feature.isin(cols)]
    nc=(s2.direction=="convergence").sum(); nd=(s2.direction=="divergence").sum()
    print(f"  {lay}: conv:div = {nc}:{nd} = {nc/max(nd,1):.2f}:1")
print("\n完成，表存 analysis/tab2_*.csv")
