# -*- coding: utf-8 -*-
"""为 Results 改写生成所需的全部聚合表与相关统计。
分层：prosody(韵律) / spectral(MFCC谱) / embedding(高维向量)。
输出打印 + 落盘 analysis/tab_*.csv。
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

IND = "analysis"
PROS = ["pitch_mean","pitch_max","pitch_min","pitch_std",
        "intensity_mean","intensity_max","intensity_min","intensity_std","speech_rate"]
SPEC = ["mfcc_centroid_dist"]
EMB  = ["embedding_cosine_dist"]
def layer(f):
    return "韵律" if f in PROS else ("谱(MFCC)" if f in SPEC else "高维(wav2vec2)")

sync = pd.read_csv(f"{IND}/synchrony_results.csv")
conv = pd.read_csv(f"{IND}/convergence_results.csv")
ei   = pd.read_csv(f"{IND}/ei_results.csv")
sam  = pd.read_csv(f"{IND}/sam_scores.csv")
feat = pd.read_csv(f"{IND}/features_clean.csv")
for d in (sync,conv,ei): d["layer"]=d["feature"].map(layer)

import re
def subj(f):
    m=re.search(r"(\d+)",str(f)); return int(m.group(1)) if m else np.nan
meta = feat.groupby("file").agg(scene=("scene","first"),
        dur=("end","max")).reset_index()
meta["dur"] = feat.groupby("file").apply(lambda d:d["end"].max()-d["start"].min(), include_groups=False).values
meta["subject_id"]=meta["file"].map(subj)

def show(title, df):
    print(f"\n###### {title} ######"); print(df.to_string(index=False))
    return df

# ---------- 同步：分层×场景 ----------
g = sync.groupby(["layer","scene"]).agg(
    n=("sync_strength","size"), strength=("sync_strength","mean"),
    ctrl=("ctrl_strength","mean"), sig=("significant","mean"),
    pos=("pos_sync_ratio","mean"), neg=("neg_sync_ratio","mean"),
    hfollow=("human_follow_ratio","mean"), aifollow=("ai_follow_ratio","mean")).round(3).reset_index()
show("同步 分层×场景", g).to_csv(f"{IND}/tab_sync_layer_scene.csv",index=False,encoding="utf-8-sig")
g2 = sync.groupby("layer").agg(strength=("sync_strength","mean"),ctrl=("ctrl_strength","mean"),
    sig=("significant","mean"),pos=("pos_sync_ratio","mean"),neg=("neg_sync_ratio","mean")).round(3).reset_index()
show("同步 分层(总体)", g2)

# ---------- 趋近：分层×场景（对称 + 方向）----------
sym=conv[conv["mode"]=="sym"]
gc = sym.groupby(["layer","scene"]).agg(n=("slope","size"),slope=("slope","mean"),
    conv=("direction",lambda x:(x=="convergence").mean()),
    div=("direction",lambda x:(x=="divergence").mean())).round(4).reset_index()
show("趋近(对称) 分层×场景", gc).to_csv(f"{IND}/tab_conv_layer_scene.csv",index=False,encoding="utf-8-sig")
dirc = conv[conv["mode"].isin(["h2a","a2h"])].groupby(["layer","mode"])["slope"].mean().round(4).unstack("mode").reset_index()
show("趋近方向 分层 (h2a=人→AI, a2h=AI→人)", dirc).to_csv(f"{IND}/tab_conv_dir_layer.csv",index=False,encoding="utf-8-sig")
# 关键特征（HHI 对照用）
show("pitch_mean 趋近斜率 分场景(对称)", sym[sym.feature=="pitch_mean"].groupby("scene")["slope"].mean().round(4).reset_index())

# ---------- EI：分层×场景 ----------
ge = ei.groupby(["layer","scene"])["EI"].mean().round(3).unstack("scene").reset_index()
show("EI(z) 分层×场景", ge).to_csv(f"{IND}/tab_ei_layer_scene.csv",index=False,encoding="utf-8-sig")
show("EI(z) 分层(总体)", ei.groupby("layer")["EI"].mean().round(3).reset_index())

# ---------- PAD × 趋同 相关（每对话）----------
per = (sync.groupby("file")["sync_strength"].mean().rename("sync")
       .to_frame()
       .join(sym.groupby("file")["slope"].mean().rename("conv_slope"))
       .join(ei.groupby("file")["EI"].mean().rename("EI"))
       .reset_index())
per["conv_pos"]=-per["conv_slope"]
per=per.merge(meta[["file","scene","subject_id","dur"]],on="file").merge(sam,on=["subject_id","scene"],how="left")
per.to_csv(f"{IND}/tab_perfile_pad.csv",index=False,encoding="utf-8-sig")

rows=[]
for resp in ["sync","conv_pos","EI"]:
    for pad in ["pleasure","arousal","dominance"]:
        d=per.dropna(subset=[resp,pad])
        r,pr=stats.pearsonr(d[resp],d[pad]); rho,ps=stats.spearmanr(d[resp],d[pad])
        rows.append({"趋同指标":resp,"情绪维度":pad,"n":len(d),
                     "Pearson_r":round(r,3),"p":round(pr,4),
                     "Spearman_rho":round(rho,3),"p_s":round(ps,4)})
cor=pd.DataFrame(rows)
show("PAD × 趋同 相关(每对话, n≈115)", cor).to_csv(f"{IND}/tab_pad_corr.csv",index=False,encoding="utf-8-sig")
print("\n完成，表已落盘 analysis/tab_*.csv")
