# -*- coding: utf-8 -*-
"""生成 Results 用 SVG 插图（英文标注，seaborn 配色），存 稿子/figs/。"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns; sns.set_theme(style="whitegrid", palette="deep")
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")
import scipy.stats as st

IND="analysis"; FIG=r"D:/HAI Entrainment/稿子/figs"; os.makedirs(FIG, exist_ok=True)
_sv = lambda p: (plt.savefig(p), plt.savefig(p[:-4]+".png", dpi=160, bbox_inches="tight"))
sync=pd.read_csv(f"{IND}/synchrony_results.csv"); conv=pd.read_csv(f"{IND}/convergence_results.csv")
ei=pd.read_csv(f"{IND}/ei_results.csv")
PROS=["pitch_mean","pitch_max","pitch_min","pitch_std","intensity_mean","intensity_max","intensity_min","intensity_std","speech_rate"]
EN={"pitch_mean":"F0 mean","pitch_max":"F0 max","pitch_min":"F0 min","pitch_std":"F0 SD",
    "intensity_mean":"Int. mean","intensity_max":"Int. max","intensity_min":"Int. min","intensity_std":"Int. SD","speech_rate":"Speech rate"}

# Fig 1: per-feature convergence slope by scene (9 prosodic)
sym=conv[conv["mode"]=="sym"]
piv=sym[sym.feature.isin(PROS)].pivot_table(index="feature",columns="scene",values="slope").reindex(PROS)
piv=piv.rename(index=EN)[["neutral","arguing","comforting"]]
ax=piv.plot(kind="bar",figsize=(9,4.5),width=0.8)
ax.axhline(0,color="k",lw=0.8); ax.set_ylabel("Convergence slope (negative = converging)")
ax.set_xlabel(""); ax.set_title("Per-feature convergence slope by scene (prosodic features)")
ax.legend(title="Scene"); plt.xticks(rotation=40,ha="right"); plt.tight_layout()
_sv(f"{FIG}/fig3_convergence_byfeature.svg"); plt.close()

# Fig 2: synchrony — significant-rate per feature (3 layers)
sf=sync.groupby("feature").significant.mean().reindex(PROS+["mfcc_centroid_dist","embedding_cosine_dist"])*100
labels=[EN.get(f,{"mfcc_centroid_dist":"MFCC dist","embedding_cosine_dist":"wav2vec2 dist"}.get(f,f)) for f in sf.index]
colors=["#4C72B0"]*9+["#DD8452","#55A868"]
plt.figure(figsize=(9,4.2)); plt.bar(labels,sf.values,color=colors)
plt.ylabel("% units significant vs. random baseline"); plt.title("Synchrony significance rate by feature")
plt.axhline(5,color="grey",ls="--",lw=1,label="5% chance level"); plt.legend()
plt.xticks(rotation=40,ha="right"); plt.tight_layout(); _sv(f"{FIG}/fig2_synchrony_significance.svg"); plt.close()

# Fig 3: EI distribution + QQ
fig,axes=plt.subplots(1,2,figsize=(10,4))
try:
    import seaborn as sns
    sns.histplot(ei["EI"],kde=True,ax=axes[0],color="#4C72B0")
except Exception:
    axes[0].hist(ei["EI"],bins=30)
axes[0].set_title("Distribution of EI (z-score)"); axes[0].set_xlabel("EI (z)")
st.probplot(ei["EI"].dropna(),dist="norm",plot=axes[1])
axes[1].set_title("Normal Q–Q plot of EI")
axes[1].get_lines()[0].set_color("#4C72B0"); axes[1].get_lines()[1].set_color("#C44E52")
plt.tight_layout(); _sv(f"{FIG}/fig5_EI_distribution.svg"); plt.close()

# Fig 4: EI by layer x scene (grouped bar)
ei["layer"]=ei["feature"].map(lambda f:"Prosody" if f in PROS else ("MFCC" if f=="mfcc_centroid_dist" else "wav2vec2"))
p2=ei.pivot_table(index="layer",columns="scene",values="EI")[["neutral","arguing","comforting"]]
ax=p2.plot(kind="bar",figsize=(7,4),width=0.75); ax.axhline(0,color="k",lw=0.8)
ax.set_ylabel("Mean EI (z)"); ax.set_xlabel(""); ax.set_title("EI by representation layer and scene")
ax.legend(title="Scene"); plt.xticks(rotation=0); plt.tight_layout()
_sv(f"{FIG}/fig4_EI_layer_scene.svg"); plt.close()

print("SVG 已生成:", os.listdir(FIG))
