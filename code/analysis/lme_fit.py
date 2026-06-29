# -*- coding: utf-8 -*-
"""LME 拟合与原始统计输出（供 Results 三线表）。
模型：response ~ FIXED + (1 | speaker)，录音时长 dur_z 为协变量。
对 Synchrony / Convergence(-slope) / EI 各比较 M_null / M_scene / M_PAD，
报告 AIC/BIC、固定效应 β、SE、p、被试随机方差。结果存 analysis/lme/*.csv。
"""
import os, sys, re, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
IND="analysis"; OUT=f"{IND}/lme"; os.makedirs(OUT, exist_ok=True)

feat=pd.read_csv(f"{IND}/features_clean.csv")
sync=pd.read_csv(f"{IND}/synchrony_results.csv")
conv=pd.read_csv(f"{IND}/convergence_results.csv")
ei=pd.read_csv(f"{IND}/ei_results.csv")
sam=pd.read_csv(f"{IND}/sam_scores.csv")

g=feat.groupby("file")
meta=pd.DataFrame({"scene":g["scene"].first(),
                   "dur":g["end"].max()-g["start"].min()}).reset_index()
meta["subject_id"]=meta["file"].map(lambda f:int(re.search(r"(\d+)",str(f)).group(1)) if re.search(r"(\d+)",str(f)) else np.nan).astype("Int64")
meta["dur_z"]=(meta.dur-meta.dur.mean())/meta.dur.std()

def design(df):
    d=df.merge(meta[["file","scene","subject_id","dur_z"]],on=["file","scene"],how="left") if "scene" in df.columns \
        else df.merge(meta,on="file",how="left")
    return d.merge(sam,on=["subject_id","scene"],how="left")

def fit(data,resp,fixed,grp="subject_id"):
    need=[resp,grp]+[v for v in ["pleasure","arousal","dominance","dur_z","scene","feature"] if v in fixed]
    sub=data.dropna(subset=[c for c in need if c in data.columns]).copy()
    sub[grp]=sub[grp].astype(int)
    ols=smf.ols(f"{resp} ~ {fixed}",sub).fit()
    try:
        m=smf.mixedlm(f"{resp} ~ {fixed}",sub,groups=sub[grp]).fit(reml=False,method="lbfgs")
        params,se,pv=m.params,m.bse,m.pvalues; rev=float(m.cov_re.iloc[0,0])
    except Exception:
        params,se,pv=ols.params,ols.bse,ols.pvalues; rev=np.nan
    rows=[]
    for nm in params.index:
        if nm=="Group Var": continue
        rows.append({"term":nm,"beta":round(params[nm],4),"SE":round(se[nm],4),"p":round(pv[nm],4)})
    return {"aic":round(ols.aic,1),"bic":round(ols.bic,1),"re_var":round(rev,4),"n":len(sub),
            "groups":sub[grp].nunique(),"coefs":pd.DataFrame(rows)}

def run(data,resp,tag):
    fe=" + C(feature)" if ("feature" in data.columns and data["feature"].nunique()>1) else ""
    specs={"M_null":"1"+fe,"M_scene":"C(scene) + dur_z"+fe,"M_PAD":"pleasure + arousal + dominance + dur_z"+fe}
    comp=[];
    for name,fx in specs.items():
        r=fit(data,resp,fx); comp.append({"model":name,"AIC":r["aic"],"BIC":r["bic"],"RE_var":r["re_var"],"n":r["n"],"groups":r["groups"]})
        r["coefs"].to_csv(f"{OUT}/{tag}_{name}_coefs.csv",index=False,encoding="utf-8-sig")
    cdf=pd.DataFrame(comp); cdf.to_csv(f"{OUT}/{tag}_modelcomp.csv",index=False,encoding="utf-8-sig")
    best=cdf.loc[cdf.AIC.idxmin(),"model"]
    print(f"\n### {tag}  最优={best}")
    print(cdf.to_string(index=False))
    print("最优模型固定效应:"); print(pd.read_csv(f"{OUT}/{tag}_{best}_coefs.csv").to_string(index=False))

# EI（每对话均值, EI_raw）
eikey="EI_raw" if "EI_raw" in ei.columns else "EI"
eif=ei.groupby(["file","scene"])[eikey].mean().reset_index().rename(columns={eikey:"EI_raw"})
run(design(eif),"EI_raw","EI")
# Synchrony（file×feature）
run(design(sync),"sync_strength","SYNC")
# Convergence（对称, -slope）
cs=conv[conv["mode"]=="sym"].copy(); cs["conv_pos"]=-cs["slope"]
run(design(cs),"conv_pos","CONV")
print("\nLME 原始结果存于", OUT)
