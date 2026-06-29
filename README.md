# Acoustic-Prosodic Entrainment in Affective Human–AI Spoken Interaction (Mandarin)

Anonymized data and analysis code for the study *"Acoustic-Prosodic Entrainment in Affective
Human–AI Spoken Interaction: Evidence from Mandarin Chinese."* The corpus consists of spontaneous
Mandarin dialogues between human participants and an emotion-capable conversational AI (Doubao)
across three scenes: **neutral**, **arguing**, and **comforting**.

## Privacy / what is (not) here

To protect participants' biometric privacy, **raw audio is NOT shared**. This repository contains
only **anonymized, per–inter-pausal-unit (IPU) acoustic feature matrices**, the derived entrainment
results, and the analysis code. Speaker names have been replaced by numeric IDs (`S01`, `S02`, …),
verbatim transcripts (`text`) have been removed, and no file here contains personally identifying
information. Qualified researchers may request access to the raw audio for replication under a Data
Use Agreement by contacting the corresponding author.

## Repository layout

```
data/    anonymized feature matrices, entrainment results, and aggregate tables (CSV)
code/    analysis pipeline
  analysis/   feature extraction → entrainment → mixed-effects models → figures
  core/       reusable feature extractors (pitch/intensity, MFCC, speech rate, wav2vec2)
```

## Data files (key columns)

- **`features_clean.csv`** — one row per IPU (the input to entrainment).
  `file` = `<scene>_S<id>` dialogue id; `scene` ∈ {neutral, arguing, comforting};
  `subject_id` (1–39); `gender` (F/M); `role` (human/AI); `speaker` (diarization label);
  `start`,`end`,`duration` (s); `pitch_{mean,max,min,std}` (Hz); `intensity_{mean,max,min,std}` (dB);
  `speech_rate` (syllables/s); `mfcc_centroid_dist`, `embedding_cosine_dist`
  (cosine distance of the IPU's 13-dim MFCC mean / wav2vec2 layer-4 embedding to that speaker's
  own style centroid; larger = farther from the speaker's typical voice).
- **`synchrony_results.csv`** — per dialogue × feature short-term synchrony (window |r|, significance
  vs. time-shuffled baseline, sign ratios, lags).
- **`convergence_results.csv`** — per dialogue × feature × direction (`mode` = sym/h2a/a2h)
  convergence slope and permutation p (negative slope = convergence).
- **`ei_results.csv`** — per dialogue × feature Entrainment Index components and `EI` (z-scored).
- **`sam_scores.csv`** — per subject × scene mean SAM ratings (`pleasure`, `arousal`, `dominance`, −4…+4).
- **`speakers.csv`** — `subject_id` → `gender`.
- **`agg_*.csv`, `tab_*.csv`** — aggregate tables reported in the paper.

## Reproducing the analysis

The pipeline (in `code/`) runs: feature extraction (`extract_features.py`, requires the raw audio,
not shared) → cleaning (`clean_features.py`) → synchrony & convergence (`compute_entrainment.py`)
→ Entrainment Index (`compute_ei.py`) → mixed-effects models (`lme_fit.py`) → figures
(`make_figures.py`); robustness checks in `compare_centroid.py`, `layer_sweep.py`, `sync_conv_corr.py`.
Starting from the shared `features_clean.csv`, the entrainment, modelling, and figure steps reproduce
all reported results without the raw audio.

Dependencies: Python 3.11, `numpy pandas scipy scikit-learn statsmodels matplotlib seaborn`
(feature extraction additionally needs `praat-parselmouth`, `transformers`, `torch`, `soundfile`).

