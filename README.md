# Criminal Mind Analysis

> A 4-stage LLM pipeline for forensic crime motivation analysis — classifying, predicting, clustering, and synthesizing criminal behavior patterns using real-world datasets.

---

## Overview

**Criminal Mind Analysis** is an end-to-end AI pipeline that takes raw crime data and produces a structured forensic report explaining the likely psychological motivation behind a crime. It chains four machine learning models together, each contributing a different analytical lens, and fuses their outputs via a confidence-weighted scoring system fed into Google Gemini for final synthesis.

The pipeline processes **524,748 total crime records** across three real-world datasets and runs on Google Colab with a T4 GPU.

---

## Pipeline Architecture

```
Raw Crime Input
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  LLM-1 │ Motivation Classifier                      │
│  MiniLM-L6-v2 + Logistic Regression                 │
│  → Classifies: Emotional / Financial / Power        │
│  → Accuracy: ~67% (after bias fix)                  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  LLM-2 │ Historical Predictor                       │
│  RandomForest MultiOutputRegressor                   │
│  → Predicts motive distribution (13 categories)     │
│  → MAE: 40.98 across 13 output columns              │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  LLM-3 │ Context Clusterer                          │
│  MiniLM-L6-v2 + MiniBatchKMeans                     │
│  → Assigns behavioral cluster: Aggressive /         │
│    Opportunistic / Impulsive / Premeditated /        │
│    Domestic                                          │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  Fusion Layer                                        │
│  Confidence-weighted scoring                         │
│  High=0.95 │ Medium=0.65 │ Low=0.40                 │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  LLM-4 │ Synthesis Engine                           │
│  Google Gemini 2.5 Flash                            │
│  → Generates 6-step chain-of-thought forensic       │
│    report in structured JSON                        │
└─────────────────────────────────────────────────────┘
```

---

## Datasets

| Dataset | Records | Purpose |
|---|---|---|
| LA Crime Data (2020–present) | 326,977 | LLM-1 training & LLM-3 clustering |
| India Murder Motives | 458 | LLM-2 regression training |
| Chicago Crime 2022 | 197,313 | LLM-3 behavioral clustering |
| **Total** | **524,748** | |

### LA Crime Dataset Columns Used
`date_occ`, `time_occ`, `area_name`, `crm_cd_desc`, `vict_age`, `vict_sex`, `weapon_used_cd`, `premis_cd`, `status`

### India Murder Motives Columns
`state`, `year`, `gain`, `dowry`, `insanity`, `provocation`, `quarrel`, `communalism`, `casteism`, `witchcraft`, `love_affairs`, `terrorists`, `other_motives`, `total`

---

## Models

### LLM-1 — Motivation Classifier
- **Architecture:** `all-MiniLM-L6-v2` sentence embeddings → `LogisticRegression(class_weight='balanced')`
- **Labels:** `emotional` | `financial` | `power`
- **Training samples:** 278,155
- **Train/test split:** `GroupShuffleSplit` by `crm_cd_desc` (entire crime types held out)
- **Real accuracy:** ~67% (original 100% was due to label leakage — see Known Issues)
- **Saved format:** `joblib` → `llm1_model/`

**Class distribution:**
```
power      → 171,537 samples (61.7%)
emotional  →  63,329 samples (22.8%)
financial  →  43,289 samples (15.6%)
```

### LLM-2 — Historical Predictor
- **Architecture:** `RandomForestRegressor(n_estimators=200)` wrapped in `MultiOutputRegressor`
- **Target:** 13 continuous motive count columns per Indian state/year
- **Training samples:** 458
- **MAE:** 40.98 (averaged across 13 output columns)
- **Saved format:** `joblib` → `llm2_model/`

### LLM-3 — Context Clusterer
- **Architecture:** `all-MiniLM-L6-v2` embeddings → `MiniBatchKMeans(n_clusters=5, batch_size=256)`
- **Records clustered:** 197,313 Chicago crimes
- **Embedding dimension:** 384
- **Cluster labels:** Aggressive | Opportunistic | Impulsive | Premeditated | Domestic
- **Saved format:** `embedder.save()` + `joblib` → `llm3_model/`

### LLM-4 — Synthesis Engine
- **Model:** Google Gemini 2.5 Flash
- **Input:** JSON outputs from LLM-1, LLM-2, LLM-3 + fusion score
- **Output:** Structured JSON forensic report with 6-step chain-of-thought reasoning
- **Retry logic:** 3 attempts with exponential backoff
- **Evaluation:** Qualitative only (no ground truth)

---

## Fusion Layer

The fusion layer combines LLM-1/2/3 outputs using confidence-weighted scores:

```python
weights = {"High": 0.95, "Medium": 0.65, "Low": 0.40}

fusion_score = sum(weight[conf] for each LLM)

# Final confidence:
if fusion_score >= 1.8:  → High
elif fusion_score >= 1.0: → Medium
else:                     → Low
```

**Typical confidence per model (demo runs):**

| Model | High | Medium | Low |
|---|---|---|---|
| LLM-1 | 75% | 20% | 5% |
| LLM-2 | 25% | 50% | 25% |
| LLM-3 | 0% | 75% | 25% |
| LLM-4 | 50% | 50% | 0% |

---

## Project Structure

```
Criminal_mind_analysis/
│
├── Criminal_mind_analysis_fixed.ipynb   # Main notebook (65 cells)
│
├── llm1_model/
│   ├── embedder/                        # SentenceTransformer weights
│   ├── classifier.joblib                # LogisticRegression
│   └── metadata.json
│
├── llm2_model/
│   ├── model.joblib                     # MultiOutputRegressor
│   ├── state_encoder.joblib             # LabelEncoder for states
│   └── metadata.json
│
├── llm3_model/
│   ├── embedder/                        # SentenceTransformer weights
│   ├── kmeans.joblib                    # MiniBatchKMeans
│   └── metadata.json
│
├── processed_crime_data.csv             # Pre-processed LA crimes
├── llm2_motivation_dataset.json         # 458 India records
├── llm3_clustered_data.json             # 197,313 clustered Chicago crimes
│
└── llm_dashboard_colab.py              # Inline Colab dashboard cell
```

---

## Setup & Usage

### Requirements

```bash
pip install sentence-transformers scikit-learn pandas numpy joblib
pip install google-generativeai
```

### Colab Secrets Required

Set these in Colab → 🔑 Secrets:

| Key | Purpose |
|---|---|
| `LLM1_API_KEY` | Gemini API key (pipeline stage 1) |
| `LLM2_API_KEY` | Gemini API key (pipeline stage 2) |
| `LLM3_API_KEY` | Gemini API key (pipeline stage 3) |
| `LLM4_API_KEY` | Gemini API key (synthesis stage) |

> All four keys can point to the same Gemini API key. Multiple keys are used to distribute rate limit quota.

### Execution Order

Run cells in order. Key cells:

| Cell | Purpose |
|---|---|
| 1–8 | Imports and dataset loading |
| 9–13 | LA crime pre-processing and label creation |
| 14–17 | **LLM-1 training** (GroupShuffleSplit fix applied) |
| 18–21 | **LLM-2 training** |
| 22–26 | **LLM-3 training** |
| 27–42 | Model loading and pipeline definition |
| 43 | `run_pipeline()` definition + fusion layer |
| 44–55 | GUI setup |
| 56–57 | **Demo runs** (ROBBERY, DOMESTIC VIOLENCE, ASSAULT, MURDER) |

### Running a Crime Analysis

```python
result = run_pipeline(
    crime_description="A 35-year-old male was found at the scene with a firearm",
    area="Hollywood",
    victim_age=35,
    victim_sex="M",
    hour=23,
    weapon="HANDGUN",
    location_type="STREET"
)

print(result['forensic_report'])
```

---

## Pre-Processing

### LA Crime (Cell 9–13)
1. Load 326,977 records from CSV
2. Impute missing values (weapon → `UNKNOWN`, sex → `X`)
3. Parse `date_occ` to datetime; extract hour from `time_occ`
4. Build `crime_text` string from: victim age, sex, weapon, area, location, hour, status
5. Assign labels via keyword rules on `crm_cd_desc`
6. Encode labels with `LabelEncoder`

### Chicago (Cell 22–24)
1. Load 197,313 records
2. Handle mixed-type columns (DtypeWarning on cols 0, 8, 9, 10)
3. Build `crime_text` similarly to LA pipeline
4. Embed and cluster with MiniBatchKMeans

### India Murder Motives (Cell 19–21)
1. Load 458 records
2. Fix double-underscore column names
3. Encode state names with LabelEncoder
4. Use year + state as features; 13 motive counts as targets

---

## Known Issues & Fixes Applied

### ⚠️ Critical: Label Leakage in LLM-1 (Fixed)

**Problem:** The original notebook included `crm_cd_desc` (crime type name) inside `crime_text`. Labels were derived from keywords in `crm_cd_desc`. The model memorized the mapping — achieving 100% accuracy without learning anything.

**Fix applied in `Criminal_mind_analysis_fixed.ipynb`:**
- Removed `crm_cd_desc` from `crime_text` construction (Cell 11)
- Replaced `train_test_split` with `GroupShuffleSplit(groups=crm_cd_desc)` to hold out entire crime types for testing (Cell 17)

**Result:** Accuracy dropped from 100% → ~67%, which represents genuine generalization.

### ⚠️ Duplicate Training Cells (Fixed)
- Cell 16 (original LLM-1 training, 81,736 samples, pickle) replaced with skip notice
- Cell 17 is the authoritative training cell (278,155 samples, joblib)

### Other Known Limitations

| Issue | Severity | Status |
|---|---|---|
| India data geographic mismatch for LA crimes | Moderate | By design — documented |
| k=5 clusters chosen without elbow method | Minor | Noted |
| CV F1 contaminated (ran on full dataset) | Moderate | Noted |
| google.generativeai deprecated (use google-genai) | Minor | Noted |
| Fusion thresholds hand-tuned, not calibrated | Moderate | Noted |
| LLM-4 has no quantitative evaluation metric | Critical | Inherent limitation |

For the full list of 44 observations, see `Criminal_Mind_Analysis_Observations.docx`.

---

## Model Serialization

| Model | Format | Path |
|---|---|---|
| LLM-1 SentenceTransformer | `embedder.save()` | `llm1_model/embedder/` |
| LLM-1 LogisticRegression | `joblib` | `llm1_model/classifier.joblib` |
| LLM-2 MultiOutputRegressor | `joblib` | `llm2_model/model.joblib` |
| LLM-2 LabelEncoder | `joblib` | `llm2_model/state_encoder.joblib` |
| LLM-3 SentenceTransformer | `embedder.save()` | `llm3_model/embedder/` |
| LLM-3 MiniBatchKMeans | `joblib` | `llm3_model/kmeans.joblib` |

> **Note:** Never use `pickle` or `joblib` for SentenceTransformer objects. Always use `.save()` + `SentenceTransformer(path)` for reload to ensure compatibility across library versions.

---

## Dashboard

Run `llm_dashboard_colab.py` as a cell in your notebook (after Cell 57) to display an interactive HTML dashboard showing:

- Arc gauges for accuracy / MAE / cluster count per model
- Confidence distribution bars (High / Medium / Low)
- LLM-1 training label distribution
- Fusion layer weight and contribution score table

---

## Technology Stack

| Component | Technology | Reason |
|---|---|---|
| Text embeddings | `all-MiniLM-L6-v2` | Fast, 384-dim, good semantic capture |
| Motivation classifier | `LogisticRegression` | Interpretable, fast inference on 278K samples |
| Historical predictor | `RandomForestRegressor` | Handles multi-output regression on small data |
| Behavioral clustering | `MiniBatchKMeans` | Memory-efficient for 197K × 384 embeddings |
| Synthesis | `Gemini 2.5 Flash` | Chain-of-thought reasoning, JSON output |
| Serialization | `joblib` | 3–10× faster than pickle for sklearn models |
| Runtime | Google Colab T4 GPU | Free GPU for embedding computation |

---

## Results Summary

| Model | Metric | Value |
|---|---|---|
| LLM-1 | Accuracy (after fix) | ~67% |
| LLM-1 | CV F1 (after fix) | ~0.59 |
| LLM-2 | MAE | 40.98 |
| LLM-3 | Clusters | 5 |
| LLM-4 | Evaluation | Qualitative only |

---

## Project Documents

| Document | Contents |
|---|---|
| `Criminal_mind_analysis_fixed.ipynb` | Main notebook with bias fix applied |
| `Criminal_Mind_Analysis_Project_Report.docx` | Full 10-section technical report |
| `Criminal_Mind_Analysis_Observations.docx` | 44 observations across 7 categories |
| `llm_dashboard_colab.py` | Colab visualization dashboard cell |

---

## Author Notes

This project was built as an exploration of multi-model LLM fusion for forensic analysis. The key engineering lesson: **always verify that your model inputs and outputs are truly independent**. The 100% accuracy from label leakage was a valuable reminder that impressive metrics without careful data hygiene are meaningless.

The India → LA dataset transfer is an acknowledged limitation — real deployment would require a geographically appropriate historical dataset. The current setup demonstrates the pipeline architecture even if the predictions from LLM-2 lack direct applicability to LA crimes.

---

*Built with Python 3.12 | Google Colab T4 | 3 Datasets | 4 Models | 524,748 Records*
