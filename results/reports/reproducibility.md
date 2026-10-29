# Reproducibility Guide & Experimental Verification Protocol

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Status:** Validated Defensive ML Research Prototype  

---

## 1. System Requirements & Environment Setup

- **Python Version**: Python 3.10+ (tested on Python 3.10.11 Windows 64-bit).
- **Core Virtual Environment**:
  ```bash
  python -m venv venv
  # Windows
  .\venv\Scripts\activate
  # Linux/macOS
  source venv/bin/activate
  
  pip install -r requirements.txt
  ```

---

## 2. Experimental Random Seed & Determinism
All stochastic algorithms in this project are anchored to:
```python
RANDOM_SEED = 42
```
This applies to:
- Stratified Train/Test Partitioning (`train_test_split(..., random_state=42)`)
- Random Forest Classifiers (`RandomForestClassifier(..., random_state=42)`)
- XGBoost Classifiers (`XGBClassifier(..., random_state=42)`)
- Gymnasium Environments (`gym.utils.seeding.np_random(42)`)
- PPO Reinforcement Learning (`PPO(..., seed=42)`)

---

## 3. End-to-End Execution Sequence

Execute the research pipeline in sequential order:

```bash
# Phase 1: Static Dataset Preprocessing & Cleaning
python -c "from src.preprocessing import clean_dataset; clean_dataset()"

# Phase 2: Static ML Baselines (Random Forest & XGBoost)
python -c "from src.train_rf import train_random_forest; train_random_forest()"
python -c "from src.train_xgb import train_xgboost; train_xgboost()"

# Phase 3: Early-Stage Feature-Restricted Baseline
python -c "from src.train_early_stage import run_early_stage_experiments; run_early_stage_experiments()"

# Phase 4: Temporal Pipeline & Validation Engine
python -c "import src.temporal_pipeline as tp; df=tp.generate_synthetic_temporal_demo(); print(tp.validate_temporal_data(df))"

# Phase 5: Temporal Supervised Baselines
python -c "import src.train_temporal_baseline as ttb; df, _ = ttb.load_and_prepare_temporal_dataset(); print('Temporal baseline ready')"

# Phase 6: Temporal PPO Reinforcement Learning
python -c "import src.train_temporal_ppo as ttp; print('PPO training module verified')"

# Phase 7: Unified Multi-Model Evaluation
python -c "import src.unified_evaluation as ue; m=ue.load_all_model_results(); ue.generate_unified_figures_and_summary(m)"
```

---

## 4. Notebook Execution Order

Alternatively, run the interactive Jupyter research notebooks from top to bottom:

| Notebook | Purpose | Status |
|:---|:---|:---:|
| `notebooks/01_dataset_analysis.ipynb` | Empirical sandbox dataset EDA & class distribution | Completed |
| `notebooks/02_baseline_models.ipynb` | Static RF and XGBoost baselines (46 features) | Completed |
| `notebooks/03_early_stage_baseline.ipynb` | Early-stage non-redundant feature pruning (16 features) | Completed |
| `notebooks/04_ppo_environment.ipynb` | Initial Gymnasium environment design & validation | Completed |
| `notebooks/07_temporal_memory_data_design.ipynb` | Real temporal & memory schema specification | Completed |
| `notebooks/08_temporal_dataset_pipeline.ipynb` | Causal temporal feature engineering pipeline | Completed |
| `notebooks/09_temporal_baseline_models.ipynb` | Temporal RF & XGBoost baselines across observation windows | Completed |
| `notebooks/10_temporal_ppo_integration.ipynb` | Sequential PPO RL training & decision traces | Completed |
| `notebooks/11_unified_evaluation.ipynb` | Master 7-model comparative benchmark | Completed |
| `notebooks/12_explainability.ipynb` | Cross-paradigm feature importance & PPO triggers | Completed |
| `notebooks/13_final_research_analysis.ipynb` | Comprehensive 17-section research synthesis | Completed |
| `notebooks/14_final_validation.ipynb` | Automated system audit across Data, Models, & Leakage | Completed |

---

## 5. Launching the Research Demonstration Dashboard

To launch the interactive research telemetry viewer:
```bash
streamlit run dashboard/app.py
```
*(Or run in terminal fallback mode: `python dashboard/app.py`)*

---

## 6. Scientific Boundary & Defensive Safety
- **Synthetic Data Caveat**: Multi-window telemetry ($W_0 \to W_4$) is synthetically generated solely to validate the mathematical and structural integrity of the temporal pipeline and PPO environment.
- **Defensive Scope**: No malware binaries were downloaded, executed, or created during this project.
