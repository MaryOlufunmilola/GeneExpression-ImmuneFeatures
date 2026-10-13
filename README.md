# GeneExpression-ImmuneFeatures

Does training a model on immune phenotypes and survival, alongside immunotherapy response, help it predict response in patient cohorts it has never seen?

This repository tests that question with **deepImmune**, a multi-task deep learning model by Avinash Sahu, Keith Flaherty and X. Shirley Liu. It is a copy of the `pytorch` branch of [vinash85/deepImmune](https://github.com/vinash85/deepImmune), used and modified with Avinash Sahu's permission.

Maintainer: Olufunmilola Oyebamiji

## Status

| Part | State |
|---|---|
| Bug fixes to deepImmune, with tests (23 passing) | Done |
| Leakage-free leave-one-study-out preprocessing (`data_processing/preprocess_loco.R`) | Written; first run on real data pending |
| Split leakage checker (`scripts/check_splits.py`) | Done |
| Stratified survival analysis (`scripts/survival_analysis.py`) | Written; tested on simulated data |
| Per-fold training configs and run script | Not yet written |
| Baselines and main experiment | Not yet started |

## What changed from upstream

See [FIXES.md](FIXES.md). In short:

- L1/L2 regularization now applies (it never did before).
- Survival times are rank-normalized within cancer type, as intended.
- Binary labels must be 0/1; 1/2 coding now raises an error.
- Runs are reproducible with `--seed`.
- Config values are parsed safely instead of with `eval()`.
- Broken dead code is removed.
- A pytest suite runs on every push.

Old results from upstream code will not reproduce exactly with these fixes.

## Setup

```bash
conda create -n deepimmune python=3.11 -y
conda activate deepimmune
pip install torch                 # Linux without a GPU: add --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt
pytest -q tests                   # expect 23 passed
```

The preprocessing step also needs R with `data.table`, `readxl` and `biomaRt`:

```bash
Rscript -e 'install.packages(c("data.table","readxl"), repos="https://cloud.r-project.org"); install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install("biomaRt", ask=FALSE)'
```

## Data

No data is stored in this repository. `preprocess_loco.R` expects one folder holding per-dataset files, found by folder name at any depth:

```
<data_dir>/
  bulkrna_ICB/expression/<dataset>       genes x samples; first column EntrezID, Ensembl_id or gene symbol
  bulkrna_ICB/follow_up/<dataset>        clinical table; same file name as the expression file
  bulkrna_NonICB/expression/TCGA-<type>
  bulkrna_NonICB/follow_up/TCGA-<type>
  DatafileS1.xlsx ... DatafileS4.xlsx    TCGA immune features (optional multi-task targets)
```

Dataset names follow `<Study><Year>_<Drug>_<Cancer>_<Platform>[_Pre|_Post]`, for example `Riaz2017_PD1_Melanoma_RNASeq_Ipi.Naive_Pre`.

By default the script:

- uses pre-treatment samples from ICB cohorts as test folds, one study at a time;
- uses TCGA as extra training data;
- excludes NanoString and gene-panel cohorts.

All of these can be changed in the `cfg` block at the top of the script.

## Running

1. Edit `cfg` at the top of `data_processing/preprocess_loco.R`. Set `data_dir`, `out_dir` and `annotation_rds` as full paths.
2. Run:

```bash
Rscript data_processing/preprocess_loco.R 2>&1 | tee preprocess_log.txt
python scripts/check_splits.py <out_dir>/folds
```

Outputs in `<out_dir>`:

- `folds/<study>/dataset_train.txt`: training cohorts.
- `folds/<study>/dataset_val.txt`: a seeded 15% patient-level split of the training cohorts, used for tuning.
- `folds/<study>/dataset_test.txt`: the held-out study only.
- `folds/<study>/column_index.tsv`: column name → deepImmune index, for writing `params.json`.
- `table1_cohorts.tsv`: patients and responders per cohort.
- `fold_summary.tsv`: genes, train, validation and test sizes per fold.

Training and evaluation use the upstream scripts, now with `--seed`:

```bash
python train.py    --seed 230 --data_dir <datasets_list.txt> --model_dir <dir with params.json>
python evaluate.py --seed 230 --data_dir <datasets_list.txt> --model_dir <dir> --restore_file <best.pth.tar>
```

Survival associations of predicted immune factors are computed with `scripts/survival_analysis.py`. Run `python scripts/survival_analysis.py --help` for its options.

## Repository layout

The files used by this project:

| Path | Purpose |
|---|---|
| `train.py`, `evaluate.py`, `utils.py`, `r2python.py` | Training and evaluation |
| `model/net.py`, `model/data_generator.py` | Model, losses, metrics, data loading |
| `config_files/` | Example `params.json` and dataset lists |
| `data_processing/preprocess_loco.R` | Leakage-free input files |
| `scripts/check_splits.py`, `scripts/survival_analysis.py` | Split checks, survival analysis |
| `tests/` | pytest suite |

The other files come from upstream and are not used here. They include the attention model, the hyperparameter-search tools, `data_processing/ssgsea/` and notebooks.

## Credit and licence

deepImmune was written by Avinash Das Sahu, Keith Flaherty and X. Shirley Liu ([vinash85/deepImmune](https://github.com/vinash85/deepImmune)). Its PyTorch code started from the [CS230 project starter code](https://cs230-stanford.github.io/project-starter-code.html).

This repository keeps the original GNU GPL-3.0 licence (see [LICENSE](LICENSE)).
