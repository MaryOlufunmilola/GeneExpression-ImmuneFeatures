# Correctness and validation fixes

Branch `fix/correctness-validation`, three commits on top of `f1c2b7e` (pytorch branch, Aug 2023).

## 1. deepImmune code fixes (`model/net.py`, `model/data_generator.py`, `utils.py`, `train.py`, `evaluate.py`)

| Problem | Effect before | Fix |
|---|---|---|
| `regularized_loss` matched names ending in both "weight" and "linear" | L1/L2 never applied | Penalizes `nn.Linear` weights; with a task index, only that task's row of the final layer |
| Label selection used `hasattr(loss_fn, '__name__')` then an `is` test | A named non-Cox loss reused the previous task's label | Explicit `is_cox_loss()` branch |
| `is` used to compare strings | Works only by interpreter accident | `==` |
| Survival-time normalization indexed column `ii` (loop counter) | Survival times never rank-normalized within cancer type; another column was | Uses the survival-time column indices |
| `qnorm_array` zeroed a copy for constant columns | Constant columns returned unchanged | Zeroes the returned array |
| Binary targets not checked | 1/2 coding makes BCELoss unbounded below; outputs collapse to 1 | `ValueError` unless 0/1/NaN |
| `processDataLabels(normalize=True)` shadowed `normalize()` | Crash if called | Argument renamed `do_normalize` |
| `MSELossClip` treated as classification in uncertainty weighting | Wrong task weight | Counted as regression |
| Broken dead code | Import-time risk, confusion | Removed `FullConnectedBlock_v0`, `make_layers_LSTM`, duplicate `outputLayer_old` / `outputLayer_incomplete` |
| `import ipdb`, `import umap` at module level | Hard dependencies for debugging/plotting | Lazy imports |
| `eval()` / `exec()` on config strings | Arbitrary code from configs and CLI | `ast.literal_eval` plus `range`/`list`/`len`; explicit lookup for model class, aggregate and continuous loss |
| Only torch seeded; data shuffling used NumPy | Runs not reproducible | `--seed` seeds Python, NumPy and PyTorch |

**Behaviour changes to expect when re-running old configs:**
- configs with non-zero `l1_regularizer` / `l2_regularizer` now actually regularize;
- survival times are now rank-normalized within cancer type, as the code intended;
- data files with 1/2-coded binary targets now stop with an error instead of training silently.

Old results are not reproducible bit-for-bit with this branch.

## 2. Tests and CI

`tests/test_fixes.py` (pytest) covers: Cox partial likelihood against a hand-computed value and row-order invariance; regularization on/off and per-task rows; missing-label masking for continuous, binary and survival tasks; uncertainty-weighting coefficients and masking; binary-label validation; survival-time normalization column; config parsing; seeding; patient and cohort leakage detection.

`.github/workflows/tests.yml` runs them on push and pull request.

## 3. Leave-one-cohort-out inputs (`data_processing/preprocess_loco.R`)

Rebuilds deepImmune inputs from the Data11 compendium without leakage. One folder per held-out study, with `dataset_train.txt`, `dataset_val.txt` (tuning, from training studies only), `dataset_test.txt` (held-out study only) and `column_index.tsv` (column name → deepImmune index). Also writes `table1_cohorts.tsv` and `fold_summary.tsv`.

Check the output with `python scripts/check_splits.py Data11/loco/folds`.

## 4. Survival analysis (`scripts/survival_analysis.py`)

Replaces `deepImmune_coxregr.sh`: factors selected on one split and tested on another, Cox models stratified by cancer type, factors z-scored within stratum, one BH-FDR per table, PH test kept, measured vs predicted factor comparison, per-stratum heatmap.

## Running

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt
pytest -q tests

Rscript data_processing/preprocess_loco.R        # edit cfg$data_dir / cfg$out_dir first
python scripts/check_splits.py Data11/loco/folds

python train.py --seed 230 --data_dir <datasets_list.txt> --model_dir <dir>
python evaluate.py --seed 230 --type_file test --data_dir <datasets_list.txt> --model_dir <dir> --restore_file <best.pth.tar>
```
