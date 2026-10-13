"""Check leave-one-cohort-out split files for leakage.

Each fold directory must contain dataset_train.txt and dataset_test.txt
(tab-separated, with PATIENT_ID and COHORT columns, as written by
preprocess_loco.R). The check fails if any patient or cohort appears in both
the training and the test file of the same fold.

Usage:
    python scripts/check_splits.py <folds_root>
"""
import os
import sys

import pandas as pd


def find_overlap(train, test, column):
    """Return the sorted values of `column` present in both data frames."""
    if column not in train.columns or column not in test.columns:
        raise KeyError("column {!r} missing from split files".format(column))
    shared = set(train[column].dropna().astype(str)) & set(test[column].dropna().astype(str))
    return sorted(shared)


def check_fold(fold_dir, patient_col="PATIENT_ID", cohort_col="COHORT"):
    """Return a list of problems found in one fold directory (empty = clean)."""
    train = pd.read_csv(os.path.join(fold_dir, "dataset_train.txt"), sep="\t", low_memory=False)
    test = pd.read_csv(os.path.join(fold_dir, "dataset_test.txt"), sep="\t", low_memory=False)
    problems = []
    patients = find_overlap(train, test, patient_col)
    if patients:
        problems.append("{} patients in both train and test (e.g. {})".format(len(patients), patients[:5]))
    cohorts = find_overlap(train, test, cohort_col)
    if cohorts:
        problems.append("cohorts in both train and test: {}".format(cohorts))
    if test[cohort_col].nunique() != 1:
        problems.append("test file should hold exactly one cohort, found {}".format(test[cohort_col].nunique()))
    return problems


def main(root):
    folds = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not folds:
        print("No fold directories found in {}".format(root))
        return 1
    failed = 0
    for fold in folds:
        problems = check_fold(os.path.join(root, fold))
        status = "OK" if not problems else "LEAK"
        print("{:<50} {}".format(fold, status))
        for p in problems:
            print("    - " + p)
        failed += bool(problems)
    print("{} of {} folds have leakage".format(failed, len(folds)))
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
