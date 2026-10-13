"""Tests for the correctness and validation fixes.

Run from the repository root:
    pytest -q tests
"""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import utils
import model.net as net
import model.data_generator as dg
from scripts.check_splits import check_fold, find_overlap


# --------------------------------------------------------------------------
# Cox partial likelihood
# --------------------------------------------------------------------------
def test_cox_loss_matches_hand_computed_value():
    # times 5, 3, 8, 1 ; events 1, 0, 1, 1 ; risks 0.2, -0.1, 0.5, 0.3
    # Events: t=5 -> risk set {5, 8}; t=8 -> {8}; t=1 -> all four.
    # -(1/3) * sum_i [risk_i - log(sum_{j: t_j >= t_i} exp(risk_j))]
    expected = 0.7294835655266186
    risk = torch.tensor([0.2, -0.1, 0.5, 0.3])
    survival = torch.tensor([[5., 1.], [3., 0.], [8., 1.], [1., 1.]])
    loss = net.negative_log_partial_likelihood_loss(risk, survival)
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_cox_loss_does_not_depend_on_row_order():
    risk = torch.tensor([0.2, -0.1, 0.5, 0.3])
    survival = torch.tensor([[5., 1.], [3., 0.], [8., 1.], [1., 1.]])
    perm = torch.tensor([2, 0, 3, 1])
    a = net.negative_log_partial_likelihood_loss(risk, survival)
    b = net.negative_log_partial_likelihood_loss(risk[perm], survival[perm])
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_is_cox_loss():
    assert net.is_cox_loss(net.negative_log_partial_likelihood_loss)
    assert not net.is_cox_loss(nn.MSELoss())
    assert not net.is_cox_loss(net.accuracy)  # a plain function that is not Cox


# --------------------------------------------------------------------------
# Regularization (previously never applied)
# --------------------------------------------------------------------------
def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_regularization_applies_when_enabled():
    model = _toy_model()
    params = SimpleNamespace(l1_regularizer=0.1, l2_regularizer=0.0)
    expected = 0.1 * sum(m.weight.abs().sum().item() for m in model if isinstance(m, nn.Linear))
    assert float(net.regularized_loss(model, params)) == pytest.approx(expected, rel=1e-6)


def test_regularization_zero_when_disabled():
    params = SimpleNamespace(l1_regularizer=0.0, l2_regularizer=0.0)
    assert net.regularized_loss(_toy_model(), params) == 0.


def test_regularization_with_index_only_touches_that_output_row():
    model = _toy_model()
    params = SimpleNamespace(l1_regularizer=1.0, l2_regularizer=0.0)
    net.regularized_loss(model, params, index=1).backward()
    final, first = model[2], model[0]
    assert torch.all(final.weight.grad[0] == 0)
    assert torch.any(final.weight.grad[1] != 0)
    assert first.weight.grad is None or torch.all(first.weight.grad == 0)


# --------------------------------------------------------------------------
# Missing-label masking
# --------------------------------------------------------------------------
def test_missing_labels_are_masked_per_task():
    mse = nn.MSELoss()
    labels = torch.tensor([[1.0, 0.], [float("nan"), 1.], [3.0, float("nan")], [2.0, 1.]])
    outputs = torch.tensor([[0.5, 0.2], [9.0, 0.7], [2.0, 0.4], [2.5, 0.9]])
    loss = net.calculate_loss(labels, outputs, [mse, nn.BCELoss()])
    keep0 = [0, 2, 3]
    keep1 = [0, 1, 3]
    expected = mse(outputs[keep0, 0], labels[keep0, 0]) + nn.BCELoss()(outputs[keep1, 1], labels[keep1, 1])
    assert float(loss) == pytest.approx(expected.item(), rel=1e-6)


def test_survival_labels_masked_and_use_two_columns():
    labels = torch.tensor([[5., 1., 0.3], [3., float("nan"), 0.1], [8., 1., 0.2], [1., 1., 0.4]])
    outputs = torch.tensor([[0.2, 0.3], [-0.1, 0.1], [0.5, 0.2], [0.3, 0.5]])
    fns = [net.negative_log_partial_likelihood_loss, nn.MSELoss()]
    loss = net.calculate_loss(labels, outputs, fns)
    keep = [0, 2, 3]
    cox = net.negative_log_partial_likelihood_loss(outputs[keep, 0], labels[keep, 0:2])
    mse = nn.MSELoss()(outputs[:, 1], labels[:, 2])
    assert float(loss) == pytest.approx((cox + mse).item(), rel=1e-5)


# --------------------------------------------------------------------------
# Multi-task uncertainty weighting
# --------------------------------------------------------------------------
def test_multitask_loss_weights_and_mask():
    is_regression = torch.tensor([True, True, False])
    mtl = net.MultiTaskLoss(is_regression, reduction="sum")
    losses = torch.tensor([2.0, 4.0, 6.0])
    mask = torch.tensor([1.0, 0.0, 1.0])
    # log_vars start at 0 -> std 1: regression weight 1/2, classification 1, log(std) 0
    assert mtl(losses, mask).item() == pytest.approx(0.5 * 2.0 + 1.0 * 6.0)


def test_mseclip_counts_as_regression_in_mtl():
    params = SimpleNamespace(mtl=True, loss_fns=[net.MSELossClip(1.0), nn.BCELoss()])
    params = net.initializeMTL(params)
    assert params.mtlloss.is_regression.tolist() == [True, True, False]


# --------------------------------------------------------------------------
# Labels and normalization
# --------------------------------------------------------------------------
def test_binary_labels_coded_1_2_are_rejected():
    with pytest.raises(ValueError, match="0/1"):
        dg.check_binary_labels(np.array([[1.], [2.], [np.nan]]))


def test_binary_labels_0_1_nan_pass():
    dg.check_binary_labels(np.array([[0.], [1.], [np.nan]]))


def test_qnorm_constant_column_becomes_zero():
    out = dg.qnorm_array(np.array([3.0, 3.0, np.nan, 3.0]))
    assert np.allclose(out[[0, 1, 3]], 0) and np.isnan(out[2])


def test_survival_time_is_rank_normalized_within_cancer_type():
    # columns: gene0, gene1, OS time, OS event
    rng = np.random.default_rng(1)
    n = 40
    data = np.column_stack([rng.normal(size=n), rng.normal(size=n),
                            np.concatenate([np.arange(1, 21) * 10., np.arange(1, 21) * 1000.]),
                            np.ones(n)])
    cancertype = np.array(["A"] * 20 + ["B"] * 20)
    params = SimpleNamespace(survival_indices=[2, 3], continuous_phenotype_indices=[],
                             binary_phenotype_indices=[], input_indices=[0, 1])
    _, _, _, gen = dg.generator_survival(data.copy(), data.copy(), params, cancertype=cancertype,
                                         shuffle=False, batch_size=n, normalize_input=True,
                                         tsne_labels_mat=np.zeros((n, 1)))
    X, y, _ = next(gen)
    time_a, time_b = y[:20, 0], y[20:, 0]
    # same within-type ranks -> identical normalized times for the two types
    assert np.allclose(time_a, time_b)
    assert np.all(np.diff(time_a) > 0)


def test_continuous_loss_spec_parsing():
    assert isinstance(net.make_continuous_loss("nn.MSELoss()"), nn.MSELoss)
    clip = net.make_continuous_loss("MSELossClip(clip=2.5)")
    assert isinstance(clip, net.MSELossClip) and clip.clip == 2.5
    with pytest.raises(ValueError):
        net.make_continuous_loss("__import__('os').system('ls')")


# --------------------------------------------------------------------------
# Config parsing without eval/exec, and seeding
# --------------------------------------------------------------------------
def test_safe_eval_literals_and_range():
    assert utils.safe_eval("[1, 2, 3]") == [1, 2, 3]
    assert utils.safe_eval("list(range(3))") == [0, 1, 2]
    with pytest.raises(ValueError):
        utils.safe_eval("__import__('os').getcwd()")


def test_apply_hyper_params():
    params = SimpleNamespace()
    utils.apply_hyper_params(params, "params.batch_size=32;params.binary_phenotype_indices=[42,43]")
    assert params.batch_size == 32 and params.binary_phenotype_indices == [42, 43]
    with pytest.raises(ValueError):
        utils.apply_hyper_params(params, "import os")


def test_set_seed_makes_shuffles_reproducible():
    utils.set_seed(7)
    a = (np.random.permutation(10), torch.rand(3))
    utils.set_seed(7)
    b = (np.random.permutation(10), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_resolve_embedding_model_and_aggregate():
    assert utils.resolve_embedding_model("net.EmbeddingNet", net) is net.EmbeddingNet
    with pytest.raises(ValueError):
        utils.resolve_embedding_model("os.system", net)
    assert utils.resolve_aggregate("min", net)([3, 1, 2]) == 1


# --------------------------------------------------------------------------
# Patient and cohort leakage in split files
# --------------------------------------------------------------------------
def _write_fold(tmp_path, train_rows, test_rows):
    cols = ["PATIENT_ID", "COHORT", "GENE1"]
    pd.DataFrame(train_rows, columns=cols).to_csv(tmp_path / "dataset_train.txt", sep="\t", index=False)
    pd.DataFrame(test_rows, columns=cols).to_csv(tmp_path / "dataset_test.txt", sep="\t", index=False)
    return str(tmp_path)


def test_clean_fold_passes(tmp_path):
    fold = _write_fold(tmp_path, [["p1", "Hugo", 1.0], ["p2", "Riaz", 2.0]], [["p9", "Gide", 3.0]])
    assert check_fold(fold) == []


def test_patient_leakage_is_detected(tmp_path):
    # same patient (e.g. pre- and post-treatment samples) on both sides
    fold = _write_fold(tmp_path, [["p1", "Riaz", 1.0]], [["p1", "Gide", 3.0]])
    assert any("patients" in p for p in check_fold(fold))


def test_cohort_leakage_is_detected(tmp_path):
    fold = _write_fold(tmp_path, [["p1", "Gide", 1.0]], [["p2", "Gide", 3.0]])
    assert any("cohorts" in p for p in check_fold(fold))


def test_find_overlap():
    a = pd.DataFrame({"x": ["1", "2"]})
    b = pd.DataFrame({"x": ["2", "3"]})
    assert find_overlap(a, b, "x") == ["2"]
