"""Survival associations of deepImmune-predicted immune factors.

Replaces the analysis in deepImmune_coxregr.sh. Differences:
  * factors are selected on one split (--select-pred) and tested on another
    (--test-pred), so selection and testing never use the same samples
  * Cox models are stratified by cancer type / cohort (separate baseline
    hazards), instead of pooling all cohorts in one model
  * each factor is z-scored within stratum, so hazard ratios are per 1 SD
    and comparable across factors
  * Benjamini-Hochberg FDR is applied once across all tests in a table
    (previously it was applied to one p-value at a time, i.e. not at all)
  * the proportional-hazards test result is kept in the output table
  * predicted and measured factors are both modelled where measured values
    exist, so their associations can be compared
  * per-cancer-type models for strata with enough events, for the heatmap
  * survival time and event come from the dataset file (raw values), joined
    by row number, not from the prediction file (where time is rank-normalized)

Inputs are the files written by deepImmune's evaluate.py
(<split>_<i>_prediction.csv, tab-separated; first column sample_name = row
number in the dataset file) and the matching dataset_<split>.txt files.

Example:
    python scripts/survival_analysis.py \
        --select-pred out/val_0_prediction.csv  --select-data data/dataset_val.txt \
        --test-pred   out/test_0_prediction.csv --test-data   data/dataset_test.txt \
        --outdir results/survival
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUTPUT_SUFFIX = ".output"


# ---------------------------------------------------------------- helpers
def bh_fdr(pvalues):
    """Benjamini-Hochberg adjusted p-values; NaNs stay NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def zscore_within(values, groups):
    """z-score a Series within each group; constant groups become 0."""
    def z(x):
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd > 0 else x * 0.0
    return values.groupby(groups).transform(z)


def load_split(pred_path, data_path, time_col, event_col, strata_col):
    """Prediction file joined to raw survival and stratum from the dataset file."""
    pred = pd.read_csv(pred_path, sep="\t", low_memory=False)
    data = pd.read_csv(data_path, sep="\t", low_memory=False)
    rows = pred["sample_name"].astype(int).to_numpy()
    if rows.max() >= len(data):
        raise ValueError("sample_name exceeds rows in {}; wrong dataset file?".format(data_path))
    info = data.iloc[rows].reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    for col in [time_col, event_col]:
        if col not in info.columns:
            raise KeyError("{} not found in {}".format(col, data_path))
        pred["_" + col] = pd.to_numeric(info[col], errors="coerce").to_numpy()
    first = data.columns[0]
    pred["_stratum"] = info[strata_col if strata_col in info.columns else first].astype(str).to_numpy()
    return pred


def factor_names(pred, time_col, event_col):
    """Factors with both a measured column and a predicted '.output' column."""
    names = []
    for col in pred.columns:
        if col.endswith(OUTPUT_SUFFIX):
            base = col[: -len(OUTPUT_SUFFIX)]
            if base in pred.columns and base not in (time_col, event_col):
                names.append(base)
    return names


def select_factors(pred, names, min_corr, min_n):
    """Spearman correlation of predicted vs measured values on the selection split."""
    rows = []
    for f in names:
        a = pd.to_numeric(pred[f], errors="coerce")
        b = pd.to_numeric(pred[f + OUTPUT_SUFFIX], errors="coerce")
        ok = a.notna() & b.notna()
        n = int(ok.sum())
        rho = spearmanr(a[ok], b[ok]).correlation if n >= min_n and a[ok].nunique() > 1 else np.nan
        rows.append({"factor": f, "n_selection": n, "spearman_rho": rho,
                     "selected": bool(n >= min_n and rho >= min_corr)})
    return pd.DataFrame(rows)


def fit_cox(df, covariate, strata=True):
    """Univariate Cox model (stratified by _stratum when strata=True)."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    cols = ["_time", "_event", covariate] + (["_stratum"] if strata else [])
    d = df[cols].dropna()
    d = d[d["_time"] > 0]
    res = {"n": len(d), "events": int(d["_event"].sum()) if len(d) else 0}
    if res["events"] < 5 or d[covariate].nunique() < 2:
        return res
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="_time", event_col="_event",
                strata=["_stratum"] if strata else None)
        try:
            ph = proportional_hazard_test(cph, d, time_transform="rank").summary["p"].iloc[0]
        except Exception:
            ph = np.nan
    s = cph.summary.loc[covariate]
    res.update({"log_hr": s["coef"], "hr": s["exp(coef)"],
                "hr_lower95": s["exp(coef) lower 95%"], "hr_upper95": s["exp(coef) upper 95%"],
                "p": s["p"], "ph_test_p": ph})
    return res


# ---------------------------------------------------------------- analysis
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    sel = load_split(args.select_pred, args.select_data, args.time_col, args.event_col, args.strata_col)
    test = load_split(args.test_pred, args.test_data, args.time_col, args.event_col, args.strata_col)
    if os.path.samefile(args.select_data, args.test_data):
        raise ValueError("Selection and test splits are the same file; use separate splits.")

    names = factor_names(sel, args.time_col, args.event_col)
    selection = select_factors(sel, names, args.min_corr, args.min_n)
    selection.to_csv(os.path.join(args.outdir, "factor_selection.tsv"), sep="\t", index=False)
    chosen = selection.loc[selection["selected"], "factor"].tolist()
    print("{} of {} factors selected (Spearman >= {} on the selection split)".format(
        len(chosen), len(names), args.min_corr))
    if not chosen:
        return

    test = test.rename(columns={"_" + args.time_col: "_time", "_" + args.event_col: "_event"})
    rows, per_cancer = [], []
    for f in chosen:
        for kind, col in [("predicted", f + OUTPUT_SUFFIX), ("measured", f)]:
            vals = pd.to_numeric(test[col], errors="coerce")
            if vals.notna().sum() < args.min_n:
                continue
            zcol = "_z"
            test[zcol] = zscore_within(vals, test["_stratum"])
            res = fit_cox(test, zcol, strata=True)
            rows.append({"factor": f, "values": kind, **res})
            if kind == "predicted":
                for stratum, d in test.groupby("_stratum"):
                    if d["_event"].sum() >= args.min_events_per_stratum:
                        r = fit_cox(d, zcol, strata=False)
                        per_cancer.append({"factor": f, "stratum": stratum, **r})

    pooled = pd.DataFrame(rows)
    if "p" in pooled:
        for kind in ["predicted", "measured"]:
            m = pooled["values"] == kind
            pooled.loc[m, "fdr"] = bh_fdr(pooled.loc[m, "p"])
    pooled.to_csv(os.path.join(args.outdir, "cox_stratified.tsv"), sep="\t", index=False)

    per = pd.DataFrame(per_cancer)
    if len(per) and "p" in per:
        per["fdr"] = bh_fdr(per["p"])
    per.to_csv(os.path.join(args.outdir, "cox_per_stratum.tsv"), sep="\t", index=False)

    make_plots(pooled, per, test, args)
    print("Results written to", args.outdir)


# ---------------------------------------------------------------- plots
def make_plots(pooled, per, test, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = pooled[(pooled["values"] == "predicted")].dropna(subset=["p"]) if "p" in pooled else pd.DataFrame()
    if len(pred):
        # volcano: log2 HR (per SD) vs -log10 p, FDR < 0.05 highlighted
        fig, ax = plt.subplots(figsize=(7, 5))
        sig = pred["fdr"] < 0.05
        x = np.log2(pred["hr"])
        y = -np.log10(pred["p"])
        ax.scatter(x[~sig], y[~sig], s=12, color="0.6", label="FDR ≥ 0.05")
        ax.scatter(x[sig], y[sig], s=14, color="C3", label="FDR < 0.05")
        for _, r in pred[sig].nsmallest(15, "p").iterrows():
            ax.annotate(r["factor"], (np.log2(r["hr"]), -np.log10(r["p"])), fontsize=6,
                        xytext=(2, 3), textcoords="offset points")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("log2 hazard ratio per SD (stratified Cox)")
        ax.set_ylabel("-log10 p")
        ax.set_title("Predicted immune factors and overall survival")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "volcano.png"), dpi=300); plt.close(fig)

        # Kaplan-Meier, top factors, median split within stratum
        from lifelines import KaplanMeierFitter
        top = pred.nsmallest(6, "p")
        ncol = min(3, len(top)); nrow = int(np.ceil(len(top) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow), squeeze=False)
        for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
            v = pd.to_numeric(test[r["factor"] + OUTPUT_SUFFIX], errors="coerce")
            d = test.assign(_v=v).dropna(subset=["_v", "_time", "_event"])
            high = d.groupby("_stratum")["_v"].transform(lambda s: s > s.median())
            for label, mask, color in [("High", high, "C3"), ("Low", ~high, "C0")]:
                KaplanMeierFitter().fit(d.loc[mask, "_time"], d.loc[mask, "_event"], label=label).plot_survival_function(
                    ax=ax, ci_show=False, color=color)
            ax.set_title("{}\nHR/SD {:.2f}, FDR {:.2g}".format(r["factor"], r["hr"], r["fdr"]), fontsize=8)
            ax.set_xlabel(args.time_col); ax.set_ylabel("survival probability")
            ax.legend(title="within-stratum median split", fontsize=7, title_fontsize=7, frameon=False)
        for ax in axes.ravel()[len(top):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "km_top_factors.png"), dpi=300); plt.close(fig)

        # predicted vs measured hazard ratios
        meas = pooled[pooled["values"] == "measured"].dropna(subset=["p"])
        both = pred.merge(meas, on="factor", suffixes=("_pred", "_meas"))
        if len(both):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(both["log_hr_meas"], both["log_hr_pred"], s=14)
            lim = np.nanmax(np.abs(np.r_[both["log_hr_meas"], both["log_hr_pred"]])) * 1.1
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax.set_xlabel("log HR per SD, measured factor"); ax.set_ylabel("log HR per SD, predicted factor")
            fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "predicted_vs_measured_hr.png"), dpi=300); plt.close(fig)

    if len(per) and "log_hr" in per:
        import seaborn as sns
        mat = per.pivot_table(index="factor", columns="stratum", values="log_hr")
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * mat.shape[1]), max(4, 0.25 * mat.shape[0])))
        lim = np.nanmax(np.abs(mat.values))
        sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, ax=ax,
                    cbar_kws={"label": "log hazard ratio per SD"})
        ax.set_xlabel("cancer type / cohort"); ax.set_ylabel("predicted immune factor")
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "heatmap_log_hr.png"), dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-pred", required=True, help="prediction file used only to select factors")
    ap.add_argument("--select-data", required=True, help="dataset file matching --select-pred")
    ap.add_argument("--test-pred", required=True, help="prediction file used for survival models")
    ap.add_argument("--test-data", required=True, help="dataset file matching --test-pred")
    ap.add_argument("--outdir", default="survival_results")
    ap.add_argument("--time-col", default="OS")
    ap.add_argument("--event-col", default="OS_EVENT")
    ap.add_argument("--strata-col", default="cancertype")
    ap.add_argument("--min-corr", type=float, default=0.3)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--min-events-per-stratum", type=int, default=10)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
