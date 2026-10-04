"""EquiAlgo mitigation: structural de-biasing of the historical committee.

1. Fit an interpretable model of the committee's decisions on every observed variable.
2. Split its index into merit (R score, hours worked) and bias (region, income).
3. Rank the 4,000 applicants on the merit part only and grant exactly RATE of them.
4. Sweep the share of bias kept (lambda) and compare with fairlearn mitigators to
   draw the Pareto front (pareto_front.png).

    python model_corrige.py                       # predictions.csv + Pareto + robustness
    python model_corrige.py --rate 0.41 --hours-mult 1.0 --skip-pareto
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fairlearn.postprocessing import ThresholdOptimizer  # noqa: E402
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import (  # noqa: E402
    ROOT, baseline_model, bias_score, check_budget, encode, eo_gap, fit_committee, group,
    load_data, merit_score, merit_weights, selection_gap, top_k,
)

warnings.filterwarnings("ignore", category=FutureWarning)

SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = np.linspace(0, 1, 11)
EG_EPS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
REF_NOISE_SD = 1.0      # noise of the simulated reference, in R-score points
BASELINE_EO_GAP = 0.270  # official baseline gap against the hidden reference
RANDOM_AGREEMENT = 0.4 * 0.4 + 0.6 * 0.6  # random 40 % draw vs a 40 % reference


def simulated_reference(df, weights, rng, rate=0.40):
    """Reference labels under H1: top `rate` of merit + noise. The reference hours weight
    is drawn in [0.5, 1.5] x ours so that the evaluation is not circular."""
    ref_weights = {"r": 1.0, "h": weights["h"] * rng.uniform(0.5, 1.5)}
    merit = merit_score(df, ref_weights)
    return top_k(merit + rng.normal(0, REF_NOISE_SD, len(merit)), rate)


def final_predictions(hist, cand, rate, hours_mult, log_inc_weight=0.0):
    committee = fit_committee(hist)
    weights = merit_weights(committee)
    weights["h"] *= hours_mult
    weights["log_inc"] = log_inc_weight
    pred = top_k(merit_score(cand, weights), rate)
    check_budget(pred)
    return pred, committee, weights


def report_final(cand, pred, weights):
    g = group(cand)
    print(f"\nGrant rate {pred.mean():.2%}  (Centre {pred[g == 'Centre'].mean():.2%}, "
          f"Eloignee {pred[g == 'Eloignee'].mean():.2%})")
    by_region = pd.Series(pred).groupby(cand["region_administrative"].to_numpy()).mean()
    print(by_region.round(3).to_string())

    # Robustness: expected scores if the hidden reference follows a different hypothesis.
    r = cand["cote_r_equivalent"].to_numpy()
    h = cand["heures_travail_semaine"].to_numpy()
    log_inc = np.log(cand["revenu_familial_estime"].to_numpy())
    a = weights["h"]
    hypotheses = {
        "H1 R + a*h": r + a * h, "H1 hours x1.5": r + 1.5 * a * h, "H1 hours x0.5": r + 0.5 * a * h,
        "H2 + income": r + a * h + 1.29 * log_inc, "H3 R only": r,
        "H4 need-based": r + a * h - 1.29 * log_inc,
    }
    regions = cand["region_administrative"].to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for name, merit in hypotheses.items():
        gaps, gaps5, agree = [], [], []
        for _ in range(30):
            ref = top_k(merit + rng.normal(0, REF_NOISE_SD, len(merit)), 0.40)
            gaps.append(abs(eo_gap(pred, ref, g)))
            tprs = [pred[(regions == reg) & (ref == 1)].mean() for reg in np.unique(regions)]
            gaps5.append(max(tprs) - min(tprs))
            agree.append((pred == ref).mean())
        gap, acc = np.mean(gaps), np.mean(agree)
        rows.append({"reference": name, "eo_gap": gap, "eo_gap_5_regions": np.mean(gaps5), "agreement": acc,
                     "equity_pts~": 20 * max(0.0, 1 - gap / BASELINE_EO_GAP),
                     "utility_pts~": 15 * (acc - RANDOM_AGREEMENT) / (1 - RANDOM_AGREEMENT)})
    print("\nRobustness (simulated references, noise sd "
          f"{REF_NOISE_SD}; points assume a linear scoring rule):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def pareto(hist):
    """Every method is evaluated on held-out historical folds, two ways:
    (a) against the committee labels (what the old dashboard sees);
    (b) against a simulated merit reference of the same family as ours (R + hours, hours
        weight drawn in +-50 %): it shows the mechanism, it does not prove which reference
        the jury uses."""
    records = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train, test = train_test_split(hist, test_size=0.3, random_state=seed,
                                       stratify=hist["decision_octroi"])
        g_tr, g_te = group(train), group(test)
        y_te = test["decision_octroi"].to_numpy()
        committee = fit_committee(train)
        weights = merit_weights(committee)
        ref = simulated_reference(test, weights, rng)

        def add(method, param, pred):
            records.append({
                "method": method, "param": param, "seed": seed, "rate": pred.mean(),
                "acc_committee": (pred == y_te).mean(), "dp_gap": selection_gap(pred, g_te),
                "eo_gap_committee": eo_gap(pred, y_te, g_te),
                "eo_gap_merit": eo_gap(pred, ref, g_te), "agree_merit": (pred == ref).mean(),
            })

        merit, bias = merit_score(test, weights), bias_score(test, committee)
        for lam in LAMBDAS:
            add("ours: merit + lambda*bias", lam, top_k(merit + lam * bias, 0.40))

        base_pred, _ = baseline_model(train, test)
        add("baseline RF", np.nan, base_pred)
        drop = ("region_administrative_", "code_postal_3_")
        X, Xt = encode(train, test)
        keep = [c for c in X.columns if not c.startswith(drop)]
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
        rf.fit(X[keep], train["decision_octroi"])
        add("RF without region/postal", np.nan, top_k(rf.predict_proba(Xt[keep])[:, 1], 0.40))

        for constraint in ("true_positive_rate_parity", "demographic_parity"):
            to = ThresholdOptimizer(
                estimator=RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42),
                constraints=constraint, predict_method="predict_proba",
                objective="balanced_accuracy_score" if constraint.startswith("true") else "accuracy_score",
            )
            to.fit(X, train["decision_octroi"], sensitive_features=g_tr)
            add(f"ThresholdOptimizer {constraint}", np.nan,
                to.predict(Xt, sensitive_features=g_te, random_state=seed))

        for eps in EG_EPS:
            eg = ExponentiatedGradient(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                constraints=TruePositiveRateParity(difference_bound=eps),
                sample_weight_name="logisticregression__sample_weight",
            )
            eg.fit(X.astype(float), train["decision_octroi"], sensitive_features=g_tr)
            proba = eg._pmf_predict(Xt.astype(float))[:, 1]
            add("ExpGrad EO (committee labels), top 40%", eps, top_k(proba, 0.40))

    res = pd.DataFrame(records)
    summary = res.groupby(["method", "param"], dropna=False).agg(["mean", "std"]).drop(columns="seed")
    res.to_csv(ROOT / "pareto_results.csv", index=False)
    return res, summary


def plot_pareto(res):
    agg = res.groupby(["method", "param"], dropna=False).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    styles = {
        "ours: merit + lambda*bias": dict(marker="o", color="#1b7f5a", ls="-"),
        "ExpGrad EO (committee labels), top 40%": dict(marker="s", color="#7a5195", ls="--"),
        "baseline RF": dict(marker="X", color="#c0392b", ls="", ms=11),
        "RF without region/postal": dict(marker="P", color="#e67e22", ls="", ms=10),
        "ThresholdOptimizer true_positive_rate_parity": dict(marker="^", color="#2c7fb8", ls="", ms=9),
        "ThresholdOptimizer demographic_parity": dict(marker="v", color="#41b6c4", ls="", ms=9),
    }
    panels = [
        ("dp_gap", "acc_committee",
         "(a) Old dashboard: fidelity to the committee", "Selection-rate gap Centre - Eloignee",
         "Accuracy vs committee labels"),
        ("eo_gap_merit", "agree_merit",
         "(b) Simulated merit reference (hours weight ±50 %)", "EO gap vs merit reference",
         "Agreement with merit reference"),
    ]
    for ax, (x, y, title, xl, yl) in zip(axes, panels):
        for method, st in styles.items():
            sub = agg[agg["method"] == method].sort_values("param")
            label = method
            if method.startswith("ThresholdOptimizer"):
                label += f" (rate {sub['rate'].iloc[0]:.0%})"
            ax.plot(sub[x].abs(), sub[y], label=label, **st)
        ours = agg[agg["method"] == "ours: merit + lambda*bias"]
        for _, row in ours.iloc[[0, -1]].iterrows():
            ax.annotate(f"λ={row['param']:.0f}", (abs(row[x]), row[y]),
                        textcoords="offset points", xytext=(6, -12), fontsize=9)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Pareto front: share of committee bias kept (λ=1 committee, λ=0 merit only); "
                 f"mean of {len(SEEDS)} held-out splits", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "pareto_front.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.40)
    ap.add_argument("--hours-mult", type=float, default=1.0,
                    help="multiplier on the committee's hours weight (set from HxBuddy probes)")
    ap.add_argument("--log-income-weight", type=float, default=0.0,
                    help="merit weight on ln(income): -1.29 need-based, +1.29 income as merit")
    ap.add_argument("--skip-pareto", action="store_true")
    args = ap.parse_args()

    hist, cand = load_data()
    pred, committee, weights = final_predictions(hist, cand, args.rate, args.hours_mult, args.log_income_weight)
    print(committee.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(4).to_string())
    print(f"\nMerit score = R + {weights['h']:.4f} x hours {weights['log_inc']:+.2f} x ln(income)")
    pd.DataFrame({"id_candidat": cand["id_candidat"], "decision_octroi": pred}).to_csv(
        ROOT / "predictions.csv", index=False)
    print(f"predictions.csv written: {len(pred)} rows")
    report_final(cand, pred, weights)

    if not args.skip_pareto:
        res, summary = pareto(hist)
        cols = ["rate", "acc_committee", "dp_gap", "eo_gap_committee", "eo_gap_merit", "agree_merit"]
        pd.set_option("display.width", 250)
        print("\nPareto sweep (mean over seeds):")
        print(summary.xs("mean", axis=1, level=1)[cols].round(3).to_string())
        print("\nStd over seeds (max per column):",
              summary.xs("std", axis=1, level=1)[cols].max().round(3).to_dict())
        plot_pareto(res)
        print("pareto_front.png written")


if __name__ == "__main__":
    main()
