"""Production monitoring for the scholarship model: one report per award cycle.

Label-free indicators (computable as soon as decisions are made) plus feature drift
against the historical data. Thresholds are set from sampling error: with 1,628 remote and
2,372 centre applicants, the Centre-Eloignee rate difference has a standard error of about
0.016, so the 0.05 group-gap alert sits at roughly three standard errors. The 0.08 range
over the 5 regions (491 to 1,551 applicants each) fires in about 3 % of null simulations.

    python monitoring.py predictions.csv
"""
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import BUDGET_MAX, BUDGET_MIN, fit_committee, group, load_data, merit_score, merit_weights


@dataclass(frozen=True)
class Thresholds:
    group_gap: float = 0.05        # |grant rate Centre - Eloignee|
    region_gap: float = 0.08       # max - min grant rate over the 5 regions
    bar_gap: float = 0.5           # effective merit bar (R points), Centre - Eloignee
    psi: float = 0.2               # population stability index of any input feature


DRIFT_FEATURES = ["cote_r_equivalent", "heures_travail_semaine", "revenu_familial_estime",
                  "distance_domicile_campus_km"]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index on reference deciles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def merit_bar(merit: np.ndarray, granted: np.ndarray) -> float:
    """Effective bar: 5th percentile of the merit score among granted applicants."""
    return float(np.percentile(merit[granted], 5))


def cycle_report(applicants: pd.DataFrame, decisions: np.ndarray, reference: pd.DataFrame,
                 weights: dict[str, float], thr: Thresholds = Thresholds()) -> pd.DataFrame:
    """One row per indicator: value, threshold, alert flag. `weights` define the merit
    score (common.merit_weights) used to check that every group faces the same bar."""
    decisions = np.asarray(decisions)
    g = group(applicants)
    by_region = pd.Series(decisions).groupby(applicants["region_administrative"].to_numpy()).mean()
    granted = decisions == 1
    merit = merit_score(applicants, weights)
    rows = [
        ("grant rate (budget)", decisions.mean(), f"[{BUDGET_MIN}, {BUDGET_MAX}]",
         not BUDGET_MIN <= decisions.mean() <= BUDGET_MAX),
        ("grant-rate gap Centre - Eloignee",
         decisions[g == "Centre"].mean() - decisions[g == "Eloignee"].mean(), thr.group_gap, None),
        ("grant-rate range over 5 regions", by_region.max() - by_region.min(), thr.region_gap, None),
        ("merit bar of granted, Centre - Eloignee",
         merit_bar(merit, granted & (g == "Centre")) - merit_bar(merit, granted & (g == "Eloignee")),
         thr.bar_gap, None),
    ]
    rows += [(f"PSI {f}", psi(reference[f].to_numpy(), applicants[f].to_numpy()), thr.psi, None)
             for f in DRIFT_FEATURES]
    out = pd.DataFrame(rows, columns=["indicator", "value", "threshold", "alert"])
    numeric = out["alert"].isna()
    out.loc[numeric, "alert"] = out.loc[numeric, "value"].abs() > out.loc[numeric, "threshold"].astype(float)
    return out


if __name__ == "__main__":
    hist, cand = load_data()
    pred = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "predictions.csv")
    missing = set(cand["id_candidat"]) - set(pred["id_candidat"])
    if missing or len(pred) != len(cand):
        raise ValueError(f"predictions must cover every applicant: {len(missing)} missing, {len(pred)} rows")
    cand = cand.merge(pred, on="id_candidat", validate="one_to_one")
    weights = merit_weights(fit_committee(hist))
    print(cycle_report(cand, cand["decision_octroi"].to_numpy(), hist, weights).round(4).to_string(index=False))
