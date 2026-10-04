"""Write one submission file per hypothesis about the hidden reference ("merite reel").

P00a/P00b are aggregate calibration probes, never a final submission: accuracy of
"all ones" (out of budget on purpose) equals the reference prevalence, and "ones for
remote regions only" then gives the reference rate in each group.
Every other probe grants exactly 40 % (1,600 awards) so that HxBuddy accuracies are
directly comparable: with the same number of positives, a higher accuracy means a
ranking closer to the reference. Probes are ordered by information value.

    python probes/make_probes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    baseline_model, check_budget, fit_committee, group, load_data, merit_weights, top_k,
)

OUT = Path(__file__).resolve().parent / "files"
RATE = 0.40


def main():
    hist, cand = load_data()
    a_hat = merit_weights(fit_committee(hist))["h"]
    r = cand["cote_r_equivalent"].to_numpy()
    h = cand["heures_travail_semaine"].to_numpy()
    log_inc = np.log(cand["revenu_familial_estime"].to_numpy())
    fg = cand["premiere_generation_universitaire"].to_numpy()

    remote = (group(cand) == "Eloignee").astype(int)

    # (name, score or fixed decisions, hypothesis tested)
    probes = [
        ("P00a_all_ones", np.ones(len(cand), dtype=int), "acc = reference prevalence"),
        ("P00b_ones_remote_only", remote, "acc = (A_remote + n_centre - A_centre) / N"),
        ("P01_r_hours_hat", r + a_hat * h, f"merit = R + {a_hat:.3f} x hours (committee weight)"),
        ("P03_r_only", r, "merit = R score alone"),
        ("P04_r_hours_x1.5", r + 1.5 * a_hat * h, "hours weight 50 % higher"),
        ("P05_r_hours_x0.5", r + 0.5 * a_hat * h, "hours weight 50 % lower"),
        ("P06_plus_income", r + a_hat * h + 1.29 * log_inc, "income is merit (committee minus region)"),
        ("P07_need_based", r + a_hat * h - 1.29 * log_inc, "need-based: lower income favoured"),
        ("P08_first_gen", r + a_hat * h + 1.0 * fg, "first-generation bonus of 1 R point"),
        ("P09_r_hours_x2", r + 2.0 * a_hat * h, "hours weight doubled"),
        ("P10_need_no_hours", r - 1.29 * log_inc, "need-based, hours not merit"),
        ("P02_baseline_rf", None, "production model; anchors the utility scale"),
    ]
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, score, hypothesis in probes:
        if score is None:
            pred, _ = baseline_model(hist, cand)
        elif name.startswith("P00"):
            pred = score
        else:
            pred = top_k(score, RATE)
        rate = float(pred.mean()) if name.startswith("P00") else check_budget(pred)
        pd.DataFrame({"id_candidat": cand["id_candidat"], "decision_octroi": pred}).to_csv(
            OUT / f"{name}.csv", index=False
        )
        rows.append({"probe": name, "n_granted": int(pred.sum()), "rate": round(rate, 4), "hypothesis": hypothesis})
    log = pd.DataFrame(rows)
    log_path = Path(__file__).resolve().parent / "probe_log.csv"
    if log_path.exists():  # keep HxBuddy results already entered by hand
        entered = pd.read_csv(log_path)[["probe", "hx_accuracy", "hx_f1_macro"]]
        log = log.merge(entered, on="probe", how="left")
    else:
        log["hx_accuracy"] = np.nan
        log["hx_f1_macro"] = np.nan
    log.to_csv(log_path, index=False)
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()
