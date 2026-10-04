"""Expected HxBuddy values for every probe under every candidate reference.

Each hypothesis defines a latent merit; the simulated reference grants the top `prev`
share of merit + Gaussian noise. Comparing the observed HxBuddy accuracies with these
rows tells which hypothesis (and roughly which noise level) generated the reference.

    python probes/fingerprints.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fit_committee, group, load_data, merit_weights, top_k  # noqa: E402

FILES = Path(__file__).resolve().parent / "files"
N_DRAWS = 20


def main():
    hist, cand = load_data()
    a_hat = merit_weights(fit_committee(hist))["h"]
    r = cand["cote_r_equivalent"].to_numpy()
    h = cand["heures_travail_semaine"].to_numpy()
    log_inc = np.log(cand["revenu_familial_estime"].to_numpy())
    fg = cand["premiere_generation_universitaire"].to_numpy()
    remote = group(cand) == "Eloignee"

    hypotheses = {
        "H1 R+a*h": r + a_hat * h,
        "H1' R+1.5a*h": r + 1.5 * a_hat * h,
        "H1'' R+0.5a*h": r + 0.5 * a_hat * h,
        "H2 +income": r + a_hat * h + 1.29 * log_inc,
        "H3 R only": r,
        "H4 need-based": r + a_hat * h - 1.29 * log_inc,
        "H5 +first-gen": r + a_hat * h + 1.0 * fg,
    }
    probes = {p.stem: pd.read_csv(p)["decision_octroi"].to_numpy() for p in sorted(FILES.glob("*.csv"))}
    rng = np.random.default_rng(0)
    rows = []
    for hyp, merit in hypotheses.items():
        for sd in (0.5, 1.3):
            acc = {k: [] for k in probes}
            rate_remote, rate_centre = [], []
            for _ in range(N_DRAWS):
                ref = top_k(merit + rng.normal(0, sd, len(merit)), 0.40)
                rate_remote.append(ref[remote].mean())
                rate_centre.append(ref[~remote].mean())
                for k, pred in probes.items():
                    acc[k].append((pred == ref).mean())
            row = {"hypothesis": hyp, "noise_sd": sd,
                   "ref_rate_centre": np.mean(rate_centre), "ref_rate_remote": np.mean(rate_remote)}
            row.update({k: np.mean(v) for k, v in acc.items()})
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["hypothesis", "noise_sd"]).round(4)
    pd.set_option("display.width", 250)
    print("Expected accuracy per probe (reference prevalence assumed 40 %):")
    print(table.T.to_string())


if __name__ == "__main__":
    main()
