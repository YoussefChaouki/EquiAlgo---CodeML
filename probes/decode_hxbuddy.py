"""Recover the confusion matrix behind an HxBuddy (accuracy, macro-F1) pair.

For a submission with P predicted positives out of N, accuracy fixes TP + TN and the
macro-F1 then pins down the number of reference positives A. One probe therefore
reveals the prevalence of the hidden reference, which sets the best grant rate.

    python probes/decode_hxbuddy.py --acc 0.9123 --f1 0.9071 --pos 1600
    python probes/decode_hxbuddy.py --log probes/probe_log.csv     # decode every filled row
    python probes/decode_hxbuddy.py --p00a 0.4000 --p00b 0.5210    # reference rate per group
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N = 4000


def decode(acc, f1, pos, n=N, acc_tol=None, f1_tol=None):
    """Return every (A, TP, FP, FN, TN) consistent with the rounded displayed values."""
    # +1e-9: with N=4000 an odd TP+TN lands exactly on the rounding half-unit
    acc_tol = acc_tol if acc_tol is not None else 0.5 * 10 ** -_decimals(acc) + 1e-9
    f1_tol = f1_tol if f1_tol is not None else 0.5 * 10 ** -_decimals(f1) + 1e-9
    out = []
    for a in range(1, n):
        # accuracy fixes TP + TN = TP + (n - pos - a + TP), so TP lies in a narrow window
        centre, half = (acc * n - n + pos + a) / 2, acc_tol * n / 2
        lo, hi = max(0, pos + a - n, int(np.ceil(centre - half))), min(pos, a, int(np.floor(centre + half)))
        for tp in range(lo, hi + 1):
            fp, fn = pos - tp, a - tp
            tn = n - pos - fn
            if abs((tp + tn) / n - acc) > acc_tol:
                continue
            f1_pos = 2 * tp / (2 * tp + fp + fn)
            f1_neg = 2 * tn / (2 * tn + fp + fn)
            if abs((f1_pos + f1_neg) / 2 - f1) <= f1_tol:
                out.append((a, tp, fp, fn, tn))
    return out


def _decimals(x):
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s else 0


def summarize(acc, f1, pos):
    sols = decode(acc, f1, pos)
    if not sols:
        return "no consistent confusion matrix (check the displayed values / tolerances)"
    a = np.array([s[0] for s in sols])
    return (
        f"reference positives A in [{a.min()}, {a.max()}] -> prevalence "
        f"{a.min() / N:.2%}..{a.max() / N:.2%}; solutions (A,TP,FP,FN,TN): {sols[:6]}"
    )


N_REMOTE = 1628  # candidates in Bas-Saint-Laurent, Cote-Nord, Gaspesie (P00b grants)


def group_rates(acc_all_ones, acc_remote_only, n=N, n_remote=N_REMOTE):
    """P00a: acc = A/n. P00b: acc = (A_remote + n_centre - A_centre)/n."""
    n_centre = n - n_remote
    # counts are integers: round them so displayed-accuracy rounding does not leak into rates
    a_total = round(acc_all_ones * n)
    a_remote = round((acc_remote_only * n - n_centre + a_total) / 2)
    a_centre = a_total - a_remote
    return {"prevalence": a_total / n, "rate_centre": a_centre / n_centre,
            "rate_remote": a_remote / n_remote, "gap_centre_minus_remote": a_centre / n_centre - a_remote / n_remote}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p00a", type=float, help="HxBuddy accuracy of P00a_all_ones")
    ap.add_argument("--p00b", type=float, help="HxBuddy accuracy of P00b_ones_remote_only")
    ap.add_argument("--acc", type=float)
    ap.add_argument("--f1", type=float)
    ap.add_argument("--pos", type=int, default=1600)
    ap.add_argument("--log", type=Path)
    args = ap.parse_args()
    if args.p00a is not None and args.p00b is not None:
        print({k: round(v, 4) for k, v in group_rates(args.p00a, args.p00b).items()})
    elif args.log:
        log = pd.read_csv(args.log).dropna(subset=["hx_accuracy", "hx_f1_macro"])
        for row in log.itertuples():
            print(f"{row.probe:20s} acc={row.hx_accuracy:.4f} f1={row.hx_f1_macro:.4f}  "
                  + summarize(row.hx_accuracy, row.hx_f1_macro, row.n_granted))
    else:
        print(summarize(args.acc, args.f1, args.pos))


if __name__ == "__main__":
    main()
