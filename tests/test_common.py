import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
from common import check_budget, eo_gap, load_data, selection_gap, top_k  # noqa: E402
from decode_hxbuddy import decode  # noqa: E402


def test_top_k_grants_exact_share():
    pred = top_k(np.arange(4000, dtype=float), 0.40)
    assert pred.sum() == 1600
    assert pred[-1600:].all() and not pred[:-1600].any()


def test_check_budget_rejects_out_of_range():
    assert check_budget(np.r_[np.ones(40), np.zeros(60)]) == 0.40
    with pytest.raises(ValueError):
        check_budget(np.r_[np.ones(45), np.zeros(55)])


def test_gaps_on_hand_built_case():
    groups = np.array(["Centre"] * 4 + ["Eloignee"] * 4)
    ref = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    assert selection_gap(pred, groups) == pytest.approx(0.75 - 0.25)
    assert eo_gap(pred, ref, groups) == pytest.approx(1.0 - 0.5)


def test_decoder_recovers_prevalence():
    rng = np.random.default_rng(0)
    pred = np.zeros(4000, dtype=int)
    pred[:1600] = 1
    ref = pred.copy()
    flip = rng.choice(4000, 300, replace=False)
    ref[flip] = 1 - ref[flip]
    tp, tn = int(((pred == 1) & (ref == 1)).sum()), int(((pred == 0) & (ref == 0)).sum())
    fp, fn = 1600 - tp, int(ref.sum()) - tp
    f1 = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
    sols = decode(round((tp + tn) / 4000, 4), round(f1, 4), 1600)
    assert any(s[0] == ref.sum() for s in sols)


def test_predictions_file_is_valid():
    path = Path(__file__).resolve().parents[1] / "predictions.csv"
    import pandas as pd

    sub = pd.read_csv(path)
    _, cand = load_data()
    assert list(sub.columns) == ["id_candidat", "decision_octroi"]
    assert len(sub) == 4000 and set(sub["id_candidat"]) == set(cand["id_candidat"])
    assert set(sub["decision_octroi"].unique()) <= {0, 1}
    check_budget(sub["decision_octroi"].to_numpy())


def test_decoder_handles_half_unit_rounding():
    # TP + TN odd -> displayed accuracy sits exactly on the rounding boundary
    tp, fp, fn, tn = 1166, 434, 621, 1779
    acc = (tp + tn) / 4000
    f1 = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
    sols = decode(round(acc, 4), round(f1, 4), tp + fp)
    assert any(s[0] == tp + fn for s in sols)


def test_group_rates_inverts_p00_probes():
    from decode_hxbuddy import N_REMOTE, group_rates

    _, cand = load_data()
    remote = cand["region_administrative"].isin(
        ["Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]).to_numpy()
    assert remote.sum() == N_REMOTE
    rng = np.random.default_rng(1)
    ref = (rng.random(4000) < np.where(remote, 0.35, 0.45)).astype(int)
    acc_a = ref.mean()
    acc_b = (remote.astype(int) == ref).mean()
    out = group_rates(acc_a, acc_b)
    assert out["rate_remote"] == pytest.approx(ref[remote].mean())
    assert out["rate_centre"] == pytest.approx(ref[~remote].mean())


def test_monitor_flags_production_model_not_ours():
    from common import baseline_model, fit_committee, merit_score, merit_weights
    from monitoring import cycle_report

    hist, cand = load_data()
    w = merit_weights(fit_committee(hist))
    ours = cycle_report(cand, top_k(merit_score(cand, w), 0.40), hist, w).set_index("indicator")
    base = cycle_report(cand, baseline_model(hist, cand)[0], hist, w).set_index("indicator")
    gap = "grant-rate gap Centre - Eloignee"
    assert not ours["alert"].any()
    assert base.loc[gap, "alert"] and base.loc["merit bar of granted, Centre - Eloignee", "alert"]
