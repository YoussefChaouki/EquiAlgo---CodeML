"""Shared data loading, group definitions, scoring helpers and fairness metrics.

Used by model_corrige.py, audit_rapport.ipynb and probes/make_probes.py so that every
deliverable computes groups, budgets and gaps the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

REMOTE_REGIONS = ["Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]
CATEGORICAL = ["programme_etudes", "region_administrative", "code_postal_3"]
BUDGET_MIN, BUDGET_MAX = 0.36, 0.44

# Committee model: every observed variable enters, so that each one's weight is
# estimated. Which weights count as merit and which as bias is a normative choice
# (see merit_weights), not something the committee data can identify. Income enters in log form
# (binned fits and AIC both favour it over a linear term).
COMMITTEE_FORMULA = (
    "decision_octroi ~ r + h + np.log(revenu) + C(region, Treatment('Montreal'))"
    " + C(programme) + fg + dist_in_region"
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = pd.read_csv(DATA / "donnees_demandes.csv")
    cand = pd.read_csv(DATA / "candidats_evaluation.csv")
    return hist, cand


def group(df: pd.DataFrame) -> np.ndarray:
    """Two-block sensitive attribute used throughout the audit: Centre vs Eloignee."""
    return np.where(df["region_administrative"].isin(REMOTE_REGIONS), "Eloignee", "Centre")


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Short column names for formulas. Distance is centred within region so that its
    coefficient measures a within-region effect instead of re-absorbing the region gap."""
    out = pd.DataFrame({
        "r": df["cote_r_equivalent"],
        "h": df["heures_travail_semaine"],
        "revenu": df["revenu_familial_estime"],
        "region": df["region_administrative"],
        "programme": df["programme_etudes"],
        "fg": df["premiere_generation_universitaire"],
        "dist_in_region": df["distance_domicile_campus_km"]
        - df.groupby("region_administrative")["distance_domicile_campus_km"].transform("mean"),
    })
    if "decision_octroi" in df:
        out["decision_octroi"] = df["decision_octroi"]
    return out


def fit_committee(hist: pd.DataFrame) -> BinaryResultsWrapper:
    return smf.logit(COMMITTEE_FORMULA, model_frame(hist)).fit(disp=0)


def merit_weights(committee: BinaryResultsWrapper) -> dict[str, float]:
    """Weights of the criteria treated as legitimate, in R-score points: R score and hours
    worked, at the committee's own relative weight. Region and income are treated as bias
    (log-income weight 0); probes/DECISION_RULE.md says when to revise this."""
    return {"r": 1.0, "h": committee.params["h"] / committee.params["r"], "log_inc": 0.0}


def merit_score(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return (
        weights["r"] * df["cote_r_equivalent"].to_numpy()
        + weights["h"] * df["heures_travail_semaine"].to_numpy()
        + weights.get("log_inc", 0.0) * np.log(df["revenu_familial_estime"].to_numpy())
    )


def bias_score(df: pd.DataFrame, committee: BinaryResultsWrapper) -> np.ndarray:
    """Committee's non-merit index (region + income), in R-score points."""
    p = committee.params
    mf = model_frame(df)
    region_coef = {
        name.split("[T.")[1].rstrip("]"): v for name, v in p.items() if name.startswith("C(region")
    }
    region_coef["Montreal"] = 0.0  # reference level
    unknown = set(mf["region"]) - set(region_coef)
    if unknown:
        raise ValueError(f"unknown regions: {unknown}")
    region_term = mf["region"].map(region_coef).to_numpy()
    income_term = p["np.log(revenu)"] * np.log(mf["revenu"].to_numpy())
    return (region_term + income_term) / p["r"]


def top_k(score: np.ndarray, rate: float) -> np.ndarray:
    """Grant exactly round(rate * n) awards to the highest scores (ties broken by order)."""
    n_grant = int(round(rate * len(score)))
    order = np.argsort(-np.asarray(score), kind="stable")
    pred = np.zeros(len(score), dtype=int)
    pred[order[:n_grant]] = 1
    return pred


def check_budget(pred: np.ndarray) -> float:
    rate = float(np.mean(pred))
    if not BUDGET_MIN <= rate <= BUDGET_MAX:
        raise ValueError(f"grant rate {rate:.3%} outside budget [{BUDGET_MIN:.0%}, {BUDGET_MAX:.0%}]")
    return rate


def selection_gap(pred: np.ndarray, groups: np.ndarray) -> float:
    pred, groups = np.asarray(pred), np.asarray(groups)
    return pred[groups == "Centre"].mean() - pred[groups == "Eloignee"].mean()


def eo_gap(pred: np.ndarray, ref: np.ndarray, groups: np.ndarray) -> float:
    """Equal-opportunity gap: TPR(Centre) - TPR(Eloignee) against reference labels `ref`."""
    pred, ref, groups = np.asarray(pred), np.asarray(ref), np.asarray(groups)
    def tpr(g: str) -> float:
        return pred[(groups == g) & (ref == 1)].mean()

    return tpr("Centre") - tpr("Eloignee")


def encode(train_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encoding aligned on the training columns (as in the baseline notebook)."""
    drop = ["id_candidat", "decision_octroi", "groupe"]
    X = pd.get_dummies(train_df.drop(columns=drop, errors="ignore"), columns=CATEGORICAL)
    Xt = pd.get_dummies(target_df.drop(columns=drop, errors="ignore"), columns=CATEGORICAL)
    return X, Xt.reindex(columns=X.columns, fill_value=0)


def baseline_model(train_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The production model from baseline_model.ipynb, unchanged."""
    X, Xt = encode(train_df, target_df)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
    rf.fit(X, train_df["decision_octroi"])
    return rf.predict(Xt), rf.predict_proba(Xt)[:, 1]
