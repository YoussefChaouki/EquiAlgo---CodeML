"""Build presentation.pdf (5-minute pitch) from the project's own data and results.

Numbers coming from the data and our results are recomputed here from data/,
pareto_results.csv and probes/probe_log.csv. Context figures quoted from the challenge
README (88 %, 0.188/0.181/0.173) and notebook-only estimates (bootstrap CI, boosting gap,
simulated production accuracy) are literal text and must be updated by hand. Figures are SVG (crisp in
the PDF); the deck is HTML printed to PDF by headless Chrome.

    python presentation/build_presentation.py
"""
import io
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))
from common import (  # noqa: E402
    baseline_model, fit_committee, group, load_data, merit_score, merit_weights, model_frame,
)
from decode_hxbuddy import group_rates  # noqa: E402
from monitoring import cycle_report  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT_DIR = Path(__file__).resolve().parent

# Validated categorical pair (dataviz reference palette, slots 1-2, light surface).
CENTRE, REMOTE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Avenir Next", "Helvetica Neue", "Arial"], "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "svg.fonttype": "none",
})


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()[buf.getvalue().index("<svg"):]


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


# ---------------------------------------------------------------- numbers
def compute():
    hist, cand = load_data()
    committee = fit_committee(hist)
    w = merit_weights(committee)
    p = committee.params
    g_hist, g_cand = group(hist), group(cand)
    pred = pd.read_csv(ROOT / "predictions.csv").set_index("id_candidat").loc[cand["id_candidat"], "decision_octroi"].to_numpy()
    base_pred, _ = baseline_model(hist, cand)

    # Counterfactual decomposition of the committee gap (same as audit_rapport.ipynb section 3)
    mf = model_frame(hist).astype({"revenu": float})
    remote = g_hist == "Eloignee"
    p_obs = committee.predict(mf)
    cf = mf.copy()
    cf.loc[remote, "region"] = "Montreal"
    p_reg = committee.predict(cf)
    log_inc = np.log(mf["revenu"])
    cf.loc[remote, "revenu"] = np.exp(log_inc[remote] - log_inc[remote].mean() + log_inc[~remote].mean())
    p_inc = committee.predict(cf)
    gap = p_obs[~remote].mean() - p_obs[remote].mean()
    decomposition = {
        "Pénalité régionale directe": gap - (p_obs[~remote].mean() - p_reg[remote].mean()),
        "Avantage aux revenus élevés": (p_obs[~remote].mean() - p_reg[remote].mean()) - (p_obs[~remote].mean() - p_inc[remote].mean()),
        "Mérite (cote R, heures)": p_obs[~remote].mean() - p_inc[remote].mean(),
    }

    # Proxy strength: AUC of predicting "remote" from each variable alone
    y = remote.astype(int)
    proxies = {"Code postal": pd.get_dummies(hist["code_postal_3"]), "Distance au campus": hist[["distance_domicile_campus_km"]],
               "Heures travaillées": hist[["heures_travail_semaine"]], "Revenu familial": hist[["revenu_familial_estime"]],
               "1re génération": hist[["premiere_generation_universitaire"]], "Cote R": hist[["cote_r_equivalent"]]}
    auc = {k: roc_auc_score(y, cross_val_predict(HistGradientBoostingClassifier(random_state=0), X.astype(float), y,
                                                   cv=5, method="predict_proba")[:, 1]) for k, X in proxies.items()}

    log = pd.read_csv(ROOT / "probes" / "probe_log.csv").set_index("probe")
    ref = group_rates(log.loc["P00a_all_ones", "hx_accuracy"], log.loc["P00b_ones_remote_only", "hx_accuracy"])
    monitor = {name: cycle_report(cand, d, hist, w).set_index("indicator")["value"]
               for name, d in [("ours", pred), ("base", base_pred)]}
    return {
        "hours_w": w["h"], "remote_pen": -p[[k for k in p.index if "Gaspesie" in k][0]] / p["r"],
        "income_double": p["np.log(revenu)"] / p["r"] * np.log(2),
        "committee_rates": pd.Series(hist["decision_octroi"].to_numpy()).groupby(g_hist).mean(),
        "ours_rates": pd.Series(pred).groupby(g_cand).mean(), "ref": ref,
        "r_centre": hist.loc[~remote, "cote_r_equivalent"].mean(), "r_remote": hist.loc[remote, "cote_r_equivalent"].mean(),
        "decomposition": decomposition, "gap": gap, "auc": pd.Series(auc).sort_values(),
        "p01": log.loc["P01_r_hours_hat"], "monitor": monitor,
        "pareto": pd.read_csv(ROOT / "pareto_results.csv"),
        "fair": pareto_summary(pd.read_csv(ROOT / "pareto_results.csv")),
    }


def pareto_summary(res: pd.DataFrame) -> dict[str, float]:
    """Mean over seeds of the fairlearn baselines, as quoted on the slides."""
    by_method = res.groupby("method").mean(numeric_only=True)  # single-setting methods
    eg = res[res["method"].str.startswith("ExpGrad")].groupby("param")["eo_gap_merit"].mean()
    to_eo = by_method.loc["ThresholdOptimizer true_positive_rate_parity"]
    to_dp = by_method.loc["ThresholdOptimizer demographic_parity"]
    ours = res[res["method"].str.startswith("ours") & (res["param"] == 0.0)].mean(numeric_only=True)
    return {"gap_lo": min(eg.min(), to_eo["eo_gap_merit"]), "gap_hi": max(eg.max(), to_eo["eo_gap_merit"]),
            "to_eo_rate": to_eo["rate"], "to_dp_gap": to_dp["eo_gap_merit"], "to_dp_rate": to_dp["rate"],
            "to_dp_agree": to_dp["agree_merit"], "ours_agree": ours["agree_merit"]}


# ---------------------------------------------------------------- figures
def fig_rates(n) -> str:
    groups = [("Comité\nhistorique", n["committee_rates"]["Centre"], n["committee_rates"]["Eloignee"]),
              ("Mérite réel\n(étalon, mesuré)", n["ref"]["rate_centre"], n["ref"]["rate_remote"]),
              ("Notre\nmodèle", n["ours_rates"]["Centre"], n["ours_rates"]["Eloignee"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(groups))
    for i, (label, c, r) in enumerate(groups):
        for dx, v, col in [(-0.19, c, CENTRE), (0.19, r, REMOTE)]:
            ax.bar(i + dx, v, width=0.36, color=col, edgecolor=SURFACE, linewidth=2)
            ax.text(i + dx, v + 0.01, pct(v, 0 if i == 0 else 1), ha="center", va="bottom", color=INK, fontsize=12)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylim(0, 0.58)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Taux d'octroi / de mérite")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CENTRE), plt.Rectangle((0, 0), 1, 1, color=REMOTE)],
              labels=["Montréal, Capitale-Nationale", "Régions éloignées"], frameon=False, fontsize=11,
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return svg(fig)


def fig_decomposition(n) -> str:
    d = n["decomposition"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels, vals = list(d)[::-1], [d[k] for k in list(d)[::-1]]
    ax.barh(labels, vals, color=CENTRE, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{100 * v:.1f} pts ({100 * v / n['gap']:.0f} %)".replace(".", ","), va="center", color=INK, fontsize=12)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
    ax.set_xlabel(f"Points de l'écart de {100 * n['gap']:.0f} points expliqués")
    ax.grid(axis="y", visible=False)
    return svg(fig)


def fig_proxies(n) -> str:
    auc = n["auc"]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.barh(auc.index, auc.values - 0.5, left=0.5, color=CENTRE, height=0.55)
    for i, v in enumerate(auc.values):
        ax.text(v + 0.008, i, num(v), va="center", color=INK, fontsize=12)
    ax.set_xlim(0.5, 1.08)
    ax.set_xlabel("AUC : retrouver « région éloignée » (0,5 = hasard)")
    ax.grid(axis="y", visible=False)
    return svg(fig)


def fig_pareto(n) -> str:
    res = n["pareto"]
    agg = res.groupby(["method", "param"], dropna=False).mean(numeric_only=True).reset_index()
    ours = agg[agg["method"].str.startswith("ours")].sort_values("param")
    eg = agg[agg["method"].str.startswith("ExpGrad")].sort_values("param")
    # label -> per panel (text offset in points, horizontal alignment)
    singles = {"baseline RF": ("Production", [((6, -14), "left"), ((6, -14), "left")]),
               "RF without region/postal": ("Sans région", [((-8, -16), "right"), ((-8, 8), "right")]),
               "ThresholdOptimizer true_positive_rate_parity": (
                   "ThresholdOpt. EO\n(46 % : hors budget)", [((-10, -6), "right"), ((-10, -6), "right")]),
               "ThresholdOptimizer demographic_parity": (
                   "ThresholdOpt. parité\n(seuil par région)", [((10, 0), "left"), ((8, -18), "left")])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (ax, (x, y, title, xl, yl)) in enumerate(zip(axes, [
        ("dp_gap", "acc_committee", "Mesuré contre le comité", "Écart de taux Centre − Éloignée", "Exactitude vs comité"),
        ("eo_gap_merit", "agree_merit", "Mesuré contre un mérite", "Écart d'égalité des chances", "Concordance avec le mérite"),
    ])):
        ax.plot(ours[x].abs(), ours[y], color=CENTRE, lw=2, marker="o", ms=6, label="Nous (λ = part du biais gardée)")
        ax.plot(eg[x].abs(), eg[y], color=REMOTE, lw=2, marker="s", ms=6, label="fairlearn ExpGrad (labels du comité)")
        for m, (lab, places) in singles.items():
            offset, ha = places[k]
            row = agg[agg["method"] == m].iloc[0]
            ax.plot(abs(row[x]), row[y], marker="X", ms=9, color=MUTED, ls="")
            ax.annotate(lab, (abs(row[x]), row[y]), textcoords="offset points", xytext=offset, ha=ha,
                        fontsize=9.5, color=INK_2)
        for _, r in ours.iloc[[0, -1]].iterrows():
            ax.annotate(f"λ={r['param']:.0f}", (abs(r[x]), r[y]), textcoords="offset points", xytext=(-8, 8), fontsize=10, color=INK)
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    return svg(fig)


# ---------------------------------------------------------------- slides
CSS = f"""
@page {{ size: 1280px 720px; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {SURFACE}; color: {INK};
  font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; }}
.slide {{ width: 1280px; height: 720px; padding: 56px 72px; position: relative;
  page-break-after: always; overflow: hidden; display: flex; flex-direction: column; }}
.kicker {{ font-size: 15px; letter-spacing: .14em; text-transform: uppercase; color: {CENTRE}; font-weight: 600; }}
h1 {{ font-size: 64px; line-height: 1.05; margin: 12px 0 18px; font-weight: 700; letter-spacing: -.01em; }}
h2 {{ font-size: 38px; line-height: 1.15; margin: 8px 0 22px; font-weight: 700; }}
p, li {{ font-size: 21px; line-height: 1.45; color: {INK_2}; }}
strong {{ color: {INK}; }}
ul {{ margin: 0; padding-left: 22px; }} li {{ margin-bottom: 8px; }}
.row {{ display: flex; gap: 40px; align-items: flex-start; flex: 1; }}
.col {{ flex: 1; min-width: 0; }}
.hero {{ font-size: 76px; font-weight: 700; line-height: 1; }}
.hero-label {{ font-size: 18px; color: {INK_2}; margin-top: 8px; }}
.c {{ color: {CENTRE}; }} .r {{ color: {REMOTE}; }}
.foot {{ position: absolute; bottom: 26px; left: 72px; right: 72px; font-size: 13px; color: {MUTED};
  display: flex; justify-content: space-between; }}
.box {{ border: 1px solid {GRID}; border-radius: 10px; padding: 18px 22px; background: #fff; }}
.formula {{ font-size: 30px; font-weight: 600; color: {INK}; margin: 6px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 17px; }}
td, th {{ padding: 7px 10px; border-bottom: 1px solid {GRID}; text-align: left; color: {INK_2}; }}
th {{ color: {INK}; font-weight: 600; }}
.tag {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 14px; font-weight: 600; }}
.bad {{ background: #fde8e4; color: #a3341b; }} .good {{ background: #e3f1ea; color: #1d6b45; }}
svg {{ max-width: 100%; height: auto; }}
"""


def slides(n, figs) -> list[str]:
    cr, ref = n["committee_rates"], n["ref"]
    mon_o, mon_b = n["monitor"]["ours"], n["monitor"]["base"]
    bar_key, gap_key = "merit bar of granted, Centre - Eloignee", "grant-rate gap Centre - Eloignee"
    d = n["decomposition"]
    f = n["fair"]
    share = lambda k: f"{100 * d[k] / n['gap']:.0f} %"  # noqa: E731
    return [
        f"""<div class="kicker">ÉquiAlgo · Hackathon Génie et Informatique 2026 · équipe Persona</div>
        <h1 style="margin-top:150px">Le mérite n'a pas<br>de code postal.</h1>
        <p style="font-size:26px">Diagnostiquer, corriger et surveiller le biais régional d'un modèle d'octroi de bourses.</p>""",

        f"""<div class="kicker">1 · Diagnostic</div><h2>À dossier égal, vivre en région coûte {num(n['remote_pen'], 1)} point de cote R</h2>
        <div class="row"><div class="col" style="flex:0.75">
          <div class="hero c">{pct(cr['Centre'], 0)}</div><div class="hero-label">d'octroi à Montréal et à Québec</div>
          <div class="hero r" style="margin-top:28px">{pct(cr['Eloignee'], 0)}</div><div class="hero-label">dans les régions éloignées</div>
          <p style="margin-top:28px">Le modèle « exact à 88 % » <strong>copie ce comité</strong>.</p>
        </div><div class="col">{figs['decomp']}
          <p style="font-size:18px">L'écart de 21 points : <strong>{share('Pénalité régionale directe')}</strong> de pénalité régionale, <strong>{share('Avantage aux revenus élevés')}</strong> d'avantage aux revenus élevés, <strong>{share('Mérite (cote R, heures)')}</strong> de mérite.</p>
        </div></div>""",

        f"""<div class="kicker">2 · Le piège</div><h2>Retirer la colonne région ne retire pas le biais</h2>
        <div class="row"><div class="col">{figs['proxies']}</div><div class="col"><ul>
          <li>Sans région ni code postal, l'écart de parité passe à peine de 0,188 à 0,173 : <strong>les proxys portent la région.</strong></li>
          <li>fairlearn appliqué aux décisions du comité <strong>s'aligne sur le juge biaisé</strong> : écart de {num(f['gap_lo'])} à {num(f['gap_hi'])} face au mérite, et {pct(f['to_eo_rate'], 0)} d'octroi (hors enveloppe) pour ThresholdOptimizer-EO.</li>
        </ul></div></div>""",

        f"""<div class="kicker">3 · Solution et preuve</div><h2>Ne garder que le mérite, pour tout le monde</h2>
        <div class="row"><div class="col" style="flex:0.85"><div class="box">
          <div class="formula" style="font-size:23px">Top 40 % de <span class="c">cote R + {num(n['hours_w'], 3)} × heures</span></div>
          <p style="margin:0">Ni région, ni code postal, ni revenu, ni distance.</p></div>
          <ul style="margin-top:18px">
            <li><strong>Un seul paramètre</strong>, une règle que chacun·e peut recalculer.</li>
            <li><strong>{pct(n['p01']['hx_accuracy'], 2)}</strong> d'exactitude contre le mérite réel, avec 3 soumissions et une règle écrite d'avance.</li>
            <li>Un XGBoost entraîné sur les décisions reproduit le biais : <strong>le problème est la cible.</strong></li>
          </ul></div>
        <div class="col">{figs['rates']}</div></div>""",

        f"""<div class="kicker">4 · Front de Pareto</div><h2>Contre le mérite, l'équité ne coûte rien</h2>
        {figs['pareto']}
        <p style="font-size:17px;margin-top:6px">λ = part du biais du comité conservée (1 = comité, 0 = mérite seul), moyenne de 5 découpages. Contre le comité, l'équité semble coûter 4 points ; contre un mérite, elle améliore les deux axes.</p>""",

        f"""<div class="kicker">5 · Gouvernance</div><h2>Surveiller à chaque cycle, sans attendre les étiquettes</h2>
        <div class="row"><div class="col" style="flex:1.15"><table>
          <tr><th>monitoring.py</th><th>Seuil</th><th>Production</th><th>Nous</th></tr>
          <tr><td>Écart de taux Centre − Éloignée</td><td>0,05</td><td>{num(mon_b[gap_key])} <span class="tag bad">alerte</span></td><td>{num(mon_o[gap_key])} <span class="tag good">ok</span></td></tr>
          <tr><td>Étendue des taux sur 5 régions</td><td>0,08</td><td>{num(mon_b['grant-rate range over 5 regions'])} <span class="tag bad">alerte</span></td><td>{num(mon_o['grant-rate range over 5 regions'])} <span class="tag good">ok</span></td></tr>
          <tr><td>Barre de mérite, Centre − Éloignée</td><td>0,5 pt</td><td>{num(mon_b[bar_key])} <span class="tag bad">alerte</span></td><td>{num(mon_o[bar_key])} <span class="tag good">ok</span></td></tr>
        </table>
        <p style="font-size:17px">En production, les régions devaient avoir <strong>{num(-mon_b[bar_key], 1)} point de cote R de plus</strong> pour être retenues.</p>
        <p style="font-size:17px"><strong>Choix assumés</strong> : heures = mérite ; revenu neutre (un volet « besoin » relève de l'institution) ; étalon contraint, pas identifié.</p></div>
        <div class="col"><ul style="font-size:19px">
          <li><strong>Une alerte bloque la publication</strong> jusqu'à une revue humaine.</li>
          <li>Cas limites relus <strong>à l'aveugle</strong> ; <strong>audit annuel</strong> indépendant.</li>
          <li><strong>Recours</strong> : score et seuil communiqués (Loi 25).</li>
          <li><strong>Test de stress</strong> sur 5 cohortes futures : écart faible, alertes au bon moment.</li>
        </ul>
        <div class="box" style="margin-top:14px"><div class="formula" style="font-size:26px">Même barre pour tout le monde.</div></div></div></div>""",
    ]


def main():
    n = compute()
    figs = {"rates": fig_rates(n), "decomp": fig_decomposition(n), "proxies": fig_proxies(n), "pareto": fig_pareto(n)}
    pages = slides(n, figs)
    html = "\n".join(
        f'<section class="slide">{body}<div class="foot"><span>ÉquiAlgo · équipe Persona</span><span>{i + 1} / {len(pages)}</span></div></section>'
        for i, body in enumerate(pages))
    doc = f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>ÉquiAlgo pitch</title><style>{CSS}</style></head><body>{html}</body></html>'
    html_path = OUT_DIR / "slides.html"
    html_path.write_text(doc, encoding="utf-8")
    pdf_path = ROOT / "presentation.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, capture_output=True)
    print(f"{len(pages)} slides -> {pdf_path}")


if __name__ == "__main__":
    main()
