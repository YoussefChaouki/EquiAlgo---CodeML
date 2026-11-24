# ÉquiAlgo: merit has no postal code

🇫🇷 [Version française](README.fr.md)

Bias audit and correction of a scholarship-allocation model. Team **Persona** · Hackathon Génie
et Informatique 2026 · IVADO ÉquiAlgo challenge. The original challenge instructions are in
[`docs/`](docs/).

## In one minute

The historical committee awarded a scholarship to **48 %** of applicants from Montréal and the
Capitale-Nationale, against **27 %** in remote regions. The model in production (88 % accuracy)
faithfully reproduced that committee.

- **Diagnosis.** For an equal file, living in a remote region costs **1.4 points of R-score**
  (the Quebec college grade index). The 21-point gap breaks down into **74 % direct regional
  penalty**, **23 % advantage to high incomes** and **3 % merit**. Dropping the region column is
  not enough: postal code (AUC 1.00), distance (1.00), hours worked (0.80) and income (0.68)
  reconstruct it.
- **Correction.** We model the committee, separate its merit (R-score + 0.146 × hours worked)
  from its bias (region, income), then award the scholarship to the top **40 %** ranked on merit
  alone. There is **a single estimated parameter**, and the rule is explainable and
  reproducible.
- **Validation.** Three aggregated submissions to the challenge scorer (HxBuddy), following a
  rule written **before** submitting ([`probes/DECISION_RULE.md`](probes/DECISION_RULE.md)).
  True merit is evenly distributed (40.05 % in the centre, 39.74 % in the regions). Our
  submission reaches **94.63 % accuracy and 94.4 % macro F1** against that merit.
- **Governance.** [`monitoring.py`](monitoring.py) produces a fairness report at every award
  cycle. It would have raised 3 alerts on the production model: in production, the regions had
  to clear a bar 1.5 R-score points higher.

## Deliverables

| File | Content |
|---|---|
| [`predictions.csv`](predictions.csv) | 4,000 decisions, `id_candidat,decision_octroi`, award rate 40.0 % |
| [`audit_rapport.ipynb`](audit_rapport.ipynb) | Bias measurement, decomposition, proxies, chosen metric, identification of the reference, probes, comparison with boosting, governance, stress test, limits (in French) |
| [`model_corrige.py`](model_corrige.py) | Corrected model, Pareto front ([`pareto_front.png`](pareto_front.png)), robustness table |
| [`presentation.pdf`](presentation.pdf) | Slides of the 5-minute pitch (6 slides, in French), speaker script in [`presentation/pitch_script.md`](presentation/pitch_script.md) |

The files sit at the repository root because that is the layout the challenge asked for.

## Reproduce

Python 3.10+. The data from the official ZIP must be in `data/`
(`donnees_demandes.csv`, `candidats_evaluation.csv`); it is not redistributed here.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python model_corrige.py                 # predictions.csv + Pareto front (~30 s)
python -m pytest tests -q               # 8 tests: budget, format, metrics, decoder, monitor
jupyter nbconvert --to notebook --execute --inplace audit_rapport.ipynb   # ~1 min
python monitoring.py predictions.csv    # monitoring report for one cycle
python presentation/build_presentation.py   # regenerates presentation.pdf (needs Google Chrome)
```

All random seeds are fixed. `predictions.csv` is byte-for-byte identical to the probe
`probes/files/P01_r_hours_hat.csv`, which `python probes/make_probes.py` regenerates.

## Layout

```
common.py              data, groups, committee model, merit score, metrics, budget
model_corrige.py       corrected decision + Pareto front + robustness
monitoring.py          per-cycle monitoring indicators (label-free) and thresholds
audit_rapport.ipynb    full audit report
probes/                HxBuddy probes: generation, decoding, pre-registered rule, log
presentation/          pitch PDF generation and speaker script
tests/                 pytest tests
docs/                  official challenge instructions
```

## Stated choices and limits

- **Metric:** equal opportunity, measured **against a merit** and never against the committee's
  decisions. Here it implies parity in practice, since merit is evenly distributed.
- **Normative choices:** hours worked count as merit. Income is neither an advantage nor a
  need criterion; a need-based component is the institution's call.
- **The reference is constrained, not identified.** The probes rule out the competing
  hypotheses we tested, but a weight on hours of 0.75 to 1.25 times ours remains compatible.
  Our residual gap would then be about 0.03 at most.
- Audit over two regional blocks; intersectionality not audited; synthetic data.

## Tools

Python, pandas, NumPy, scikit-learn, statsmodels, fairlearn (ThresholdOptimizer,
ExponentiatedGradient, as comparison points), matplotlib, Jupyter, pytest. Development was
assisted by an AI coding assistant (Claude Code); the analysis, the choices and the deliverables
were reviewed and validated by the team.
