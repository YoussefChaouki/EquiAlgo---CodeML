# ÉquiAlgo : le mérite n'a pas de code postal

Équipe **Persona** · Hackathon Génie et Informatique 2026 · défi IVADO ÉquiAlgo.
Les consignes originales sont dans [`docs/`](docs/).

## En une minute

Le comité historique accordait une bourse à **48 %** des candidat·es de Montréal et de la
Capitale-Nationale, contre **27 %** en régions éloignées. Le modèle en production (88 %
d'exactitude) reproduisait fidèlement ce comité.

- **Diagnostic.** À dossier égal, vivre en région éloignée coûte **1,4 point de cote R**.
  L'écart de 21 points se décompose en **74 % de pénalité régionale directe**, **23 %
  d'avantage aux revenus élevés** et **3 % de mérite**. Retirer la colonne région ne suffit pas :
  code postal (AUC 1,00), distance (1,00), heures (0,80) et revenu (0,68) la reconstituent.
- **Correction.** On modélise le comité, on sépare son mérite (cote R + 0,146 × heures
  travaillées) de son biais (région, revenu), puis on accorde la bourse aux **40 %** les mieux
  classés sur le mérite seul. Il n'y a qu'**un seul paramètre estimé**, et la règle est
  explicable et reproductible.
- **Validation.** Trois soumissions HxBuddy agrégées, selon une règle écrite **avant** de
  soumettre ([`probes/DECISION_RULE.md`](probes/DECISION_RULE.md)). Le mérite réel est réparti
  également (40,05 % au centre, 39,74 % en région). Notre soumission obtient **94,63 %
  d'exactitude et 94,4 % de F1 macro** contre ce mérite.
- **Gouvernance.** [`monitoring.py`](monitoring.py) produit un rapport d'équité à chaque cycle
  d'octroi. Il aurait levé 3 alertes sur le modèle de production : en production, les régions
  devaient franchir une barre plus haute de 1,5 point de cote R.

## Livrables

| Fichier | Contenu |
|---|---|
| [`predictions.csv`](predictions.csv) | 4 000 décisions, `id_candidat,decision_octroi`, taux d'octroi 40,0 % |
| [`audit_rapport.ipynb`](audit_rapport.ipynb) | Mesure du biais, décomposition, proxys, métrique choisie, identification de l'étalon, sondes, comparaison avec le boosting, gouvernance, test de stress, limites |
| [`model_corrige.py`](model_corrige.py) | Modèle corrigé, front de Pareto ([`pareto_front.png`](pareto_front.png)), table de robustesse |
| [`presentation.pdf`](presentation.pdf) | Support du pitch de 5 minutes (9 slides), script oral dans [`presentation/pitch_script.md`](presentation/pitch_script.md) |

## Reproduire

Python 3.10+. Les données du ZIP officiel doivent être dans `data/`
(`donnees_demandes.csv`, `candidats_evaluation.csv`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python model_corrige.py                 # predictions.csv + front de Pareto (~30 s)
python -m pytest tests -q               # 8 tests : budget, format, métriques, décodeur, moniteur
jupyter nbconvert --to notebook --execute --inplace audit_rapport.ipynb   # ~1 min
python monitoring.py predictions.csv    # rapport de surveillance d'un cycle
python presentation/build_presentation.py   # régénère presentation.pdf (Google Chrome requis)
```

Toutes les graines aléatoires sont fixées. `predictions.csv` est identique octet pour octet à
la sonde `probes/files/P01_r_hours_hat.csv`, qu'on régénère avec `python probes/make_probes.py`.

## Structure

```
common.py              données, groupes, modèle du comité, score de mérite, métriques, budget
model_corrige.py       décision corrigée + front de Pareto + robustesse
monitoring.py          indicateurs de surveillance par cycle (sans étiquettes) et seuils
audit_rapport.ipynb    rapport d'audit complet
probes/                sondes HxBuddy : génération, décodage, règle pré-enregistrée, journal
presentation/          génération du PDF du pitch et script oral
tests/                 tests pytest
docs/                  consignes officielles du défi
```

## Choix assumés et limites

- **Métrique :** égalité des chances, mesurée **contre un mérite** et jamais contre les décisions
  du comité. Ici, elle entraîne de fait la parité, puisque le mérite est réparti également.
- **Choix normatifs :** les heures travaillées comptent comme du mérite. Le revenu n'est ni un
  avantage ni un critère de besoin ; un volet fondé sur le besoin relève de l'institution.
- **L'étalon est contraint, pas identifié.** Les sondes excluent les hypothèses concurrentes
  testées, mais un poids des heures de 0,75 à 1,25 fois le nôtre reste compatible. Notre écart
  résiduel serait alors d'environ 0,03 au plus.
- Audit en deux blocs régionaux ; intersectionnalité non auditée ; données synthétiques.

## Outils

Python, pandas, NumPy, scikit-learn, statsmodels, fairlearn (ThresholdOptimizer,
ExponentiatedGradient, comme points de comparaison), matplotlib, Jupyter, pytest. Le
développement a été assisté par un assistant de code IA (Claude Code) ; l'analyse, les choix et
les livrables ont été relus et validés par l'équipe.
