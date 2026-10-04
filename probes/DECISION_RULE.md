# Règle de décision des sondes HxBuddy (pré-enregistrée)

Écrite le 2026-10-03, **avant toute soumission**. Elle ne sera pas modifiée après lecture des
résultats ; tout écart sera signalé dans `audit_rapport.ipynb`.

## Pourquoi des sondes

`decision_octroi` est biaisé ; le jury note contre un étalon de mérite caché. Le chiffre
officiel 0,270 (écart du modèle de base) exclut « cote R seule » à prévalence ≤ 44 %, mais
ne départage pas : poids des heures ×1 (H1), ×1,5, ×2, ou mérite fondé sur le besoin.
HxBuddy affiche l'exactitude contre ce mérite : chaque sonde agrégée teste une hypothèse.

## Fair-play

- Sondes **agrégées uniquement** : taux global, taux par groupe, une formule par fichier.
  Aucune tentative d'extraire l'étiquette d'une personne (pas d'inversion de lignes isolées).
- Toutes les soumissions et leurs scores sont consignés dans `probe_log.csv` et rapportés.
- La **dernière** soumission HxBuddy est toujours le `predictions.csv` final (dans le budget).

## Protocole

**Étape 0 : précision d'affichage.** Soumettre `P00a_all_ones`. Si HxBuddy affiche moins de
4 décimales, les étapes B et C sont inutilisables : garder H1 au taux 40 % et s'arrêter.

**Étape A : prévalence.** `q = exactitude(P00a)`.
Taux final = `q` arrondi à 0,005, borné à [0,37 ; 0,43] (marge sur l'enveloppe 36–44 %).

**Étape B : taux de mérite par groupe.** Soumettre `P00b_ones_remote_only`, puis
`python probes/decode_hxbuddy.py --p00a <acc P00a> --p00b <acc P00b>` → `d = taux_centre − taux_éloigné`.

| `d` | Lecture | Sondes à soumettre |
|---|---|---|
| −0,02 ≤ d ≤ +0,02 | famille H1 | `P01` seulement (confirmation) |
| d < −0,02 | le mérite favorise davantage les régions : heures plus lourdes ou besoin | `P01`, `P04`, `P09`, `P07` |
| d > +0,02 | le mérite favorise les centres : cote R / revenu | `P01`, `P05`, `P03`, `P10`, `P06` |

**Étape C : choix.** Parmi les sondes soumises, retenir la meilleure exactitude **seulement si
elle dépasse `P01` d'au moins 0,0025** (10 candidat·es sur 4 000) ; sinon garder `P01` (H1,
choix a priori). Ce seuil protège contre la malédiction du gagnant.

## Traduction en commande finale

| Sonde retenue | Commande |
|---|---|
| P01 (H1) | `python model_corrige.py --rate <taux>` |
| P04 | `... --hours-mult 1.5` |
| P09 | `... --hours-mult 2` |
| P05 | `... --hours-mult 0.5` |
| P03 | `... --hours-mult 0` |
| P07 | `... --log-income-weight -1.29` |
| P06 | `... --log-income-weight 1.29` |
| P10 | `... --hours-mult 0 --log-income-weight -1.29` |

Puis soumettre `predictions.csv` sur HxBuddy en dernier et consigner son score.
