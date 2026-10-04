# Script du pitch (5 minutes, 6 slides)

Environ 520 mots, ce qui laisse de la marge pour respirer. Les temps sont cumulés. Les
**gras** marquent les phrases à ne pas rater.

---

**Slide 1 · Titre (0:00 → 0:15)**
Bonjour, nous sommes l'équipe Persona. Notre message tient en une phrase : **le mérite n'a pas
de code postal.**

**Slide 2 · Diagnostic (0:15 → 1:05)**
Le comité accordait une bourse à 48 % des candidat·es de Montréal et de Québec, contre 27 % en
région. Le modèle en production, « exact à 88 % », ne fait que copier ce comité. Nous avons
modélisé ses 10 000 décisions : **à dossier égal, vivre en région coûte 1,4 point de cote R.**
Sur l'écart de 21 points, 74 % viennent d'une pénalité régionale directe, 23 % d'un avantage
donné aux revenus élevés, et seulement 3 % du mérite.

**Slide 3 · Le piège (1:05 → 1:45)**
Le réflexe, c'est de supprimer la colonne région. Ça ne marche pas : le code postal et la
distance retrouvent la région presque parfaitement. Et égaliser l'égalité des chances contre
les décisions du comité, **c'est s'aligner sur le juge biaisé.** Les outils fairlearn appliqués
ainsi gardent un écart important, et l'un d'eux sort même du budget.

**Slide 4 · Solution et preuve (1:45 → 2:50)**
Nous séparons, dans la décision du comité, le mérite du biais, et **nous ne gardons que le
mérite** : la cote R plus 0,146 par heure travaillée. On accorde la bourse aux 40 % les mieux
classés, sans région, sans revenu, sans code postal. Un seul paramètre, une règle que chacun·e
peut recalculer. Avec trois soumissions et une règle écrite d'avance, nous avons mesuré que le
mérite réel est réparti également : 40 % au centre comme en région. Notre soumission atteint
94,6 % d'exactitude. Et un XGBoost entraîné sur les décisions reproduit le biais : **le problème
n'est pas la puissance du modèle, c'est sa cible.**

**Slide 5 · Pareto (2:50 → 3:30)**
À gauche, mesuré contre le comité, l'équité semble coûter quatre points d'exactitude. À droite,
mesuré contre un mérite, plus on retire de biais, plus on gagne sur les deux axes. **Contre le
mérite, l'équité ne coûte rien.**

**Slide 6 · Gouvernance (3:30 → 4:40)**
En production, un moniteur tourne à chaque cycle, sans attendre d'étiquettes. Sur l'ancien
modèle, il lève trois alertes, et il montre qu'**en région, il fallait 1,5 point de cote R de
plus pour être retenu.** Une alerte bloque la publication jusqu'à une revue humaine. Les cas
limites sont relus à l'aveugle, un audit annuel est mené contre un étalon indépendant, et chaque
personne refusée connaît son score et peut demander une révision. Nos choix sont assumés : les
heures comptent comme mérite, et un volet fondé sur le besoin reste une décision de
l'institution. Nos tests de stress sur cinq cohortes futures montrent un écart faible et des
alertes au bon moment.

**Conclusion (4:40 → 5:00)**
**Même barre pour tout le monde.** Tout est reproductible dans notre dépôt. Merci.

---

## Questions probables du jury

| Question | Réponse courte |
|---|---|
| Pourquoi l'égalité des chances plutôt que la parité ? | Ce sont des bourses de mérite : parmi les méritants, chaque région doit avoir la même chance. Comme le mérite est réparti également ici, on obtient aussi la parité. |
| Les heures ne sont-elles pas un proxy de la région ? | Si (AUC 0,80). Nous les gardons parce qu'elles font partie du mérite mesuré par les sondes, et nous vérifions le résultat : taux égaux entre groupes. |
| Vos sondes HxBuddy, c'est de l'overfitting ? | Trois soumissions agrégées, une règle écrite avant, aucune lecture individuelle. Tout est journalisé dans `probes/`. |
| Pourquoi pas ThresholdOptimizer en parité ? | Il ferme l'écart, mais avec un seuil différent par région au moment de décider (traitement différencié), et il colle moins au mérite (0,888 contre 0,902). |
| Et si le mérite réel pesait autrement les heures ? | Sur la plage compatible avec les sondes (0,75 à 1,25 fois), notre écart reste d'environ 0,03 au plus. |
| Comment ça généralise ? | Un seul paramètre (IC [0,13 ; 0,16]), aucune dérive entre l'historique et les candidat·es (PSI < 0,01), un test de stress sur 5 scénarios, et le moniteur en production. |
