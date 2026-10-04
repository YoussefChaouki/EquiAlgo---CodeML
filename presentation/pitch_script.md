# Script du pitch (5 minutes, 9 slides)

Environ 700 mots, soit environ 140 mots par minute. Les temps sont cumulés. Les **gras**
marquent les phrases à ne pas rater.

---

**Slide 1 · Titre (0:00 → 0:15)**
Bonjour. Nous sommes l'équipe Persona. Notre message tient en une phrase : **le mérite n'a pas
de code postal.** Voici comment nous l'avons démontré, corrigé, et comment on le surveille.

**Slide 2 · Le constat (0:15 → 0:45)**
Le comité accordait une bourse à 48 % des candidat·es de Montréal et de Québec, contre 27 % en
région. L'écart de cote R n'est que de 0,7 point. Le modèle en production affiche 88 %
d'exactitude, mais cette exactitude se mesure contre les décisions du comité : **le modèle est
récompensé quand il copie le comité.**

**Slide 3 · Diagnostic (0:45 → 1:20)**
Nous avons modélisé les 10 000 décisions du comité. Résultat : **à dossier égal, vivre en région
coûte 1,4 point de cote R.** L'écart de 21 points se décompose ainsi : 74 % de pénalité régionale
directe, 23 % d'avantage aux revenus élevés, et seulement 3 % de mérite.

**Slide 4 · Le piège (1:20 → 2:00)**
Le réflexe, c'est de supprimer la colonne région. Ça ne marche pas : le code postal et la
distance retrouvent la région presque parfaitement, les heures et le revenu en partie. Second
piège : égaliser l'égalité des chances contre les décisions du comité, **c'est s'aligner sur le
juge biaisé.** Les outils fairlearn appliqués ainsi gardent un écart de 0,16 à 0,30, et l'un
d'eux sort même du budget.

**Slide 5 · Notre solution (2:00 → 2:40)**
Nous séparons ce qui, dans la décision du comité, relève du mérite et ce qui relève du biais.
Le mérite, c'est la cote R plus 0,146 par heure travaillée. Le biais, ce sont la région et le
revenu. **Nous accordons la bourse aux 40 % les plus méritants, sans région, sans revenu, sans
code postal.** Un seul paramètre estimé, une règle que chaque candidat·e peut recalculer. Nous
avons aussi testé un modèle de type XGBoost : entraîné sur les décisions, il reproduit le biais.
**Le problème n'est pas la puissance du modèle, c'est sa cible.**

**Slide 6 · Pareto (2:40 → 3:15)**
À gauche, mesuré contre le comité, l'équité semble coûter quatre points d'exactitude. C'est
l'illusion du vieux tableau de bord. À droite, mesuré contre un mérite, plus on retire de biais,
plus on gagne sur les deux axes. **Contre le mérite, l'équité ne coûte rien.**

**Slide 7 · La preuve (3:15 → 3:50)**
Nous avons écrit notre règle de décision **avant** de soumettre, puis utilisé trois soumissions
agrégées. Le mérite réel concerne 40 % des candidat·es au centre comme en région. Notre
soumission obtient 94,6 % d'exactitude. Sur le classement HxBuddy, les équipes de tête sont toutes autour de 94 % ; **nous y sommes avec
trois soumissions, parce qu'en production, il n'y a pas de classement pour vous corriger.**

**Slide 8 · Gouvernance (3:50 → 4:30)**
Notre moniteur tourne à chaque cycle, sans attendre d'étiquettes. Sur le modèle de production,
il lève trois alertes. Il révèle surtout ceci : **en région, il fallait 1,5 point de cote R de
plus pour être retenu.** Une alerte bloque la publication jusqu'à une revue humaine. Les cas
limites sont relus à l'aveugle, un audit annuel compare la règle à un étalon indépendant, et
chaque personne refusée connaît son score et peut demander une révision.

**Slide 9 · Limites (4:30 → 5:00)**
Nos choix sont explicites : les heures comptent comme mérite, et un volet fondé sur le besoin
reste une décision de l'institution. L'étalon est contraint par nos mesures, pas identifié
exactement. Nos tests de stress sur cinq cohortes futures montrent que l'écart reste faible
sans réentraînement, et que le moniteur alerte quand il le faut. **Même barre pour tout le monde.**
Merci.

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
