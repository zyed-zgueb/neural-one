# CR-01 — Étape 1 : conception — 2026-11-23

> Lecture en 2 minutes : sections 1 à 3.

## 1. En une phrase
La conception est prête, mais l'analyse montre que l'hypothèse Shadow, telle que formulée, **ne peut
pas** gagner en feedback immédiat sur une tâche binaire : son seul terrain de victoire plausible est
le feedback retardé, et même là il faut écarter deux explications triviales (indexation, replay).

## 2. Ce qui a changé
- `docs/CONCEPTION.md` : architecture, maths des modèles, protocole, statistiques.
- `docs/JOURNAL_DECISIONS.md` : 7 décisions proposées, 1 adoptée.
- `docs/PREDICTIONS.md` : mes 5 prédictions, écrites avant toute expérience.
- `docs/comptes-rendus/MODELE_CR.md` : format des comptes rendus.
- Aucun code.

## 3. Ce dont j'ai besoin de vous
Questions d'intuition (la réponse oriente l'expérience) :

- **Q1. Quand vous imaginez la Shadow Memory, que fait-elle que la trace ne fait pas ?**
  Mon analyse réduit Shadow A/B à une trace d'éligibilité et Shadow C à un replay. Si votre intuition
  contient un ingrédient que j'ai perdu en formalisant (ex. la réactivation *modifie* le souvenir,
  le shadow *influence* la décision suivante, il n'y a qu'un shadow « actif » à la fois…), c'est
  maintenant qu'il faut le dire : c'est précisément là que pourrait se trouver la nouveauté.
- **Q2. Selon vous, un humain apprend-il cercle/carré en 4 exemples grâce à la mémoire ou grâce à
  une perception déjà formée ?** Si c'est la perception, l'encodeur fixe (D02) est légitime ; sinon
  vous préférerez peut-être tout apprendre depuis les pixels, au risque que tout échoue.
- **Q3. Où placeriez-vous vos propres paris ?** Remplissez P06+ dans `PREDICTIONS.md` *avant* les
  résultats ; je confronterai vos prédictions et les miennes dans chaque compte rendu.
- Validations techniques D01–D07 : un « ok global » suffit si rien ne vous gêne.

## 4. Faits, interprétations, suppositions
| Type | Énoncé | Source | Confiance |
|------|--------|--------|-----------|
| Fait (math) | En binaire, r = ±1 après la décision révèle l'étiquette | déduction directe | certaine |
| Fait (math) | Shadow B avec K=1, feedback immédiat ≡ trace d'éligibilité | même formule Δw | certaine (sera testé) |
| Fait (math) | Shadow C avec réactivation complète ≡ replay d'expérience | même formule | certaine (sera testé) |
| Fait (géométrie) | À taille nominale égale, carré et cercle n'ont pas la même quantité d'encre (4d vs πd) | calcul | certaine |
| Interprétation | Le few-shot dépendra plus de l'encodeur que de la règle | littérature few-shot | forte |
| Interprétation | La perturbation de nœud n'apprendra presque rien en couche cachée avec 4 exemples | variance ∝ nb de neurones | moyenne |
| Supposition | Un niveau de difficulté intermédiaire séparera les modèles | à établir par pilote | faible |

## 5. Surprises
La plus importante : l'hypothèse se réduit, mathématiquement, à des mécanismes connus dans tous
les cas simples. Ce n'est pas une réfutation — c'est la localisation de l'endroit exact où la
nouveauté devrait se trouver (Q1).

## 6. Prédictions vs réalité
Aucune expérience encore. Prédictions P01–P05 enregistrées.

## 7. Décisions
D01–D07 proposées, D08 adoptée (voir journal).

## 8. Limites de ce compte rendu
- Analyse purement théorique ; les équivalences seront vérifiées par tests à l'étape 3–4.
- Je peux avoir formalisé la Shadow Memory trop étroitement (d'où Q1).
- Les références bibliographiques sont citées de mémoire ; elles seront vérifiées dans `MODELES.md`.

## 9. Prochaine étape
Étape 2 : générateur, splits disjoints, encodeurs, sonde anti-raccourcis, galerie par niveau.
Risque principal : ne pas réussir à égaliser toutes les statistiques triviales (encre, étendue)
sans dénaturer les formes ; la sonde dira si c'est le cas.
