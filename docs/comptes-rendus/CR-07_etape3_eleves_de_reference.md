# CR-07 — Étape 3 : élèves de référence (sans Shadow) et système visuel inné — 2026-11-23

> Lecture en 2 minutes : sections 1 à 3. Figures : `docs/figures/step3_*.png`.
> Données : `results/step3/` (graine 0, commit indiqué dans chaque JSON).

## 1. En une phrase
Sans Shadow, trois faits dominent : **la perception décide de presque tout** ; quand l'élève peut
revoir ses exemples, **apprendre par récompense locale vaut la rétropropagation** ; mais quand chaque
expérience n'est vécue **qu'une fois**, l'élève local est très en retard — c'est exactement l'écart
que la Shadow Memory devra combler. Côté dessin, les élèves apprennent le cercle et **jamais le carré** :
ils cessent de regarder le modèle.

## 2. Ce qui a changé
- Réseau commun (`core/network.py`), élèves de reconnaissance M0–M2 + références, élèves dessinateurs
  G0–G2, système visuel inné SV0 (`data/innate_vision.py`), 48 tests verts.
- Trois campagnes : reconnaissance (100 répétitions, réglage équitable, ~54 min), dessin
  (10 répétitions × 8 000 essais), système inné (30 répétitions, exploratoire).

## 3. Questions pour votre intuition
1. **Mes élèves dessinateurs finissent par faire des ronds quoi qu'on leur montre.** Avez-vous déjà vu un
   enfant (ou vous-même) répéter le dessin qu'il sait faire au lieu de regarder le modèle ? Qu'est-ce
   qui le fait enfin regarder ?
   *Pourquoi : c'est la panne principale ; votre réponse oriente le premier rôle de la Shadow.*
2. **La fermeture ne sert à rien pour distinguer un carré d'un cercle (les deux sont fermés).** Selon
   vous, à quoi sert alors au bébé de savoir qu'une forme est fermée ?
   *Pourquoi : je pense qu'elle sert à se juger soi-même en dessinant (« mon trait se referme ») ; votre
   intuition peut confirmer ou proposer autre chose.*
3. **Un élève qui ne fait que retenir par cœur les exemples réussit aussi bien que les meilleurs.**
   Qu'est-ce qui, pour vous, distinguerait de l'extérieur un élève qui a compris d'un élève qui a juste
   une bonne mémoire, quand les deux répondent juste ?

## 4. Faits, interprétations, suppositions

### Reconnaissance (niveau L2, formes jamais vues, 100 répétitions appariées)
| Type | Énoncé | Confiance |
|------|--------|-----------|
| Fait | Pixels bruts (E0) et orientations localisées (E2) : aucun élève ne dépasse 69 % avec 32 exemples par forme | — |
| Fait | Histogramme d'orientations (E3), 20 passages : les quatre règles d'apprentissage atteignent 90 % avec **4 exemples par forme** et 95 % avec 8 ; écarts ≤ 1,7 point | — |
| Fait | E3, **1 seul passage** (chaque exemple vu une fois), 8 exemples/forme : rétropropagation 96 %, élève local M2 80 % (−16 points, p < 10⁻¹², Holm), REINFORCE 84 % ; M2 n'atteint jamais 90 % en moyenne à 32 exemples | — |
| Fait | Le « plus proche voisin » (mémoire pure) égale la rétropropagation en 1 passage (90 % à 8 exemples/forme) | — |
| Fait | Récompense aléatoire : 47–54 % pour tous (contrôle négatif réussi) | — |
| Interprétation | Avec deux formes, la récompense ±1 contient l'étiquette : à expérience répétée, la règle locale n'est pas un handicap | forte |
| Interprétation | Le handicap de la règle locale est **par expérience vécue** : elle tire moins d'une seule présentation. C'est le terrain légitime de la Shadow (réactiver une expérience = en tirer plus), mais aussi celui du replay : la comparaison Shadow vs replay à mises à jour égales est décisive | forte |
| Interprétation | Avec E3, la généralisation est dans la perception : un élève qui mémorise suffit. Le succès en reconnaissance ne prouve donc pas la compréhension — d'où le « test de l'élève qui sait » par le dessin (D28) | forte |

### Dessin (formes jamais vues, 10 répétitions)
| Type | Énoncé | Confiance |
|------|--------|-----------|
| Fait | Avec « presque ça », G1 et G2 dessinent des cercles acceptés (80–100 %) mais **0 % de carrés**, après 8 000 essais | — |
| Fait | Avec le pouce seul : G1 15 %, G2 0–4 % (uniquement des cercles) | — |
| Fait | Remis devant **son propre carré d'entraînement**, l'élève trace un cercle | diagnostic sur 2 répétitions |
| Fait | G0 (imite le geste du maître) : cercles 100 %, carrés 5 % (1 démo) à 30 % (4 démos) | — |
| Interprétation | Les élèves convergent vers le geste qui rapporte le plus en moyenne (le cercle) et **cessent d'explorer et de regarder le modèle** | forte |
| Interprétation | Le carré exige une séquence précise conditionnée par ce qu'on a vu ; le cercle se fait « à l'aveugle » (toujours le même virage) | moyenne |

### Système visuel inné SV0 (exploratoire, hyperparamètres fixés)
| Type | Énoncé | Confiance |
|------|--------|-----------|
| Fait | La topologie seule (fermeture, morceaux, bouts, jonctions, trous) : 48–52 % pour cercle/carré — elle ne contient pas la réponse, comme prévu | — |
| Fait | SV0 = E3 en L2 (orientation globale + topologie) : mêmes résultats que E3 | — |
| Fait | En L4 (contours fragmentés), ajouter la topologie **dégrade** l'apprentissage (M0, 32 ex. : 87 % contre 93 %) | — |
| Fait | Ajouter une carte grossière de position des orientations effondre tout (89 % → 67 % à 8 ex./forme) | — |
| Interprétation | Pour reconnaître, **moins mais invariant** vaut mieux que **plus mais variable** ; des primitives exactes mais non pertinentes pour la tâche gênent l'apprentissage à peu d'exemples | forte |
| Supposition | La fermeture sera utile au **dessin** (auto-évaluation), pas à la reconnaissance | à tester (H11) |

## 5. Surprises
1. **Le cercle est appris, le carré jamais** — alors que le hasard frôle plus souvent le carré (CR-06).
   Mon pari P11 (« cercle avant carré ») se trouve donc confirmé, pour une raison que je n'avais pas vue.
2. **L'élève local égale la rétropropagation** dès qu'il revoit ses exemples. Je pariais que la
   rétropropagation garderait l'avantage (P02) : vrai seulement en un passage.
3. **Ma version du système inné était mauvaise** à cause d'un choix à moi (la carte de position), pas de
   votre intuition. Corrigé : la carte est désormais optionnelle et désactivée.

## 6. Prédictions vs réalité
| ID | Prédiction | Résultat | Statut |
|----|-----------|----------|--------|
| P02 (IA) | M0 a le plus petit N90 | 1 passage : oui (à égalité avec la mémoire pure) ; 20 passages : égalité des 4 règles | partiellement juste |
| P03 (IA) | Pixels, L2 : personne à 90 % avec 2 ex./forme | personne ne dépasse 58 %, même à 32 | juste |
| P11 (IA) | Cercle appris avant le carré | cercle 80–100 %, carré 0 % | juste (pour une autre raison) |
| P13 (IA) | Encouragement gradué nécessaire au début | pouce seul : 0–15 % ; gradué : 40–50 % | juste |
| P22 (humain) | Ouvert/fermé sont des primitives innées utiles | inutiles pour reconnaître cercle/carré (par construction) ; utilité pour le dessin à tester | en cours |
| P23 (IA) | SV0 ≈ E3 en L2, SV0 > E3 en L4 | L2 : oui ; L4 : **non**, SV0 < E3 | faux |

## 7. Décisions
D36 (SV0 sans carte de position par défaut), D37 (cibles de l'étape 4). Voir le journal.

## 8. Limites
- Comparaison SV0 exploratoire (hyperparamètres non réglés, 30 répétitions).
- Dessin : 10 répétitions, réglage léger (6 configurations), un seul niveau (L2).
- Les élèves dessinateurs utilisent un « réflexe postural » fourni (point de départ, taille du pas).
- Les références bibliographiques de §13 restent à vérifier.

## 9. Prochaine étape (étape 4 : la Shadow Memory)
Deux cibles précises, mesurables, issues de ces résultats :
1. **Reconnaissance, 1 passage** : la Shadow (réactiver le souvenir d'une expérience) permet-elle à
   l'élève local de rattraper la rétropropagation — et **mieux qu'un replay** à nombre de mises à jour égal ?
2. **Dessin** : la Shadow perceptive (garder le modèle en tête et y comparer son dessin, « regarde bien
   le modèle ») et le signal inné de fermeture débloquent-ils le carré ?
