# CR-06 — Étape 2 : le monde de l'élève est prêt — 2026-11-23

> Lecture en 2 minutes : sections 1 à 3. Figures dans `docs/figures/step2_*.png`.

## 1. En une phrase
Le monde fonctionne : le maître montre des formes sans raccourci exploitable (à partir du niveau L2),
reconnaît sans erreur les carrés et cercles bien faits, et **un élève qui gribouille au hasard
n'obtient presque jamais de pouce levé** — 1 carré sur 60 000 essais, aucun cercle. Sans
encouragement, l'élève ne pourrait pas apprendre : votre intuition « c'est presque ça » devient
une nécessité, pas un confort.

## 2. Ce qui a changé
- Code : générateur de formes, moteur de rendu partagé, niveaux, encodeurs, stylo, maître
  catégoriel et copiste, sondes anti-raccourcis (`src/sll/`).
- 34 tests, tous verts. Expérience reproductible : `experiments/step2_world.py` (graine 0, ~70 s).
- Figures : galerie des niveaux, galerie des gestes, sondes, notes du hasard.

## 3. Questions pour votre intuition
1. **Regardez le « meilleur carré » trouvé par hasard (figure des gestes, dernière ligne, coché ✓).**
   Le maître l'accepte. Vous, l'accepteriez-vous comme un carré « avec des défauts acceptables » ?
   *Pourquoi : c'est la frontière de tolérance du maître ; votre réponse la règle.*
2. **Le hasard s'approche bien plus souvent d'un carré que d'un cercle.** Pour un enfant, qu'est-ce
   qui est plus facile à dessiner : un carré ou un rond ? Et pourquoi, selon vous ?
   *Pourquoi : j'avais parié l'inverse (P11) ; votre expérience m'aide à comprendre lequel de nous deux se trompe.*

## 4. Faits, interprétations, suppositions
| Type | Énoncé | Source | Confiance |
|------|--------|--------|-----------|
| Fait | Le maître accepte 100 % des carrés/cercles du générateur en L0–L2 et hors distribution, 0 % de la mauvaise classe | 300 formes/classe/niveau | — |
| Fait | En L3–L4, 2 à 7 % des cercles déformés sont refusés (trop ovales) ; aucun carré refusé | idem | — |
| Fait | Gestes « parfaits » de toutes tailles (7–24 px) et rotations : 100 % acceptés | 200 + 200 | — |
| Fait | Stylo et générateur : images identiques pour le même carré (IoU = 1,00) → aucun biais de style | 50 carrés | — |
| Fait | Gribouillis aléatoires : 1 carré accepté / 60 000, 0 cercle / 60 000 (IC 95 % < 0,03 %) | 3 × 20 000 | — |
| Fait | « Presque ça » : 0,25 % des gribouillis dépassent 0,5 pour le carré, 0 % pour le cercle | idem | — |
| Fait | Aucune fuite train/test : 0 quasi-doublon ; distance médiane au plus proche exemple d'apprentissage > distance typique entre images de test | 3 niveaux | — |
| Fait | Niveaux L2–L4 : encre, position, épaisseur ne révèlent pas la forme (49–53 %) | 3 000 images/niveau | — |
| Fait | Niveau L1 : la **texture du trait** révèle la forme (80 %) | idem | — |
| Interprétation | En L1, un carré non tourné a des traits alignés sur la grille de pixels : nets, sans dégradé, alors que le cercle est toujours anti-crénelé. C'est un indice réel mais trivial. | analyse | forte |
| Interprétation | Le cercle est plus dur à trouver par hasard que le carré : il exige 16 virages identiques d'affilée, alors qu'un quadrilatère approximatif tolère des irrégularités | analyse des gestes | moyenne |
| Supposition | Un apprentissage guidé par « presque ça » pourra remonter la pente de 0,4 à 1 | à tester étape 3 | faible |

## 5. Surprises et corrections
1. **Raccourci de position détecté et corrigé.** Les carrés droits étant plus petits que les cercles
   de même périmètre, ils pouvaient s'approcher plus des bords : la position trahissait la forme (57 %).
   Corrigé : la zone de placement ne dépend plus de la classe (sonde revenue à 48–50 %).
2. **Le maître était trop généreux.** Ma première échelle donnait 0,99 sur 1 à un gribouillis
   informe. Corrigé : la note décroît dès qu'on s'éloigne du parfait, même à l'intérieur de la
   tolérance. Le meilleur gribouillis plafonne désormais à 0,69 (carré) et 0,46 (cercle).
3. **Le cercle est plus difficile que le carré pour un élève qui explore** — contraire à mon pari P11.

## 6. Prédictions vs réalité
| ID | Prédiction | Ce qu'on voit déjà | Statut |
|----|-----------|--------------------|--------|
| P07 | Sans note graduée, presque jamais de récompense au début | 1 / 60 000 | **confirmée** (pour l'exploration aveugle) |
| P11 | Le cercle sera appris avant le carré | le cercle est bien plus rare par hasard | **indice contraire** (à confirmer en apprentissage) |

## 7. Décisions
D29 (L2 = niveau de référence ; L1 garde un indice de texture documenté), D30 (échelle de note
révisée), D31 (le maître juge le tracé, pas les pixels). Voir le journal.

## 8. Limites
- Le maître juge un tracé (suite de points). C'est un juge géométrique fiable, mais plus précis
  qu'un œil humain sur une image floue.
- Les gribouillis aléatoires partent du centre avec un pas fixe ; un élève réel choisira aussi
  position et taille.
- Les tolérances (« carré acceptable ») sont mes choix ; elles attendent votre regard (question 1).

## 9. Prochaine étape
Étape 3 : le réseau commun et les premiers élèves — références de classification (M0–M2) et premiers
élèves dessinateurs (G0–G2) : behavior cloning, REINFORCE, traces d'éligibilité. Premier vrai test :
un élève sans Shadow parvient-il à remonter la pente de « presque ça » ?
