# CR-02 — Paradigme maître–élève — 2026-11-23

## 1. En une phrase
Votre exemple du maître et de l'élève résout le blocage principal de CR-01 : en faisant **refaire** la
forme plutôt que la **nommer**, la récompense redevient pauvre en information et la Shadow Memory
acquiert un rôle de **gabarit intérieur** qui ne se réduit plus à une trace d'éligibilité.

## 2. Ce qui a changé
- `docs/PARADIGME_IMITATION.md` : conception v2 (perception, stylo, maître, Shadows, modèles G0–G5′).
- Journal : D09 (votre décision, adoptée), D10–D14 proposées.
- Prédictions P07–P11.
- Aucun code.

## 3. Ce dont j'ai besoin de vous
- **Q4. Le maître juge-t-il « un carré » ou « ce carré » ?** Votre phrase dit « récompenserait un carré
  refait ». Je l'ai lue comme un jugement de *catégorie* (D11). Si vous pensiez « une copie fidèle »,
  le paradigme mesure autre chose (la copie, pas le concept).
- **Q5. L'élève voit-il le geste du maître, ou seulement le résultat ?** Si le maître montre *comment*
  il trace, c'est de l'imitation de geste (plus facile, information plus riche) ; s'il montre seulement
  l'image, l'élève doit inventer le geste. J'ai pris « seulement l'image » comme cas principal et le
  geste comme référence G0.
- **Q6. L'élève a-t-il droit à plusieurs essais par démonstration ?** Votre intuition sur « le petit
  nombre d'exemples » porte-t-elle sur les démonstrations du maître, ou sur le nombre total de
  tentatives ? (D13 les compte séparément.)
- **Q7. L'élève se juge-t-il lui-même ?** Un enfant qui dessine voit que son carré « ne ressemble pas »
  avant que le maître ne parle. C'est G5 ; je l'ai rendu central. Confirmez-vous que c'est dans
  l'esprit de votre idée ?

## 4. Faits, interprétations, suppositions
| Type | Énoncé | Confiance |
|------|--------|-----------|
| Fait | Récompense scalaire sur un geste de 16 actions ≠ étiquette : elle ne dit pas quelle action corriger | certaine |
| Fait | Une Shadow perceptive utilisée comme gabarit fournit un signal (s_int) qu'aucune trace d'éligibilité ne fournit | certaine (par construction) |
| Interprétation | Le modèle biologique le plus proche est l'apprentissage du chant chez l'oiseau (gabarit + pratique + dopamine) | forte |
| Interprétation | La principale difficulté sera l'exploration : un carré par hasard est rare | forte |
| Supposition | Un espace moteur « stylo + virages » est assez simple pour une règle locale | moyenne |

## 5. Surprises
Le changement de paradigme supprime d'un coup les deux réductions de CR-01 (récompense = étiquette ;
Shadow = trace) — mais il en introduit une nouvelle : le gabarit interne pourrait suffire *seul*
(G5′), ce qui rendrait le maître presque superflu. Ce serait un résultat intéressant en soi.

## 6. Prédictions vs réalité
Aucune expérience. P07–P11 ajoutées (IA). Vos paris sont bienvenus avant l'étape 3.

## 7. Décisions
D09 adoptée (humain). D10–D14 proposées.

## 8. Limites
- Le choix de l'espace moteur (virages discrets incluant 90°) est un biais inductif fort : il facilite
  l'*expression* du carré. Un contrôle sans 90° est prévu.
- Le maître catégoriel est un critère géométrique que j'écris : il définit implicitement ce qu'est
  « un carré ». Il sera documenté et testé (élève aléatoire, élève parfait).

## 9. Prochaine étape
Étape 2 élargie : générateur + encodeurs + sondes, **plus** le stylo et le rendu partagé, avec une
galerie de gestes aléatoires et de gestes « parfaits ». Risque : que le rendu du stylo soit
distinguable du rendu du générateur (un biais de style qui fausserait le gabarit).
