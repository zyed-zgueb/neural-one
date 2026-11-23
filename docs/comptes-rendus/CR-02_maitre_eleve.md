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

## 3. Questions pour votre intuition
*(Révisé : les questions techniques initiales Q4–Q7 ont été tranchées par l'IA, voir D15–D18.)*

1. **Quand vous apprenez un geste nouveau (une signature, un pas de danse, un nœud), qu'est-ce qui
   reste en vous juste après que le modèle a disparu ?** Une image ? Une sensation dans la main ? Un rythme ?
   *Pourquoi je demande : cela me dit quoi mettre dans la mémoire Shadow.*
2. **Un bon maître fait-il autre chose que dire « bravo » ou « raté » ?** Que fait le meilleur
   professeur que vous ayez eu quand l'élève se trompe ?
   *Pourquoi : aujourd'hui mon maître est muet sauf un pouce levé ou baissé ; peut-être est-ce trop pauvre, ou au contraire l'essentiel.*
3. **Un enfant rate son carré. Qu'est-ce qui fait qu'il essaie *autrement* la fois suivante, plutôt que de refaire la même erreur ?**
   *Pourquoi : c'est le point où mes élèves artificiels risquent de bloquer.*
4. **Le souvenir du modèle montré doit-il s'effacer ?** Imaginez un élève qui ne l'oublie jamais, et un autre qui l'oublie en deux secondes : lequel apprend le mieux, et pourquoi ? Le sommeil a-t-il un rôle ?
   *Pourquoi : la vitesse d'oubli est un réglage central et je n'ai aucune bonne raison d'en choisir une.*
5. **À quoi reconnaîtriez-vous qu'un élève a vraiment « compris » le carré, et pas seulement appris à le recopier ?**
   Et s'il apprenait « trop vite pour être honnête », à quoi ressemblerait sa triche ?
   *Pourquoi : vos réponses deviendront des tests.*

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
