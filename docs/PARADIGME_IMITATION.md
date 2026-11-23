# Paradigme « Maître–Élève » : apprendre en refaisant

> Statut : **proposition v2, à valider**. Complète `CONCEPTION.md` (qui reste valable pour le générateur,
> les statistiques, l'instrumentation et les références). Origine : intuition humaine (D09).

## 1. L'idée

> Le maître montre un carré. L'élève refait un carré. Le maître récompense un carré refait.

Ce n'est plus une tâche de **classification** (dire « carré ») mais une tâche de **production**
(dessiner un carré). Trois conséquences changent la nature du problème.

### 1.1 La récompense redevient pauvre en information
En classification binaire, r = ±1 équivaut à l'étiquette (difficulté n°1 de la conception).
Ici, l'élève produit un geste de dimension élevée (une séquence de traits) et ne reçoit **qu'un scalaire**.
La récompense ne dit plus *quoi* corriger : le problème d'attribution du crédit devient réel, et c'est
exactement le terrain où un mécanisme de mémoire peut faire une différence.

### 1.2 La Shadow Memory reçoit un rôle qui n'est pas réductible à une trace
Le modèle est montré, puis **disparaît**. Pour refaire, l'élève doit **garder en lui** une représentation
de ce qu'il a perçu. La Shadow a alors deux fonctions distinctes :

| Fonction | Contenu | Rôle | Équivalent connu |
|----------|---------|------|------------------|
| **Shadow perceptive** (modèle intérieur) | état perceptif du carré montré | guide la production ; sert de **gabarit** pour juger sa propre production | mémoire de travail ; gabarit du chant chez l'oiseau chanteur |
| **Shadow motrice** (souvenir du geste) | états internes à chaque trait produit | permet d'attribuer la récompense, arrivée en fin de dessin, aux bons traits | trace d'éligibilité / replay |

La première fonction est le point qui manquait dans ma formalisation : **la Shadow ne stocke pas
seulement de quoi attribuer le crédit, elle stocke un contenu qui sert de référence**. L'élève peut
comparer « ce que j'ai dessiné » (re-perçu) à « ce que je garde du modèle » et produire un **signal
interne**, avant même le jugement du maître.

### 1.3 Refaire, c'est extraire la structure
Pour refaire un carré *à une autre taille, une autre position, une autre orientation*, il faut avoir
capté ce qui fait un carré (4 côtés égaux, 4 angles droits, fermeture), pas ses pixels. C'est
l'argument de Lake et al. (2015, *Bayesian Program Learning*) : les humains apprennent un caractère
manuscrit en un exemple parce qu'ils le perçoivent **comme une façon de le tracer**.

## 2. Ancrage dans les travaux existants (connu vs nouveau)

- **Apprentissage du chant chez l'oiseau** (diamant mandarin) : le jeune mémorise le chant du tuteur
  (gabarit), puis s'exerce ; une erreur de performance (comparaison chant produit / gabarit) est portée
  par la dopamine et module la plasticité (Konishi 1965 ; Fee & Goldberg 2011 ; Gadagkar et al. 2016).
  → **C'est le modèle biologique le plus proche de votre proposition.**
- **Apprentissage par imitation / démonstration** (behavior cloning, inverse RL).
- **Analyse par synthèse, graphique inverse** (Lake et al. 2015 ; Ganin et al. 2018 SPIRAL : un agent
  RL dessine pour reproduire une image).
- **Babillage moteur et modèles directs** (Wolpert, Kawato) ; neurones miroirs.
- **Règles à trois facteurs avec récompense en fin de séquence** (problème de la récompense distale,
  Izhikevich 2007).

**Ce qui est connu** : RL de dessin, imitation, gabarit chez l'oiseau.
**Hypothèse nouvelle à tester** : un réseau à **plasticité locale** (sans backprop) qui garde une
**Shadow perceptive** du modèle et une **Shadow motrice** de son geste, et combine (i) un signal
interne de comparaison au gabarit et (ii) la récompense globale du maître, apprend-il le *concept*
(généralisation à de nouvelles tailles, positions, rotations) avec **peu de démonstrations** ?

## 3. Le monde de l'élève

### 3.1 Perception
Même générateur et mêmes encodeurs que la conception (§2–3). Une image montrée → état perceptif `p`.

### 3.2 Espace moteur : un stylo, pas des pixels
Produire 1 024 pixels avec une récompense scalaire est hors de portée d'une règle locale (variance
énorme). L'élève contrôle donc un **stylo** :
- état initial (position de départ, cap, longueur de pas) déterminé par l'élève à partir de `p` ;
- puis T pas (ex. T = 16) ; à chaque pas, l'action est un **virage** choisi dans un ensemble discret
  (ex. {−90°, −45°, −22,5°, 0°, +22,5°, +45°, +90°}) ;
- le tracé est rendu par **le même moteur de rendu** que le générateur → l'élève et le maître
  partagent la même « physique ».

Un carré = « tout droit, tout droit, virage de 90°… » ; un cercle = « petit virage constant ».

> ⚠️ **Biais inductif fort et assumé** : le jeu de virages contient 90°. Il rend le carré facile à
> *exprimer*, pas à *apprendre* (il faut encore découvrir quand tourner). Variante de contrôle prévue :
> virages continus ou jeu d'angles ne contenant pas 90° exactement. Tous les modèles partagent
> le même espace moteur.

### 3.3 Le maître (environnement)
Le maître voit le dessin final et rend un jugement. Deux maîtres, à ne pas confondre :

| Maître | Critère | Ce qu'il récompense | Risque |
|--------|---------|---------------------|--------|
| **Copiste** | similarité pixel au modèle montré | la copie | récompense la **mémorisation** de l'exemple |
| **Catégoriel** (principal) | « est-ce un carré ? » : fermeture, nombre de coins, angles, égalité des côtés (calcul géométrique sur le tracé) | **un** carré, pas *ce* carré | le critère doit être réglé pour que le hasard ne le satisfasse pas |

Variante combinée : catégorie + taille/position approximativement respectées (on veut que l'élève
refasse *ce* carré mais qu'il soit jugé sur sa forme, pas sur sa superposition pixel).

Récompense binaire (+1/−1) ou graduée ; immédiate en fin de dessin, retardée (après plusieurs
dessins), bruitée — mêmes options que `FeedbackScheduler`.

## 4. Déroulement d'un essai

```
T0  Démonstration : le maître montre une image (carré ou cercle).
T1  L'image disparaît. Shadow perceptive ← p  (décroissance γ_p, bruit éventuel).
T2  Production : l'élève trace T pas ; à chaque pas, Shadow motrice ← (état, action).
T3  Re-perception : l'élève voit son propre dessin → p̂.
T4  Signal interne (optionnel) : s_int = sim(p̂, Shadow perceptive).
T5  Jugement du maître : r.
T6  Modulation : δ = combinaison de r et/ou s_int (brut ou erreur de prédiction).
T7  Plasticité locale sur les états de la Shadow motrice (et réactivation éventuelle).
```

L'élève peut **s'exercer plusieurs fois** sur une même démonstration (comme en vrai), ou non.

## 5. Modèles comparés (version imitation)

Même réseau de base pour tous : perception `p` (+ contexte : pas courant, position du stylo) → couche
cachée → politique sur les virages.

| Modèle | Apprentissage | Mémoire | Signal |
|--------|---------------|---------|--------|
| G0 | backprop, imitation supervisée du **geste** du maître (behavior cloning) — référence haute, *dispose d'une information que les autres n'ont pas* (le geste) | — | geste |
| G1 | REINFORCE avec backprop | — | r maître |
| G2 | local, trace d'éligibilité sur les T pas | trace synaptique | r maître |
| G3 | local, Shadow motrice (états + actions), attribution au moment de r | Shadow motrice | r maître |
| G4 | G3 + réactivation (rejouer le geste en interne et consolider) | Shadow motrice + réactivation | r maître |
| G5 | G3 + **Shadow perceptive comme gabarit** → signal interne s_int | Shadow perceptive + motrice | r maître + s_int |
| G5′ | G5 **sans maître** (s_int seul) | idem | s_int seul |
| Replay | replay classique des gestes avec r | tampon | r maître |

Comparaisons clés :
- **G5 vs G3** : le gabarit interne accélère-t-il l'apprentissage ? (cœur de l'hypothèse)
- **G5′** : l'élève apprend-il seul en se comparant à son souvenir ? Si oui, le maître sert surtout à
  *calibrer* ; si non, le gabarit seul ne suffit pas.
- **G3 vs G2** : stocker des états plutôt que des produits synaptiques change-t-il quelque chose
  quand la récompense arrive après 16 actions ?
- **G4 vs Replay** à nombre de mises à jour égal.
- **Contrôle « gabarit brouillé »** : G5 avec une Shadow perceptive remplacée par celle d'une *autre*
  image → si G5 apprend aussi bien, ce n'est pas le contenu du gabarit qui aide.

## 6. Comment savoir qu'il a appris le concept (et pas l'exemple)

1. **Reproduction de formes nouvelles** : montrer des carrés/cercles jamais vus (tailles, positions,
   rotations, épaisseurs différentes, y compris hors distribution) → taux de dessins jugés corrects
   par le maître catégoriel.
2. **Fidélité vs catégorie** : comparer la récompense « catégorie » à la similarité pixel avec les
   démonstrations d'apprentissage. Un élève qui redessine toujours le carré appris, quel que soit le
   modèle montré, a mémorisé.
3. **Reconnaissance dérivée** : sans entraînement supplémentaire, classer une image par
   « analyse par synthèse » (quelle catégorie de geste reproduit le mieux l'image) → comparable aux
   modèles M0–M5 de la conception.
4. **Rappel libre** (étape ultérieure) : « dessine un carré » sans modèle montré.

## 7. Comptabilité honnête des ressources

Trois quantités distinctes, toujours rapportées séparément :
- **démonstrations** (images montrées par le maître) — la métrique principale « peu d'exemples » ;
- **essais** (dessins produits, chacun jugé) — l'entraînement de l'élève ;
- **mises à jour** de poids (dont réactivations).

Apprendre avec 4 démonstrations mais 5 000 essais n'est pas la même découverte qu'avec 4
démonstrations et 20 essais. Les deux sont intéressantes ; elles ne doivent pas être confondues.

## 8. Difficultés scientifiques nouvelles

1. **Le critère du maître définit le concept.** Un maître catégoriel trop laxiste récompense des
   gribouillis fermés ; trop strict, rien n'est jamais récompensé (pas de signal). Il faudra vérifier
   qu'un élève aléatoire obtient un taux de récompense faible mais non nul.
2. **Exploration** : avec 7 virages et 16 pas, il y a 7¹⁶ gestes. Un carré exact est rare par hasard.
   Risque : aucune récompense au départ. Pistes : récompense graduée, maître « bienveillant » au début
   (curriculum), ou signal interne s_int qui fournit un gradient continu — ce qui est justement une
   raison théorique pour laquelle G5 *pourrait* aider.
3. **Le signal interne est aussi bon que la perception** : si la perception ne distingue pas carré et
   cercle, comparer son dessin au gabarit ne sert à rien. Les encodeurs (E0/E1/E2) restent un facteur.
4. **Position, taille, orientation** : faut-il que l'élève les reproduise ? Version 1 proposée : l'élève
   choisit départ et échelle à partir de la perception, mais le maître ne juge que la forme.
5. **Risque de copiste universel** : un élève qui apprendrait à suivre n'importe quel contour serait
   récompensé sans concept. Avec la Shadow perceptive seule (pas d'accès à l'image pendant le tracé)
   et une perception grossière, ce risque est limité ; à mesurer quand même (test sur formes
   hors catégorie, ex. triangle : un copiste le reproduit, un élève « carré/cercle » non).

## 9. Ce que deviennent les expériences précédentes

La tâche de classification (M0–M5) **reste** : elle sert de point de comparaison, de test de la
perception et de version simple pour valider le code. Le paradigme maître–élève devient
l'**expérience principale** sur l'hypothèse.

Plan révisé :
| Étape | Contenu |
|-------|---------|
| 2 | Générateur, encodeurs, sondes (inchangé) **+ moteur de stylo et rendu partagé** |
| 3 | Maître catégoriel et copiste ; validation : taux de récompense d'un élève aléatoire, d'un élève « parfait » |
| 4 | Réseau commun ; M0–M2 (classification) et G0–G2 (imitation) |
| 5 | Shadow perceptive et motrice ; G3–G5′ ; tests d'équivalence |
| 6 | Feedback retardé (dans la séquence et entre dessins) |
| 7 | Dashboard : galerie « modèle montré / dessin de l'élève / jugement » essai par essai |
| 8 | Campagnes pré-enregistrées et rapport |
