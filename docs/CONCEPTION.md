# SHADOW LEARNING LAB — Document de conception (Étape 1)

> Statut : **proposition à valider**. Aucun code n'est écrit tant que ce document n'est pas validé.
> Objet : fixer l'architecture technique, la formalisation mathématique des modèles, le protocole
> expérimental et les risques scientifiques **avant** toute implémentation.

---

## 0. Résumé exécutif

1. L'hypothèse testée est formulée de façon falsifiable (§1) : *à budget d'exemples, de paramètres,
   de mises à jour et de mémoire égal*, une mémoire Shadow + récompense globale + plasticité locale
   atteint-elle 90 % / 95 % de précision sur des formes **inédites** avec moins d'exemples que les
   références (backprop, REINFORCE, traces d'éligibilité, replay) ?
2. Point critique identifié : dans une tâche **binaire**, une récompense ±1 après la décision
   contient **exactement la même information que l'étiquette** (on déduit la bonne classe du signe).
   L'avantage potentiel de la Shadow Memory ne peut donc **pas** venir d'une information
   supplémentaire ; il ne peut venir que (a) d'une meilleure attribution du crédit en cas de feedback
   retardé/entrelacé, ou (b) d'un nombre de mises à jour plus grand par exemple (réactivation ≈ replay).
   Le protocole est construit pour séparer ces deux sources.
3. Prédiction a priori honnête : en feedback immédiat, Shadow A est **mathématiquement équivalente** à
   une trace d'éligibilité (on l'utilisera comme test unitaire). Les différences ne peuvent apparaître
   qu'avec feedback retardé, décisions intercalées, ou réactivation. Il est probable que M0
   (backprop supervisée) reste la référence la plus efficace en nombre d'exemples ; si c'est le cas,
   le laboratoire doit le montrer clairement.
4. La généralisation à partir de 4 images dépend d'abord du **biais inductif de la représentation**
   (pixels bruts vs. filtres fixes), pas de la règle d'apprentissage. La représentation d'entrée est
   donc un **facteur expérimental contrôlé**, identique pour tous les modèles.
5. Implémentation : **NumPy pur** (y compris la backprop de M0, écrite à la main) pour que chaque
   quantité (activations, traces, ΔW) soit observable et que tous les modèles partagent le même code
   réseau. Streamlit pour l'interface.

---

## 1. Hypothèses formalisées

On distingue l'hypothèse principale d'hypothèses secondaires, toutes avec un critère de réfutation.

| ID | Hypothèse | Comparaison | Réfutée si |
|----|-----------|-------------|------------|
| H1 | La Shadow Memory réduit le nombre d'exemples nécessaires (feedback immédiat) | M3/M4/M5 vs M2 | N90 non significativement inférieur (test apparié, correction Holm) |
| H2 | Shadow améliore l'attribution du crédit retardé | M3–M5 vs M2 vs Replay, délai D ∈ {1,2,4,8} | Pas de différence de précision finale / pente de dégradation avec D |
| H3 | Le gain de M5 vient de la réactivation et non du simple surcroît de mises à jour | M5 vs Replay à nombre de mises à jour égal | M5 ≤ Replay apparié |
| H4 | Stocker la décision est nécessaire (B > A) | M4 vs M3 | Pas de différence |
| H5 | Les modèles apprennent un concept et non les exemples | tous | Précision test ≈ k-NN/prototype sur mêmes features, ou effondrement en OOD |
| H0 (contrôle) | Le gain disparaît avec récompense aléatoire | tous, reward shuffle | Si un modèle « apprend » avec récompense aléatoire → artefact |

**Ce qui est connu** vs **ce qui est nouveau** :
- Connu : règles à trois facteurs (Δw = η·r·e), traces d'éligibilité, REINFORCE, replay d'expérience,
  apprentissage hebbien modulé, critique/erreur de prédiction de récompense.
- Hypothèse nouvelle (à tester, étroite) : stocker **des états neuronaux** (O(N) par épisode, indexés
  par épisode) plutôt que **des produits synaptiques** (O(N²), superposés), puis recalculer la trace
  au moment du feedback — éventuellement après **réactivation partielle** avec les poids courants.
  La question précise est : *ce découplage stockage-d'état / calcul-de-plasticité apporte-t-il un gain
  au-delà des traces et du replay, à ressources égales ?*

---

## 2. Générateur de formes (`data/shapes_generator.py`)

### 2.1 Rendu
- Rendu **vectoriel puis rasterisé par sur-échantillonnage** (ex. 8× → 256×256 puis moyenne 8×8 → 32×32),
  identique pour les deux classes : aucun artefact d'anti-aliasing propre à une classe.
- Forme = contour noir (valeur 0) sur fond blanc (1). En interne on utilisera l'« encre »
  `x = 1 − pixel` (fond = 0), plus naturel pour des neurones.
- Chaque image est entièrement déterminée par un vecteur de paramètres `θ` + une graine →
  traçabilité totale (on stocke `θ`, pas seulement l'image).

### 2.2 Paramètres
| Paramètre | Cercle | Carré |
|-----------|--------|-------|
| centre (cx, cy) | uniforme dans la zone valide (forme entièrement visible) | idem |
| taille | rayon r | demi-côté s |
| épaisseur contour | t (px) | t |
| rotation | sans objet (mais on tire quand même un angle « fantôme » pour équilibrer le flux aléatoire) | φ ∈ [0, 90°) |
| déformation | ellipticité légère, jitter radial basse fréquence | coins légèrement arrondis, jitter des sommets, légère asymétrie |
| bruit | gaussien / sel-poivre, niveau σ | idem |
| occlusion / fragmentation (niveaux hauts) | arcs manquants | segments manquants |

### 2.3 Contrôle des biais « raccourcis » (fondamental)
Un classifieur peut séparer cercles et carrés **sans comprendre la forme** si des statistiques
triviales diffèrent. À même « taille » nominale, un cercle de diamètre d a un périmètre πd et un carré
de côté d un périmètre 4d → **quantité d'encre différente**. Mesures :
- **Appariement de l'encre** : les tailles sont tirées de façon à ce que la distribution du périmètre
  (donc de l'encre totale) soit identique entre classes (on tire un périmètre P, puis r = P/2π, côté = P/4).
- **Appariement de l'encombrement** : la boîte englobante d'un carré tourné à 45° grossit ; on
  contrôle que la distribution de l'étendue spatiale est équilibrée.
- **Sondes anti-raccourcis (test automatique)** : une régression logistique entraînée sur
  {encre totale, centroïde, variance spatiale, boîte englobante, histogramme d'intensité} sur 10 000
  images doit rester à **≈ 50 % (IC 95 % contenant 0,5)**. Sinon le générateur est rejeté.

### 2.4 Niveaux de difficulté
| Niveau | Variations |
|--------|-----------|
| L0 | centré, taille fixe, épaisseur fixe, carré non tourné — contrôle de sanité |
| L1 | + position et taille variables |
| L2 | + épaisseur variable + rotation du carré |
| L3 | + déformations géométriques + bruit |
| L4 | + fragmentation / occlusion partielle |
| L-OOD | test avec paramètres **hors** de l'intervalle d'apprentissage (ex. apprentissage sur tailles petites, test sur grandes ; ou zones de position disjointes) |

### 2.5 Séparation train/test et absence de fuite
- Flux aléatoires **disjoints** : `SeedSequence(master).spawn()` → graines distinctes pour train,
  validation (réglage des hyperparamètres) et test. Le test n'est **jamais** utilisé pour régler quoi que ce soit.
- Vérification : distance minimale (pixels et paramètres θ) entre chaque image test et l'ensemble
  d'apprentissage ; on rapporte sa distribution et on exclut les quasi-doublons (seuil configurable).
- Jeu de test : **≥ 2 000 images** équilibrées par niveau (IC binomial à 95 % ≈ ±1,3 pt à 90 %).
- Évaluation stratifiée : précision par intervalle de taille, de rotation, d'épaisseur, de distance au
  plus proche exemple d'apprentissage → « sensibilité aux variations géométriques ».

---

## 3. Représentation d'entrée : facteur contrôlé

Avec 4 images et des formes qui changent de position, un classifieur sur pixels bruts est quasi
condamné au hasard quelle que soit la règle d'apprentissage. Si on ne contrôle pas ce point, **tous les
modèles échouent et l'expérience ne dit rien sur la Shadow Memory**.

On propose un **encodeur fixe, non appris, partagé par tous les modèles**, avec trois options :
- `E0 pixels` : x ∈ ℝ¹⁰²⁴ (référence brute).
- `E1 random` : projection aléatoire fixe + non-linéarité (contrôle « features sans a priori »).
- `E2 edges` : banc de filtres d'orientation fixes (type Gabor, 8 orientations) + pooling spatial grossier
  (ex. 4×4) → ~128 dimensions. Biais inductif inspiré de V1, explicitement documenté.

On compare toujours les modèles **à encodeur égal**. Le fait qu'un encodeur rende la tâche facile est
un résultat en soi, distinct de la question Shadow vs non-Shadow.

---

## 4. Réseau commun (`core/network.py`)

Un seul réseau, une seule initialisation, pour tous les modèles ; seule la **règle de mise à jour** change.

```
x ∈ ℝ^D  (sortie encodeur)
h = f(W1 x + b1)          ∈ ℝ^H   (H = 64 par défaut, f = ReLU ou sigmoïde)
z = W2 h + b2             ∈ ℝ^2
p = softmax(z / T)
a ~ p   (entraînement, politique stochastique)   |   a = argmax z (évaluation, déterministe)
```

- **Même nombre de paramètres** pour tous : |W1| + |b1| + |W2| + |b2|.
- **Deux régimes** pour la couche cachée :
  - `readout` : W1 fixe (aléatoire), seul W2 apprend — isole l'effet de la règle sur un lecteur linéaire ;
  - `full` : W1 et W2 apprennent (backprop pour M0/M1 ; règle locale pour M2–M5, voir §5).
- **Instrumentation** : chaque pas produit un enregistrement `StepRecord`
  (x, h, z, p, a, y, r, δ, traces/shadow, ΔW1, ΔW2, ‖W‖) via un système de *hooks*, désactivable
  pour les grandes campagnes (sinon O(H·D) par pas).
- **Évaluation** toujours en mode déterministe (argmax) pour **ne pas confondre stochasticité et apprentissage**.

---

## 5. Modèles et règles d'apprentissage

Notation : r ∈ {−1,+1} récompense brute ; δ = r − b signal modulateur, avec b = 0 (récompense brute)
ou b = V̂ (moyenne glissante → **erreur de prédiction de récompense**, RPE). Le choix brut/RPE est un
paramètre de toutes les variantes à récompense.

### M0 — Backprop supervisée (référence haute)
Perte d'entropie croisée avec l'étiquette y ; gradient exact par backprop manuelle ; SGD.

### M1 — Récompense, sans trace (REINFORCE)
Politique stochastique a ~ p, gradient de politique de Williams (1992) :
ΔW = η · δ · ∇_W log p(a|x), **avec backprop**.
Rôle : séparer l'effet « récompense au lieu d'étiquette » de l'effet « local au lieu de backprop ».
Le feedback doit être **immédiat** (pas de mémoire) ; en feedback retardé M1 ne peut associer r qu'à
l'activité courante → montre le problème de la récompense distale.

> Variante M1-local (optionnelle) : même règle mais sans backprop vers W1 → équivalente à M2 avec décroissance infinie.

### Règle locale commune à M2–M5 (règle à trois facteurs)
Couche de sortie (pré = h, post = action choisie) :
  e²ᵢⱼ = (𝟙[a=i] − pᵢ) · hⱼ
Couche cachée, sans backprop — **perturbation de nœud** : on injecte un bruit ξ ~ 𝒩(0,σ²) dans les
unités cachées (h = f(W1x + b1 + ξ)), et
  e¹ⱼₖ = ξⱼ · f′(·) · xₖ
Δw = η · δ · e. Les deux traces sont **locales** (pré, post, bruit propre au neurone) ; seul δ est global.
Option alternative à évaluer : AGREL/AuGMEnT (Roelfsema & van Ooyen 2005 ; Rombouts et al. 2015), qui
utilise un signal de « feedback attentionnel » depuis l'unité choisie — **à signaler** car ce signal
n'est pas purement global.

### M2 — Trace d'éligibilité classique (référence locale)
Par synapse : e(t) = λ·e(t−1) + c(t), où c(t) est le produit local ci-dessus au pas t, λ = exp(−1/τ).
À la réception de δ : Δw = η·δ·e(t). La trace est une **superposition** des épisodes passés.
Coût mémoire : O(|W|) par synapse.

### M3 — Shadow A (activations internes)
Tampon S de capacité K ; chaque entrée k = {x_k, h_k, ξ_k, force s_k, âge}.
s_k décroît : s_k ← γ·s_k à chaque pas ; éviction FIFO ou par force minimale.
À la réception de δ pour un épisode : Δw = η · δ · Σ_k ω_k · c(h_k, x_k, …), où ω_k dépend de la
**politique d'attribution** (§6). Shadow A **ne connaît pas l'action choisie** : pour la couche de
sortie, elle doit utiliser une action *reconstruite* â = argmax(W2 h_k) avec les poids courants, qui
peut différer de l'action réellement émise (exploration, ou poids modifiés entre-temps). La couche
cachée reste mise à jour via ξ_k. Tester si ce manque pénalise est l'objet de H4.

### M4 — Shadow B (activations + décision)
Comme A, plus a_k et p_k stockés → la trace de sortie exacte (𝟙[a=i] − pᵢ)hⱼ est recalculable.
**Résultat attendu par construction** : en feedback immédiat avec K=1, M4 ≡ M2 (test unitaire d'équivalence).

### M5 — Shadow C (état complet + réactivation partielle)
Stocke l'état complet (x, h, z, a, état précédent). Au feedback :
1. **Réactivation** : on réinjecte un indice partiel x̃ = m ⊙ x_k + bruit (m masque de fraction ρ,
   intensité α) dans le réseau **avec les poids courants** → h′, z′.
2. Optionnellement on mélange avec l'état stocké : h* = β·h_k + (1−β)·h′.
3. Plasticité appliquée sur l'activité réactivée ; R réactivations possibles (R = nombre de « replays »).
Différence avec M2 : la trace est **recalculée** avec les poids actuels et peut être appliquée plusieurs fois.

### Références supplémentaires (indispensables pour l'interprétation)
- **Replay** classique (Lin 1992 ; Mnih et al. 2015) : tampon (x, a, r) ; ré-présentation complète
  de l'entrée avec R répétitions. M5 avec ρ=1, α=1, β=0 **est** un replay : la nouveauté de M5 se
  limite à la réactivation partielle/bruitée et à la décroissance. On le montrera explicitement.
- **k-NN / prototype** sur la même représentation : référence de **mémorisation pure**. Un modèle qui
  ne fait pas mieux qu'un 1-NN n'a pas montré d'abstraction du concept.
- **Réseau non entraîné** et **récompense aléatoire** (contrôles négatifs).

### Tableau récapitulatif
| Modèle | Signal | Crédit cachée | Mémoire | Mémoire (floats) |
|--------|--------|---------------|---------|------------------|
| M0 | étiquette | backprop | — | 0 |
| M1 | récompense | backprop | — | 0 |
| M2 | récompense | perturbation de nœud | trace par synapse | |W| |
| M3 | récompense | idem | K × (D+H) | K(D+H+1) |
| M4 | récompense | idem | K × (D+H+2) | idem + décision |
| M5 | récompense | idem + réactivation | K × état complet | idem |
| Replay | récompense | idem | K × (x,a,r) | K(D+2) |

La **mémoire auxiliaire** est une ressource rapportée et, dans une expérience dédiée, égalisée
(ex. K choisi pour que K(D+H) ≈ |W|).

---

## 6. Feedback retardé et attribution (`learning/reward.py`, `experiments/delayed_reward.py`)

### 6.1 Ordonnanceur de feedback
Un `FeedbackScheduler` découple **décision** et **récompense** :
- `immediate` : r au pas t ;
- `delayed(D)` : r de l'épisode t délivrée au pas t+D, avec D décisions intercalées (elles-mêmes récompensées plus tard) ;
- `batched(K)` : un unique r = moyenne (ou signe de la majorité) après K décisions ;
- `noisy(p)` : r inversée avec probabilité p ;
- `raw` vs `rpe` : δ = r ou r − V̂.

### 6.2 Confusion majeure à éviter : l'information d'indexation
Dans le scénario T0–T7 du cahier des charges, le feedback dit « **la première** réponse était fausse ».
Ce feedback porte une **étiquette d'épisode** que la trace classique ne peut pas exploiter, par nature.
Si on donne cet index à Shadow et pas aux autres, on compare des modèles **qui ne disposent pas des
mêmes informations**. On teste donc deux protocoles :
- **Feedback indexé** : l'identité de l'épisode est fournie. Shadow peut cibler l'épisode k ;
  le replay aussi (il reçoit le même index) ; la trace ne le peut pas → l'avantage éventuel est attribué
  à « mémoire épisodique adressable », ce qui est **connu** (episodic control, replay).
- **Feedback non indexé** : seul r arrive. Shadow doit attribuer selon une politique (récence,
  force s_k, uniforme). Si elle pondère par récence exponentielle, elle **redevient** une trace :
  on s'attend à M3 ≈ M2. C'est le test le plus exigeant pour l'hypothèse.

Métriques spécifiques : précision vs délai D ; « erreur d'attribution » (fraction de ΔW attribuée au
mauvais épisode, mesurable car on connaît la vérité terrain) ; corrélation entre ΔW et le ΔW idéal
(celui qu'on aurait obtenu en feedback immédiat).

---

## 7. Protocole expérimental (`experiments/few_shot.py`, `benchmarks.py`)

### 7.1 Équité entre modèles
- **Conception appariée** : pour une répétition donnée, tous les modèles reçoivent les **mêmes images
  d'apprentissage, dans le même ordre, la même initialisation des poids, le même jeu de test**.
  On compare donc des différences intra-répétition (beaucoup plus puissant statistiquement).
- **Budget contrôlé sur trois axes distincts** :
  1. nombre d'exemples **distincts** N (la métrique principale) ;
  2. nombre de **présentations** (passes sur les exemples) — fixé identique, ex. 1 passe (online)
     ou P passes ;
  3. nombre de **mises à jour de poids** — les réactivations de M5 et le replay comptent comme
     des mises à jour ; on rapporte et, dans une expérience dédiée, on égalise.
- **Réglage des hyperparamètres équitable** : même budget de recherche (ex. 30 configurations
  aléatoires) par modèle, sur un **jeu de validation distinct**, sélection par la même métrique.
- Temps de calcul mesuré (wall-clock + nombre d'opérations approximatif).

### 7.2 Budgets
Proposition : N **par classe** ∈ {1, 2, 4, 8, 16, 32} (soit 2 à 64 exemples au total), car « 2
exemples » au total = 1 par classe. Le scénario phare « 2 cercles + 2 carrés » correspond à N=2/classe.

### 7.3 Métriques
- **Principale** : N90 et N95 = plus petit budget N pour lequel la précision test moyenne (ou par
  répétition) ≥ 90 % / 95 %. Données **censurées** (certains runs n'atteignent jamais le seuil) :
  on rapporte la proportion de runs atteignant le seuil à chaque N (courbe type survie) plutôt
  qu'une moyenne qui masquerait les échecs.
- Secondaires : précision, rappel, F1, matrice de confusion, aire sous la courbe d'apprentissage,
  écart train/test (mémorisation), précision vs 1-NN, précision stratifiée par paramètre géométrique,
  précision OOD, rétention.
- **Rétention** : après apprentissage, phase d'interférence (images non récompensées, ou récompense
  aléatoire, ou tâche voisine) puis re-test ; on mesure la chute de précision.

### 7.4 Statistique
- Répétitions : **≥ 100 par condition** (la variance due au tirage des 4 images est énorme) ;
  analyse de puissance pilote pour ajuster (objectif : détecter une différence de 5 pts avec puissance 0,8).
- Moyennes, écarts-types, **IC 95 % par bootstrap** (10 000 tirages).
- Comparaisons **appariées** : test de Wilcoxon signé + taille d'effet (Cliff's δ / différence médiane
  avec IC) ; correction de **Holm-Bonferroni** pour les comparaisons multiples.
- **Pré-enregistrement** : les hypothèses H1–H5, les métriques, les tests et les seuils sont
  figés dans `configs/preregistration.yaml` et commités **avant** les campagnes définitives.
  Toute analyse ajoutée après coup est étiquetée « exploratoire ».

---

## 8. Architecture logicielle

```
shadow_learning_lab/
├── pyproject.toml
├── README.md
├── docs/
│   ├── CONCEPTION.md          (ce document)
│   ├── MODELES.md             (maths de chaque modèle, liens littérature)
│   └── RAPPORT.md             (étape 7)
├── configs/
│   ├── default.yaml
│   ├── preregistration.yaml
│   └── experiments/*.yaml
├── src/sll/
│   ├── data/
│   │   ├── shapes_generator.py   (rendu, paramètres, niveaux)
│   │   ├── datasets.py           (splits disjoints, contrôle de fuite)
│   │   └── encoders.py           (E0 pixels, E1 random, E2 edges)
│   ├── core/
│   │   ├── network.py            (réseau commun + hooks d'instrumentation)
│   │   ├── records.py            (StepRecord, journalisation)
│   │   └── rng.py                (gestion des graines)
│   ├── learning/
│   │   ├── plasticity.py         (règle à 3 facteurs, perturbation de nœud)
│   │   ├── traces.py             (trace d'éligibilité classique)
│   │   ├── shadow_memory.py      (tampon, décroissance, capacité, réactivation)
│   │   ├── replay.py             (replay classique)
│   │   └── reward.py             (FeedbackScheduler, RPE, bruit)
│   ├── models/
│   │   ├── base.py               (interface Learner : act / receive_feedback / evaluate)
│   │   ├── baseline.py           (M0)
│   │   ├── reward_network.py     (M1)
│   │   ├── eligibility_network.py(M2)
│   │   ├── shadow_network.py     (M3, M4, M5)
│   │   └── references.py         (k-NN, prototype, replay, non entraîné)
│   ├── experiments/
│   │   ├── runner.py             (exécution appariée, cache, parallélisme)
│   │   ├── few_shot.py
│   │   ├── delayed_reward.py
│   │   ├── retention.py
│   │   └── benchmarks.py
│   └── analysis/
│       ├── metrics.py
│       ├── statistics.py         (bootstrap, Wilcoxon, Holm, censure)
│       └── plots.py
├── app/dashboard.py              (Streamlit)
├── results/                      (un dossier par run : config.yaml, hash, commit git, résultats .parquet)
└── tests/
```

Choix techniques :
- **NumPy** pour tout (réseaux de quelques milliers de paramètres → rapide sur CPU) ; backprop de M0
  écrite à la main et **vérifiée par différences finies** (test). PyTorch n'apporterait que de l'opacité ici.
- Interface unique `Learner` : `act(x) → (a, état)`, `receive_feedback(event)`, `evaluate(X)`.
  L'environnement (scheduler de feedback) pilote la boucle, ce qui garantit que tous les modèles voient
  exactement le même flux d'événements.
- Configurations en **dataclasses + YAML** ; chaque résultat est stocké avec sa config, le hash de
  config, le commit git et les graines → reproductibilité bit à bit.
- Parallélisation des répétitions par `joblib` (processus) ; résultats en Parquet.

Tests prévus (exemples) :
- générateur : déterminisme par graine, forme entièrement visible, **sonde anti-raccourcis ≈ 50 %**,
  absence de doublons train/test ;
- réseau : gradient backprop = différences finies ;
- **équivalence M4 ≡ M2** en feedback immédiat, K=1 ;
- **équivalence M5 ≡ Replay** pour ρ=1, α=1, β=0 ;
- perturbation de nœud : estimateur non biaisé du gradient (en espérance, sur un cas jouet) ;
- contrôle négatif : récompense aléatoire → précision ≈ 50 %.

---

## 9. Interface (`app/dashboard.py`, Streamlit)

Barre latérale : niveau de difficulté, encodeur, modèle(s), paramètres Shadow (K, γ, intensité, ρ, α, β, R),
signal (brut/RPE, bruit), feedback (immédiat/retardé D/groupé K), budget N, nombre de répétitions, graine.

Onglets :
1. **Données** — images d'apprentissage et échantillon de test, avec leurs paramètres θ et la distance au plus proche exemple d'apprentissage.
2. **Pas à pas** — un run unique, rejouable : entrée, activations cachées, décision, récompense, contenu de la Shadow (forces s_k), ΔW sous forme de cartes de chaleur (W1 reprojeté en 32×32).
3. **Courbes** — précision vs exemples consommés, avec IC ; plusieurs modèles superposés.
4. **Comparaison** — tableau N90/N95, tests appariés, tailles d'effet.
5. **Feedback retardé** — précision et erreur d'attribution en fonction du délai.

Les campagnes longues sont lancées en CLI ; l'interface lit les résultats en cache (pas de calcul lourd dans Streamlit).

---

## 10. Difficultés scientifiques identifiées

1. **Équivalence informationnelle récompense/étiquette en binaire.** Avec 2 classes, r = ±1 révèle
   l'étiquette. Pour que « récompense » signifie réellement moins d'information, il faudra plus tard
   passer à K > 2 classes (r ne dit plus quelle était la bonne classe). Proposé en extension.
2. **Variance de l'apprentissage local à récompense globale.** La perturbation de nœud a une variance
   qui croît avec le nombre de neurones ; avec 4 exemples, la couche cachée n'apprendra presque rien.
   En pratique, le régime `readout` sera probablement le seul informatif pour les petits budgets. C'est
   une limite à documenter, pas à masquer.
3. **Le few-shot dépend de la représentation.** Généraliser cercle/carré sur 4 images exige des
   invariances (position, taille, rotation) que ni une règle locale ni la backprop ne peuvent extraire
   de 4 exemples. D'où l'encodeur comme facteur contrôlé (§3).
4. **Shadow A ≡ trace d'éligibilité** dans le cas simple. La distinction n'existe qu'en présence de
   plusieurs épisodes en vol, de feedback indexé ou de réactivation. Il faut l'énoncer pour ne pas
   « découvrir » un effet qui n'est qu'un renommage.
5. **Réactivation = mises à jour supplémentaires.** Un gain de M5 peut provenir simplement de R mises
   à jour par exemple (comme le replay ou plusieurs époques pour M0). D'où le contrôle à nombre de mises
   à jour égal (H3).
6. **Information d'indexation du feedback retardé** (§6.2) : source majeure de comparaison injuste.
7. **Stochasticité.** Les modèles à récompense explorent ; leur précision en entraînement est bruitée.
   Évaluation toujours déterministe et sur test indépendant.
8. **Petits échantillons et censure** : N90 n'est pas défini si le seuil n'est jamais atteint ;
   variance énorme selon les 4 images tirées → nombreuses répétitions, appariement, statistiques robustes.
9. **Réglage des hyperparamètres** : un modèle mieux réglé gagne. Budget de recherche identique, sur validation.
10. **Plafond trivial** : en L0, tout modèle atteint 100 % → aucun pouvoir discriminant ; en L4,
    tout le monde est au hasard. Il faut un niveau « intermédiaire » où les modèles se séparent —
    déterminé par une étude pilote.

---

## 11. Liens avec la littérature (à détailler dans `docs/MODELES.md`)

- **Traces d'éligibilité / RL** : Sutton & Barto (2018), TD(λ) ; Williams (1992) REINFORCE.
- **Règles à trois facteurs** : Frémaux & Gerstner (2016) ; Gerstner et al. (2018) « Eligibility traces
  and plasticity on behavioral time scales » ; Izhikevich (2007) « Solving the distal reward problem » ;
  Legenstein et al. (2008) R-STDP.
- **Perturbation de nœud / poids** : Fiete & Seung (2006) ; Werfel et al. (2005) (analyse de variance).
- **Apprentissage sans backprop** : AGREL (Roelfsema & van Ooyen 2005), AuGMEnT (Rombouts et al. 2015),
  feedback alignment (Lillicrap et al. 2016), e-prop (Bellec et al. 2020).
- **Replay** : Wilson & McNaughton (1994) ; Lin (1992) ; Mnih et al. (2015) ; systèmes d'apprentissage
  complémentaires (McClelland et al. 1995 ; Kumaran et al. 2016).
- **Mémoire épisodique pour le RL** : Model-Free Episodic Control (Blundell et al. 2016) ; Neural
  Episodic Control (Pritzel et al. 2017).
- **Étiquetage synaptique et plasticité à échelle comportementale** : Frey & Morris (1997) synaptic
  tagging & capture ; Bittner et al. (2017) BTSP — biologiquement le plus proche de l'idée « Shadow ».
- **Few-shot** : réseaux prototypiques (Snell et al. 2017) — référence pour « mémorisation vs abstraction ».

---

## 12. Plan des étapes suivantes

| Étape | Livrable | Critère de fin |
|-------|----------|----------------|
| 2 | Générateur + splits + encodeurs + sondes anti-raccourcis | Tests verts, sonde ≈ 50 %, galerie d'images par niveau |
| 3 | Réseau commun, M0, M1, M2, k-NN, contrôles | Backprop vérifiée, courbes pilotes, choix du niveau discriminant |
| 4 | Shadow A/B/C + replay | Tests d'équivalence M4≡M2, M5≡Replay |
| 5 | FeedbackScheduler + expérience retardée | Erreur d'attribution mesurée |
| 6 | Dashboard Streamlit | Toutes les vues §9 |
| 7 | Pré-enregistrement, campagnes, rapport | Rapport avec IC, tests, conclusions positives **ou négatives** |

---

## 13. Décisions à valider

1. **Budgets** : par classe {1,2,4,8,16,32} (proposé) ou au total {2,…,32} ?
2. **Encodeur fixe** comme facteur contrôlé (E0/E1/E2) : accepté ? Ou pixels bruts uniquement ?
3. **M1 = REINFORCE avec backprop** (proposé, pour séparer « récompense » et « localité ») ?
4. **Crédit de la couche cachée** sans backprop : perturbation de nœud (purement global, proposé) et/ou AGREL (signal de feedback non strictement global) ?
5. **Régime de présentation** : une seule passe (online, proposé comme principal) + régime multi-passes à nombre de mises à jour égalisé ?
6. **NumPy pur** (proposé) plutôt que PyTorch ?
7. Extension future à **K > 2 classes** pour casser l'équivalence récompense/étiquette ?

---

## 14. Collaboration humain–IA et comptes rendus

Le laboratoire a un second objectif : **garder l'humain informé et en position de décider**.
Répartition assumée : l'humain apporte l'intuition (quelles idées valent d'être testées, ce qui
« sonne faux »), l'IA apporte le raisonnement formel, l'implémentation et l'analyse.

Dispositifs :
- **Compte rendu à chaque étape** (`docs/comptes-rendus/`, format `MODELE_CR.md`) : lecture en
  2 minutes en tête, puis séparation explicite **faits / interprétations / suppositions** avec niveau
  de confiance, surprises, limites, et **questions qui sollicitent l'intuition** plutôt que de simples
  validations.
- **Journal des décisions** (`JOURNAL_DECISIONS.md`) : qui a décidé, pourquoi, alternatives écartées.
- **Registre de prédictions** (`PREDICTIONS.md`) : humain et IA parient séparément avant chaque
  campagne ; la confrontation est systématique. Les surprises sont les résultats les plus informatifs.
- Les résultats négatifs sont rapportés avec la même visibilité que les positifs.
- Le dashboard inclura un onglet « Compte rendu » affichant le dernier CR et les prédictions.
