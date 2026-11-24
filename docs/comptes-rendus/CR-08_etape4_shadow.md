# CR-08 — Étape 4 : la Shadow Memory à l'épreuve — 2026-11-24

> Lecture en 2 minutes : sections 1 à 3. Figures : `docs/figures/step4_reconnaissance.png`, `step4_dessin.png`.
> Données : `results/step4/` (graine 0 ; commit dans chaque JSON).

## 1. En une phrase
**Pour reconnaître, la Shadow marche** : en réactivant ses souvenirs, l'élève local passe de 80 % à
97 % avec 8 exemples par forme vécus une seule fois, et Shadow C bat le replay — mais tout le gain
vient d'un seul ingrédient, le **rappel imparfait**, et la rétropropagation dotée du même replay reste
devant. **Pour dessiner, rien ne marche** : aucun élève, avec ou sans Shadow, ne trace un seul carré ;
le blocage n'est pas la mémoire, c'est l'exploration des gestes.

## 2. Ce qui a changé
- `learning/shadow_memory.py` (mémoire : capacité, oubli, force, renforcement par le rappel) ;
  `models/shadow_classifiers.py` (Replay, Shadow A/B/C, rétropropagation + replay) ;
  `models/shadow_drawers.py` (Shadow motrice, gabarit intérieur, sensation de fermeture, contrôles).
- Tests d'équivalence : Shadow B sans réactivation ≡ trace classique ; Shadow C neutre ≡ replay. 53 tests verts.
- Trois campagnes : reconnaissance (100 répétitions), ablation de Shadow C (100 répétitions), dessin
  (11 élèves × 2 × 10 répétitions × 8 000 essais, ~47 min).

## 3. Questions pour votre intuition
1. **Mes élèves n'arrivent jamais au carré en essayant seuls, même encouragés ; le seul qui y arrive un
   peu (30 %) est celui qui voit le maître tracer le geste.** Un enfant apprend-il le carré en regardant
   le résultat, ou en regardant la main du maître ?
   *Pourquoi : cela décide si l'étape suivante doit montrer le geste (votre « il remontre en insistant »).*
2. **Pour vous, un carré est-ce « quatre fois la même chose » ?** Quand vous le dessinez, pensez-vous
   « un côté, je tourne, et je recommence » ?
   *Pourquoi : un élève qui saurait répéter un petit geste trouverait le carré presque immédiatement ;
   mais lui donner cette capacité est un choix fort.*
3. **Le rappel imparfait aide, le rappel qui ancre n'aide pas.** Dans votre expérience, est-ce le fait
   de se souvenir *flou* qui fait généraliser, ou le fait de se souvenir *souvent* ?

## 4. Faits, interprétations, suppositions

### Reconnaissance — chaque exemple vécu une fois, 20 mises à jour par expérience (100 répétitions)
| Exemples/forme | 1 | 2 | 4 | 8 | 32 |
|---|---|---|---|---|---|
| Local sans mémoire | 59 % | 64 % | 71 % | 80 % | 97 % |
| Local + replay | 70 % | 81 % | 88 % | 94 % | 98 % |
| Local + Shadow C | **78 %** | 82 % | **90 %** | **97 %** | 97 % |
| Rétropropagation, 1 passage | 59 % | 73 % | 86 % | 96 % | 100 % |
| Rétropropagation + replay | **88 %** | **93 %** | **97 %** | **98 %** | 100 % |

| Type | Énoncé | Confiance |
|------|--------|-----------|
| Fait | Shadow C > replay à 8 ex./forme : +3,2 points (IC 95 % [+0,8 ; +5,6], p_Holm < 10⁻⁴) ; pas de différence significative à 2 et 32 | — |
| Fait | Ablation : sans rappel partiel, Shadow C perd 10 points ; sans mélange, sans renforcement par le rappel : aucun changement ; sans oubli : léger recul non significatif | — |
| Fait | Replay + rappel partiel seul : +4,7 points sur le replay (98,2 %), au niveau de la rétropropagation + replay (98,3 %) à 8 ex./forme | — |
| Fait | Shadow A (sans garder la décision) : −10 à −24 points | — |
| Fait | Rétropropagation + même replay : +10 à +12 points sur Shadow C à 1–2 exemples/forme | — |
| Interprétation | Le rappel imparfait agit comme une **régularisation** : il empêche d'apprendre les détails exacts des quelques exemples. C'est très proche de techniques connues (bruit d'entrée, « dropout ») | forte |
| Interprétation | « Se rappeler reconstruit » : soutenu (version imparfaite). « Se rappeler ancre » : pas d'effet mesurable ici | moyenne |

**Correction importante en cours de route** : mon premier replay rejouait d'anciens choix sans tenir
compte du fait que l'élève ne les ferait plus ; il *dégradait* l'élève (64 % au lieu de 80 %). J'ai
ajouté la correction standard (pondérer chaque souvenir par la probabilité actuelle du choix). Sans
cette correction, la Shadow aurait semblé gagner contre un adversaire de paille.

### Dessin (10 répétitions, 8 000 essais, formes jamais vues)
| Type | Énoncé | Confiance |
|------|--------|-----------|
| Fait | 0 carré accepté pour **tous** les élèves, sur le test **et sur leurs propres carrés d'entraînement** | — |
| Fait | Meilleurs cercles : Shadow motrice G3 (90–100 %), rétropropagation + bonus d'exploration (100 %) | — |
| Fait | Gabarit intérieur et/ou fermeture : cercles en recul (0–20 %), carrés toujours 0 % ; le contrôle « gabarit brouillé » fait aussi bien, voire mieux | — |
| Fait | Mon signal intérieur ne distingue pas les formes : face à un carré montré, un carré parfait obtient 0,90 et un cercle parfait 0,89 (la fermeture, identique pour les deux, domine) | diagnostic |
| Fait | Face à un carré, la note du maître reste ≈ 0 pendant tout l'apprentissage (max 0,28 en 8 000 essais) : « presque ça » ne donne aucune pente | diagnostic |
| Fait | Encourager chaque critère séparément (moyenne au lieu de produit) : toujours 0 carré | diagnostic, 3 rép. |
| Interprétation | Le carré est une **aiguille dans une botte de foin** de 7¹⁶ gestes : le cercle s'obtient en répétant un seul geste et attire tout l'apprentissage ; rien ne guide vers le carré tant qu'on n'en est pas déjà proche | forte |
| Interprétation | Le blocage est **moteur** (exploration), pas mnésique : aucune variante de mémoire ne peut aider si aucun essai n'approche jamais du carré | forte |

## 5. Surprises
1. Le gain de la Shadow en reconnaissance tient entièrement au **flou du rappel**, pas au renforcement
   par le rappel que je pensais central (votre intuition 4 n'est que partiellement soutenue).
2. Mon gabarit intérieur a **aggravé** l'effondrement sur le cercle : il récompense surtout « c'est
   fermé », et le cercle est la forme fermée la plus facile.
3. G2 (trace classique) est à 0 ici alors qu'il atteignait 45 % à l'étape 3 : le réglage de cette
   campagne (2 répétitions, mesure très grossière) a choisi de mauvais paramètres. Les comparaisons
   *entre* élèves dessinateurs de cette campagne sont donc fragiles ; la conclusion « 0 carré pour
   tous » ne l'est pas.

## 6. Prédictions vs réalité
| ID | Prédiction | Résultat | Statut |
|----|-----------|----------|--------|
| P25 (IA, 10 %) | Version forte : moins d'exemples que la rétropropagation | la rétropropagation + replay reste devant | non confirmée |
| P26 (IA, 65 %) | La Shadow comble ≥ 2/3 du retard en 1 passage | 80 % → 97 % (retard comblé en entier à 8 ex.) | **juste** |
| P27 (IA, 30 %) | Shadow > replay à mises à jour égales | oui (+3 pts à 8 ex.), mais grâce au seul rappel imparfait, qui marche aussi ajouté au replay | **juste, avec réserve** |
| P24 (IA, 60 %) | La fermeture divise par 2 les essais nécessaires | aucun effet positif | faux |
| P28 (IA, 60 %) | Gabarit + fermeture débloquent le carré (≥ 50 %) | 0 % | faux |
| P14 (IA, 50 %) | Le rappel reconstructif généralise mieux que le replay fidèle | oui, en reconnaissance (8 ex./forme) | juste (partiel) |

## 7. Décisions
D38 (correction hors-politique du replay), D39 (diagnostic : le dessin est bloqué par l'exploration
motrice ; prochaine étape à décider avec l'humain). Voir le journal.

## 8. Limites
- Reconnaissance : un seul encodeur (E3), un seul niveau (L2), tâche binaire où la récompense équivaut
  à l'étiquette.
- Dessin : réglage des hyperparamètres trop grossier ; signal intérieur mal conçu (ma responsabilité) ;
  10 répétitions.
- Le « rappel imparfait » n'a été testé qu'avec un masquage aléatoire de l'entrée, pas avec une vraie
  reconstruction par un réseau génératif.

## 9. Prochaine étape — trois voies, à choisir ensemble
1. **Montrer le geste en insistant** (votre intuition 2) : l'élève observe la main du maître, garde le
   geste en Shadow, puis s'exerce. Seule piste qui a déjà produit des carrés (imitation : 30 %).
2. **Des gestes qui se répètent** : donner à l'élève la capacité innée de répéter un petit geste
   (« un côté, je tourne, je recommence »). Le carré devient trouvable ; mais c'est un biais fort, à
   assumer et à documenter (cf. Sablé-Meyer et al. sur la géométrie comme langage de programmes).
3. **Refaire le gabarit intérieur** pour qu'il distingue vraiment les formes (comparer la *structure*
   des orientations, pas seulement la fermeture), puis retester.
Ma recommandation : 1 puis 3. La voie 2 rendrait le carré facile pour tout le monde et ne testerait
plus la Shadow.
