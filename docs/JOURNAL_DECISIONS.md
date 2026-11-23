# Journal des décisions

Chaque décision de conception ou d'analyse est tracée. « Proposée » = en attente de validation humaine.
Une décision n'est jamais modifiée en place : on ajoute une nouvelle entrée qui la remplace.

| ID | Date | Décision | Alternatives écartées | Raison | Décidé par | Statut |
|----|------|----------|-----------------------|--------|------------|--------|
| D01 | 2026-11-23 | Budgets comptés par classe {1,2,4,8,16,32} | Total {2,…,32} | « 2 exemples » au total = 1 par classe, ambigu | IA | Proposée |
| D02 | 2026-11-23 | Encodeur fixe = facteur contrôlé (pixels / aléatoire / orientations) | Pixels seuls | Sans invariances, tous les modèles risquent d'être au hasard à 4 exemples | IA | Proposée |
| D03 | 2026-11-23 | M1 = REINFORCE avec backprop | M1 local | Sépare « récompense vs étiquette » de « local vs backprop » | IA | Proposée |
| D04 | 2026-11-23 | Couche cachée locale par perturbation de nœud | AGREL (signal non strictement global) | Respecte « signal purement global » | IA | Proposée |
| D05 | 2026-11-23 | Régime principal : une passe ; secondaire : multi-passes à mises à jour égalisées | Époques libres | Évite de confondre exemples et mises à jour | IA | Proposée |
| D06 | 2026-11-23 | NumPy pur, backprop manuelle vérifiée | PyTorch | Instrumentation totale, code réseau commun | IA | Proposée |
| D07 | 2026-11-23 | Feedback retardé testé en version indexée ET non indexée | Version indexée seule | Évite de donner à la Shadow une information que la trace n'a pas | IA | Proposée |
| D08 | 2026-11-23 | Comptes rendus au format MODELE_CR.md + registre de prédictions | — | Garder l'humain informé ; séparer faits / interprétations | Humain + IA | Adoptée |
| D09 | 2026-11-23 | Paradigme principal = maître–élève : l'élève **refait** la forme, le maître récompense le dessin | Classification seule | Récompense scalaire sur une production riche ≠ étiquette ; donne à la Shadow un rôle de gabarit | Humain | Adoptée |
| D10 | 2026-11-23 | Espace moteur = stylo avec virages discrets, rendu par le même moteur que le générateur | Sortie pixel 32×32 | Sortie pixel inapprenable par règle locale avec récompense scalaire | IA | Proposée |
| D11 | 2026-11-23 | Maître principal catégoriel (juge la forme), maître copiste en contrôle | Copiste seul | Le copiste récompense la mémorisation de l'exemple | IA | Proposée |
| D12 | 2026-11-23 | Deux Shadows : perceptive (gabarit) et motrice (crédit du geste) | Une seule Shadow | Fonctions distinctes : contenu de référence vs attribution du crédit | IA | Proposée |
| D13 | 2026-11-23 | Démonstrations, essais et mises à jour comptés séparément | Un seul compteur | « Peu d'exemples » doit dire de quoi | IA | Proposée |
| D14 | 2026-11-23 | Classification M0–M5 conservée comme expérience secondaire | Abandon | Point de comparaison et validation du code | IA | Proposée |
| D15 | 2026-11-23 | Le maître juge la catégorie (« un carré ») ; la taille/position du modèle comptent faiblement | Copie fidèle | Mesurer le concept, pas la copie | IA (ex-Q4) | Adoptée, réversible |
| D16 | 2026-11-23 | Le maître montre seulement l'image ; montrer le geste = référence G0 | Geste montré | Cas le plus exigeant et le plus proche de « refaire ce qu'on a vu » | IA (ex-Q5) | Adoptée, réversible |
| D17 | 2026-11-23 | « Peu d'exemples » = peu de démonstrations ; les essais de l'élève sont comptés et rapportés à part | Compteur unique | Fidèle à l'objectif, sans masquer le coût en essais | IA (ex-Q6) | Adoptée, réversible |
| D18 | 2026-11-23 | L'auto-évaluation par gabarit (G5) est centrale | Maître seul | Cohérent avec l'image du maître et de l'élève | IA (ex-Q7) | Adoptée, réversible |
| D19 | 2026-11-23 | Règle de collaboration : aucune question technique à l'humain ; questions d'intuition créative uniquement | — | Demande explicite de l'humain | Humain | Adoptée |
| D20 | 2026-11-23 | Shadow sensorimotrice (visuel + plan de geste + proprioception) ; pratique mentale comptée à part | Shadow visuelle seule | Intuition 1 | Humain → IA | Adoptée |
| D21 | 2026-11-23 | Trois maîtres : muet / remontre / remontre en insistant ; comparaisons d'élèves à maître égal | Maître muet seul | Intuition 2 | Humain → IA | Adoptée |
| D22 | 2026-11-23 | Encouragement gradué (erreur de prédiction) + rafraîchissement de la Shadow après jugement | Pouce binaire seul | Intuition 3 | Humain → IA | Adoptée |
| D23 | 2026-11-23 | Shadow C redéfinie : rappel reconstructif qui renforce et alimente une mémoire de concept | Réactivation = replay | Intuition 4 ; distingue enfin Shadow C du replay | Humain → IA | Adoptée |
| D24 | 2026-11-23 | Boucle intérieure permanente : leçon / rêverie / sommeil ; sommeil en 2 phases (rejouer-ancrer, recombiner-alléger) | Apprentissage uniquement pendant les leçons | Intuition « penser quand on ne lui demande rien » | Humain → IA | Adoptée |
| D25 | 2026-11-23 | Expérience phare : gain de généralisation après sommeil, sans aucune nouvelle donnée, avec contrôles (veille calme, faux sommeil) | — | Test le plus net et le plus lisible de l'hypothèse | IA | Adoptée, réversible |
| D26 | 2026-11-23 | Concept intérieur = assemblée de sous-concepts entremêlés, émergeant de sensations primitives visuelles et motrices (aucun détecteur de forme programmé) | Prototype d'image ; détecteurs codés à la main | Intuition « concept sans détails, angles, coins, côtés égaux qui s'entremêlent » | Humain → IA | Adoptée |
| D27 | 2026-11-23 | Indice de clarté du concept (abstraction, netteté, stabilité), mesuré avant/après sommeil | — | Intuition « plus clair après une nuit » | Humain → IA | Adoptée |
| D28 | 2026-11-23 | Critère principal = « test de l'élève qui sait » : carrés de toutes tailles, sans modèle, acceptés, divers, défauts acceptables | Précision de classification | Définition humaine de « savoir ce qu'est un carré » | Humain | Adoptée |
| D29 | 2026-11-23 | Niveau de référence des expériences = L2 ; L1 conservé mais son indice de texture (traits alignés sur la grille) est documenté | Supprimer L1 | Les sondes montrent L2–L4 sans raccourci, L1 avec un indice trivial | IA | Adoptée, réversible |
| D30 | 2026-11-23 | Note graduée : exp(−½(écart/tolérance)²) par critère, moyenne géométrique ; 0,61 à la limite de tolérance | Note = 1 dans toute la tolérance | L'ancienne échelle donnait 0,99 à un gribouillis | IA | Adoptée |
| D31 | 2026-11-23 | Le maître juge le tracé (points), pas l'image | Juger les pixels | Juge fiable et identique pour formes du générateur et gestes de l'élève | IA | Adoptée, réversible |
| D32 | 2026-11-23 | Zone de placement des formes indépendante de la classe | Zone selon l'étendue de la forme | Fuite de position détectée par la sonde (57 %) | IA | Adoptée |
