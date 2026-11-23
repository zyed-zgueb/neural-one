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
