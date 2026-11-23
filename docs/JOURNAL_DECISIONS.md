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
