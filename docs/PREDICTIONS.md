# Registre des prédictions

Avant chaque campagne, l'humain et l'IA écrivent **séparément** leurs prédictions.
Elles sont commitées avant les résultats, puis confrontées dans le compte rendu.
But : calibrer les deux formes de jugement (intuition humaine, raisonnement de l'IA),
et rendre visibles les surprises — qui sont les résultats les plus informatifs.

| ID | Date | Auteur | Prédiction | Confiance (%) | Expérience | Résultat | Correcte ? |
|----|------|--------|------------|---------------|------------|----------|------------|
| P01 | 2026-11-23 | IA | En feedback immédiat, M4 (Shadow B) ≈ M2 (trace) : différence de N90 non significative | 90 | few_shot | — | — |
| P02 | 2026-11-23 | IA | M0 (backprop) a le plus petit N90 parmi M0–M5, à encodeur égal | 75 | few_shot | — | — |
| P03 | 2026-11-23 | IA | Sur pixels bruts (E0), niveau L2, aucun modèle n'atteint 90 % avec 2 exemples/classe | 85 | few_shot | — | — |
| P04 | 2026-11-23 | IA | Avec feedback retardé indexé (D ≥ 4), M4/M5 dépassent M2 nettement ; l'écart disparaît largement sans indexation | 65 | delayed_reward | — | — |
| P05 | 2026-11-23 | IA | M5 ne bat pas un replay classique à nombre de mises à jour égal | 70 | delayed_reward | — | — |
| P06 | 2026-11-23 | Humain | <à compléter> | | | | |
