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
| P07 | 2026-11-23 | IA | Sans récompense graduée ni signal interne, G1–G3 n'obtiennent presque jamais de récompense au début (exploration) et apprennent très lentement | 75 | imitation | — | — |
| P08 | 2026-11-23 | IA | G5 (gabarit interne + maître) apprend plus vite que G3 (maître seul), en nombre d'essais | 60 | imitation | — | — |
| P09 | 2026-11-23 | IA | G5′ (gabarit seul, sans maître) progresse mais plafonne sous G5 | 55 | imitation | — | — |
| P10 | 2026-11-23 | IA | Le contrôle « gabarit brouillé » annule l'avantage de G5 | 70 | imitation | — | — |
| P11 | 2026-11-23 | IA | Le cercle sera appris avant le carré (virage constant plus facile que l'alternance droit/virage) | 60 | imitation | — | — |
| P12 | 2026-11-23 | IA | H7 : « remontrer en insistant » est le facteur au plus gros effet de toute l'étude | 70 | imitation | — | — |
| P13 | 2026-11-23 | IA | H8 : l'encouragement gradué rend l'apprentissage possible là où le pouce binaire échoue (début) | 75 | imitation | — | — |
| P14 | 2026-11-23 | IA | H9 : le rappel reconstructif généralise mieux que le replay fidèle hors distribution | 50 | imitation | — | — |
| P15 | 2026-11-23 | IA | H6 : la composante motrice de la Shadow apporte peu tant que l'espace moteur est simple | 55 | imitation | — | — |
| P16 | 2026-11-23 | IA | Gain hors-ligne positif après sommeil sur formes nouvelles, supérieur au « faux sommeil » | 45 | sommeil | — | — |
| P17 | 2026-11-23 | IA | Sans allègement (homéostasie), le sommeil ancre surtout les erreurs : gain nul ou négatif | 55 | sommeil | — | — |
| P18 | 2026-11-23 | IA | Le sommeil aide plus quand les démonstrations étaient peu nombreuses (1–2 par forme) que nombreuses | 55 | sommeil | — | — |
