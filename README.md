# Shadow Learning Lab

Laboratoire expérimental : un élève artificiel peut-il apprendre le concept de carré et de cercle
avec très peu de démonstrations, grâce à une mémoire temporaire de ses états (Shadow Memory),
une récompense globale du maître et une plasticité locale ?

- Conception : `docs/CONCEPTION.md`, `docs/PARADIGME_IMITATION.md`
- Comptes rendus : `docs/comptes-rendus/` (dernier en date en premier à lire)
- Décisions : `docs/JOURNAL_DECISIONS.md` — Paris : `docs/PREDICTIONS.md`

## Utilisation

```bash
pip install -e ".[dev]"            # numpy, matplotlib, pyyaml, pytest
python -m pytest -q                # tests
PYTHONPATH=src python experiments/step2_world.py   # étape 2 : validation du monde (~1 min)
```

Résultats : `results/<étape>/summary.json` (avec graine et commit git), figures dans `docs/figures/`.

## Structure

```
src/sll/
  core/rng.py               graines indépendantes par rôle (train/test/...)
  data/render.py            moteur de rendu partagé (maître ET élève)
  data/levels.py            niveaux de difficulté L0–L4, hors distribution
  data/shapes_generator.py  cercles et carrés paramétrés
  data/datasets.py          tirages disjoints, épisodes few-shot, contrôle de fuite
  data/encoders.py          encodeurs fixes : pixels, aléatoire, orientations
  world/pen.py              le stylo de l'élève (virages), proprioception
  world/teacher.py          maître catégoriel (pouce + « presque ça ») et maître copiste
  analysis/                 sondes anti-raccourcis, statistiques, figures
experiments/                scripts d'expériences reproductibles
tests/                      tests
```
