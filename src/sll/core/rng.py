"""Gestion des graines aléatoires.

Principe : une graine maîtresse + un *rôle* nommé (« train », « test », « teacher »…) donnent un
générateur indépendant et reproductible. Deux rôles différents ne partagent jamais de flux
aléatoire : c'est la première protection contre les fuites train/test.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _role_key(role: str) -> int:
    return int.from_bytes(hashlib.sha256(role.encode("utf-8")).digest()[:8], "little")


def make_rng(master_seed: int, *roles: str | int) -> np.random.Generator:
    """Générateur déterministe pour (graine maîtresse, rôle1, rôle2, ...)."""
    keys = [int(master_seed)] + [_role_key(str(r)) for r in roles]
    return np.random.default_rng(np.random.SeedSequence(keys))
