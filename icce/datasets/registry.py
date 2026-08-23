"""Name -> loader registry so eval scripts take `--dataset levir_cd`."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from . import levir_cc, levir_cd, s2looking, whu_cd
from .common import ChangePair

LOADERS: Dict[str, Callable[..., List[ChangePair]]] = {
    "levir_cd": levir_cd.load,
    "levir_cc": levir_cc.load,
    "whu_cd": whu_cd.load,
    "s2looking": s2looking.load,
}

GSD: Dict[str, float] = {
    "levir_cd": levir_cd.GSD_M,
    "levir_cc": levir_cd.GSD_M,   # LEVIR-CC crops inherit LEVIR-CD resolution
    "whu_cd": whu_cd.GSD_M,
    "s2looking": s2looking.GSD_M,
}


def load(name: str, split: str = "test", limit: Optional[int] = None, **kw) -> List[ChangePair]:
    key = name.strip().lower().replace("-", "_")
    if key not in LOADERS:
        raise KeyError(f"unknown dataset '{name}'. available: {sorted(LOADERS)}")
    return LOADERS[key](split=split, limit=limit, **kw)


def available() -> List[str]:
    return sorted(LOADERS)
