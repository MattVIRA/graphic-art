"""Macros du fond (T16) : un curseur qui déplace plusieurs paramètres.

Chaque paramètre déclare dans le schéma son écart quand une macro passe de
0,5 (neutre) à 1 ; à 0 l'écart est opposé. Plusieurs macros sur un même
paramètre s'additionnent, puis le résultat est borné à la plage du
paramètre (ou bouclé pour une teinte / un angle). Les paramètres en loi
log reçoivent l'écart en log (un facteur), pour une réponse régulière.
"""

import math

from background.schema import PARAMS


def to_space(p, v: float) -> float:
    return math.log(max(v, p.lo)) if p.law == "log" else v


def from_space(p, x: float) -> float:
    return math.exp(x) if p.law == "log" else x


def bound(p, v: float) -> float:
    """Ramène une valeur dans la plage du paramètre."""
    if p.law == "angle":
        span = p.hi - p.lo
        return p.lo + (v - p.lo) % span
    v = min(max(v, p.lo), p.hi)
    if p.kind == "int" or p.options is not None:
        v = float(round(v))
    return v


def apply(values: dict) -> dict:
    """values : réglages du panneau (macros comprises) → réglages après
    macros, pour tous les paramètres du schéma."""
    out = {}
    for p in PARAMS:
        v = values.get(p.key, p.default)
        if p.macros:
            x = to_space(p, v)
            for name, delta in p.macros.items():
                m = values.get(f"bg_m_{name}", 0.5)
                x += (2.0 * m - 1.0) * delta
            v = from_space(p, x)
        out[p.key] = bound(p, v)
    return out
