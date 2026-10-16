"""EOTF du rendu certifié face à decimal (50 chiffres) : erreurs observées au plus la moitié des
bornes démontrées (certified::eotf_relative_error), chemin double (u = 2^-53) et double-double
(u = 2^-100). Usage : certified_eotf.py <fel_certified_dump>
"""
from __future__ import annotations

import math
import subprocess
import sys
from decimal import Decimal, getcontext

getcontext().prec = 50
M1 = Decimal(2610) / 16384
M2 = Decimal(2523) / 32
C1 = Decimal(3424) / 4096
C2 = Decimal(2413) / 128
C3 = Decimal(2392) / 128


def eotf(x: Decimal) -> Decimal:
    p = (x.ln() / M2).exp()
    return Decimal(0) if p <= C1 else (((p - C1) / (C2 - C3 * p)).ln() / M1).exp()


def bound(value: float, kappa: float, unit: float) -> float:
    exponent = math.frexp(value)[1]
    return (800 * kappa + 60 + 4.2 * (abs(exponent) + 1)) * unit


def main() -> None:
    lines = subprocess.run([sys.argv[1]], check=True, capture_output=True, text=True).stdout.split("\n")
    worst = {"double": 0.0, "double-double": 0.0}
    for line in filter(None, lines):
        x, fast, kappa_fast, high, low, kappa = (float.fromhex(v) for v in line.split())
        exact = eotf(Decimal(x))
        if exact == 0:
            continue
        for name, value, k, unit in (("double", Decimal(fast), kappa_fast, 2.0**-53),
                                     ("double-double", Decimal(high) + Decimal(low), kappa, 2.0**-100)):
            if not math.isfinite(k):
                continue
            ratio = float(abs(value - exact) / exact) / bound(float(exact), k, unit)
            worst[name] = max(worst[name], ratio)
            if ratio > 0.5:
                sys.exit(f"EOTF {name} : x = {x!r}, erreur {ratio:.3f} × la borne (κ = {k})")
    print("EOTF certifiée : erreur maximale / borne", {k: round(v, 4) for k, v in worst.items()})


if __name__ == "__main__":
    main()
