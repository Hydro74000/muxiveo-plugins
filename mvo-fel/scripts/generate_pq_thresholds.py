"""Seuils d'arrondi PQ dans le domaine linéaire : T(k+½) = EOTF((k+½)/65535), k = 0..65534.

Chaque seuil est calculé à 60 chiffres significatifs (module decimal, sans dépendance), puis
arrondi vers le bas en double : T_bas ≤ T exact < suivant(T_bas). Un code RGB16 k est certain
pour une valeur linéaire encadrée [R-e, R+e] si T_bas(k-½) < R-e et R+e ≤ T_bas(k+½).
Partie basse (reprises en double-double) : T_bas_lo = T - T_bas arrondi au plus proche,
|T - (T_bas + T_bas_lo)| ≤ 2^-105·T.

Sortie : 65 535 parties hautes puis 65 535 parties basses, doubles petit-boutistes (1 Mio),
écrites dans data/pq_thresholds.bin.
Usage : generate_pq_thresholds.py [--check]   (--check : régénère et compare au fichier)
"""
from __future__ import annotations

import argparse
import hashlib
import math
import struct
import sys
from decimal import Decimal, getcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data/pq_thresholds.bin"
COUNT = 65535

getcontext().prec = 60
M1 = Decimal(2610) / Decimal(16384)
M2 = Decimal(2523) / Decimal(32)
C1 = Decimal(3424) / Decimal(4096)
C2 = Decimal(2413) / Decimal(128)
C3 = Decimal(2392) / Decimal(128)


def eotf(signal: Decimal) -> Decimal:
    """PQ EOTF exacte (même formule que pq_eotf), signal dans ]c1^m2, 1]."""
    p = (signal.ln() / M2).exp()
    return (((p - C1) / (C2 - C3 * p)).ln() / M1).exp()


def round_down(value: Decimal) -> tuple[float, float]:
    """Plus grand double inférieur ou égal à la valeur exacte, et reste au plus proche."""
    result = float(value)
    if Decimal(result) > value:
        result = math.nextafter(result, -math.inf)
    upper = math.nextafter(result, math.inf)
    if not Decimal(result) <= value < Decimal(upper):
        raise ArithmeticError(f"Encadrement impossible pour {value}")
    # Marge : la valeur exacte ne doit pas coller à un double (précision de calcul 60 chiffres).
    if min(value - Decimal(result), Decimal(upper) - value) < value * Decimal("1e-40"):
        raise ArithmeticError(f"Seuil trop proche d'un double : {value}")
    low = float(value - Decimal(result))
    if not 0 < low <= upper - result:
        raise ArithmeticError(f"Partie basse invalide pour {value}")
    return result, low


def thresholds() -> bytes:
    pairs = [round_down(eotf((Decimal(k) + Decimal("0.5")) / Decimal(65535))) for k in range(COUNT)]
    return struct.pack(f"<{2*COUNT}d", *(high for high, _ in pairs), *(low for _, low in pairs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Comparer au fichier versionné")
    args = parser.parse_args()
    table = thresholds()
    values = struct.unpack(f"<{COUNT}d", table[:8*COUNT])
    if any(not b > a for a, b in zip(values, values[1:])) or not (0 < values[0] and values[-1] < 1):
        raise ArithmeticError("Seuils non strictement croissants ou hors de ]0, 1[")
    if args.check:
        if TABLE.read_bytes() != table:
            sys.exit("Table des seuils PQ différente du calcul")
        print("Table des seuils PQ conforme :", hashlib.sha256(table).hexdigest())
        return
    TABLE.parent.mkdir(exist_ok=True)
    TABLE.write_bytes(table)
    print(TABLE, hashlib.sha256(table).hexdigest())


if __name__ == "__main__":
    main()
