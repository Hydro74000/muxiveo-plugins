"""Vérifie que include/trt_plugin_abi.h est identique à la référence de muxiveo-rife (sdk.json > abi).

Usage : python check_abi.py [fichier de référence local]
Sans argument, la référence est lue sur GitHub (dépôt et révision de sdk.json).
"""
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    abi = json.loads((HERE / "sdk.json").read_text(encoding="utf-8"))["abi"]
    if len(sys.argv) > 1:
        reference = Path(sys.argv[1]).read_text(encoding="utf-8")
        origin = sys.argv[1]
    else:
        origin = f"https://raw.githubusercontent.com/{abi['repository']}/{abi['ref']}/{abi['path']}"
        with urllib.request.urlopen(origin, timeout=60) as resp:  # nosec B310 — GitHub
            reference = resp.read().decode("utf-8")
    local = (HERE / "include" / "trt_plugin_abi.h").read_text(encoding="utf-8")
    if local != reference:
        raise SystemExit(f"include/trt_plugin_abi.h diffère de la référence {origin} : recopier l'en-tête")
    print(f"ABI identique à {origin}")


if __name__ == "__main__":
    main()
