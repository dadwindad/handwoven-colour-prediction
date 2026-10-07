"""Observer sensitivity: the key physics and hybrid models with the 2-degree instead of the
10-degree reference white (the observer only enters the CIELAB <-> XYZ conversion of the
physics layer). All 625 fabrics, same splits and inner CV as the main comparison.

    .venv/bin/python -m weavecolor.evaluate_observer    -> results/observer_check.csv
"""
import csv

import numpy as np

from . import data as D
from .evaluate import RESULTS, SCHEMES, out_of_fold, subsets
from .models import HybridModel, StearnsNoechel, YuleNielsen

MODELS = {
    "S-N (Stearns-Noechel)": lambda w: StearnsNoechel(w),
    "Yule-Nielsen (n, w, k)": lambda w: YuleNielsen(w),
    "Hybrid: S-N + XGBoost": lambda w: HybridModel(StearnsNoechel, "XGBoost", w),
    "Hybrid: S-N + GP": lambda w: HybridModel(StearnsNoechel, "GP", w),
}


def main():
    ds = D.load()
    idx = subsets(ds)["all"]
    out = RESULTS / "observer_check.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scheme", "white", "mean_de00"])
        for name, make in MODELS.items():
            for scheme, splitter in SCHEMES.items():
                for white in ("D65_10", "D65_2"):
                    err = out_of_fold(ds, lambda: make(white), splitter, idx, scheme)
                    w.writerow([name, scheme, white, f"{np.nanmean(err[idx]):.4f}"])
                    f.flush()
                    print(name, scheme, white, round(float(np.nanmean(err[idx])), 3), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
