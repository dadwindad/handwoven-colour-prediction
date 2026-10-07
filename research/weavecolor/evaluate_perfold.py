"""Leave-one-yarn-out with every single prediction kept (a fabric of two different yarns is
predicted twice, once with each yarn unseen). The main comparison stores the mean of the two
errors per fabric; quantiles, coverage, shares within a threshold and uncertainty ranking are
computed from the single predictions, which are what a weaver actually faces.

    .venv/bin/python -m weavecolor.evaluate_perfold --subset all
    .venv/bin/python -m weavecolor.evaluate_perfold --subset clean
    .venv/bin/python -m weavecolor.evaluate_perfold --subset all --shuffled-inner --key-only

Writes results/loyo_folds_<subset>[_shuffled].csv: model, fold (held-out yarn), row, de00.
With --shuffled-inner the hyper-parameters are tuned with shuffled inner folds (the earlier
protocol) but both folds are still kept, which separates the two protocol changes.
"""
import argparse
import csv
import time

import numpy as np

from . import data as D
from .evaluate import RESULTS, model_factories, out_of_fold, split_leave_one_yarn_out, subsets

KEY = {"เฉลี่ย L*a*b*", "S-N (Stearns-Noechel)", "ML ล้วน: GP", "ML ล้วน: Polynomial Ridge", "ML ล้วน: SVR",
       "ML ล้วน: MLP (BPNN)", "ML ล้วน: XGBoost", "Hybrid: S-N + GP", "Hybrid: S-N + MLP (BPNN)",
       "Hybrid: S-N + XGBoost"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["all", "clean"], default="all")
    ap.add_argument("--shuffled-inner", action="store_true")
    ap.add_argument("--key-only", action="store_true")
    a = ap.parse_args()
    ds = D.load()
    idx = subsets(ds)[a.subset]
    out = RESULTS / f"loyo_folds_{a.subset}{'_shuffled' if a.shuffled_inner else ''}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "row", "de00"])
        for make in model_factories():
            name = make().name
            if a.key_only and name not in KEY:
                continue
            t0 = time.time()
            rows = []
            err = out_of_fold(ds, make, split_leave_one_yarn_out, idx, "leave-one-yarn-out", rows, a.shuffled_inner)
            for k, i, e in rows:
                w.writerow([name, ds.codes[k], i, f"{e:.4f}"])
            f.flush()
            print(f"{name:42} mean {np.nanmean(err[idx]):5.2f} ({len(rows)} predictions, {time.time() - t0:4.0f}s)",
                  flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
