"""Few-shot calibration of a new loom set-up (leave-one-loom-out, all 625 fabrics).

Each model is trained on four looms (grouped inner CV over those looms) and predicts the fifth.
A weaver then weaves and measures n fabrics on the new loom; the mean residual of those n
fabrics (a CIELAB offset) is added to every other prediction on that loom.

Three ways of choosing the calibration fabrics, to separate a loom-level effect from a
warp-yarn effect (the loom is nested in its five warp yarns):
  random      n fabrics drawn at random from the new loom; evaluated on its other fabrics
  cross-warp  n fabrics drawn from one warp yarn of the new loom; evaluated only on fabrics
              of the loom's other four warp yarns (an offset that transfers between warps
              reflects something the loom's fabrics share beyond their warp colour)
  placebo     the same n fabrics, but drawn from another loom's out-of-fold predictions;
              evaluated on the new loom (an offset that helps here is just a global bias of
              the model, not a loom effect)
  within-warp n fabrics drawn from one warp yarn; evaluated on that warp yarn's other fabrics
              (a warp-specific offset)
  swatch      for each warp yarn, its same-colour fabric (warp = weft) alone, n = 1; evaluated on
              that warp yarn's other fabrics (a practical per-yarn calibration)

    .venv/bin/python -m weavecolor.evaluate_calibration    -> results/calibration.csv
"""
import csv

import numpy as np

from . import data as D
from .colorlib import delta_e_2000
from .evaluate import RESULTS, inner_cv, split_leave_one_loom_out, subsets
from .models import HybridModel, MLModel, StearnsNoechel

MODELS = {
    "S-N (Stearns-Noechel)": lambda: StearnsNoechel(),
    "ML ล้วน: GP": lambda: MLModel("GP"),
    "ML ล้วน: MLP (BPNN)": lambda: MLModel("MLP (BPNN)"),
    "ML ล้วน: XGBoost": lambda: MLModel("XGBoost"),
    "Hybrid: S-N + GP": lambda: HybridModel(StearnsNoechel, "GP"),
    "Hybrid: S-N + MLP (BPNN)": lambda: HybridModel(StearnsNoechel, "MLP (BPNN)"),
    "Hybrid: S-N + XGBoost": lambda: HybridModel(StearnsNoechel, "XGBoost"),
}
N_SHOTS = (0, 1, 2, 3, 5, 10)
DRAWS = 200


def calibrated_error(pred, y, cal, ev, cal_pred=None, cal_y=None):
    """Mean CIEDE2000 on rows ev after adding the mean residual of rows cal (offset in CIELAB)."""
    if len(cal) == 0 and cal_pred is None:
        return delta_e_2000(pred[ev], y[ev]).mean()
    cp = pred[cal] if cal_pred is None else cal_pred
    cy = y[cal] if cal_y is None else cal_y
    offset = (cy - cp).mean(0)
    return delta_e_2000(pred[ev] + offset, y[ev]).mean()


def main():
    ds = D.load()
    idx = subsets(ds)["all"]
    rng = np.random.default_rng(0)
    out = RESULTS / "calibration.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "strategy", "n", "loom", "mean_de00"])
        for name, make in MODELS.items():
            # out-of-fold predictions for every loom
            preds, rows = {}, {}
            for tr, te in split_leave_one_loom_out(ds, idx):
                k = int(ds.loom[te][0])
                preds[k] = make().fit(ds, tr, cv=inner_cv(ds, tr, "leave-one-loom-out")).predict(ds, te)
                rows[k] = te
            for k, te in rows.items():
                p, y, warp = preds[k], ds.lab[te], ds.warp[te]
                all_rows = np.arange(len(te))
                for n in N_SHOTS:
                    # random fabrics from the new loom
                    errs = []
                    for _ in range(DRAWS if n else 1):
                        cal = rng.choice(len(te), n, replace=False)
                        errs.append(calibrated_error(p, y, cal, np.setdiff1d(all_rows, cal)))
                    w.writerow([name, "random", n, k, f"{np.mean(errs):.4f}"])
                    if n == 0 or n > 5:
                        continue
                    # fabrics of one warp yarn, evaluated on the loom's other warp yarns
                    errs = []
                    for wy in np.unique(warp):
                        same = np.flatnonzero(warp == wy)
                        other = np.flatnonzero(warp != wy)
                        for _ in range(DRAWS // 5):
                            cal = rng.choice(same, n, replace=False)
                            errs.append(calibrated_error(p, y, cal, other))
                    w.writerow([name, "cross-warp", n, k, f"{np.mean(errs):.4f}"])
                    # placebo: offset from n fabrics of another loom
                    errs = []
                    for j in rows:
                        if j == k:
                            continue
                        for _ in range(DRAWS // 4):
                            cal = rng.choice(len(rows[j]), n, replace=False)
                            errs.append(calibrated_error(p, y, np.array([], int), all_rows,
                                                         cal_pred=preds[j][cal], cal_y=ds.lab[rows[j]][cal]))
                    w.writerow([name, "placebo", n, k, f"{np.mean(errs):.4f}"])
                    # fabrics of one warp yarn, evaluated on that warp yarn's other fabrics
                    errs = []
                    for wy in np.unique(warp):
                        same = np.flatnonzero(warp == wy)
                        for _ in range(DRAWS // 5):
                            cal = rng.choice(same, n, replace=False)
                            errs.append(calibrated_error(p, y, cal, np.setdiff1d(same, cal)))
                    w.writerow([name, "within-warp", n, k, f"{np.mean(errs):.4f}"])
                    if n == 1:
                        # the warp yarn's same-colour fabric, a swatch a weaver can weave first
                        weft = ds.weft[te]
                        errs, base = [], []
                        for wy in np.unique(warp):
                            same = np.flatnonzero(warp == wy)
                            sw = np.flatnonzero((warp == wy) & (weft == wy))
                            rest = np.setdiff1d(same, sw)
                            errs.append(calibrated_error(p, y, sw, rest))
                            base.append(calibrated_error(p, y, np.array([], int), rest))
                        w.writerow([name, "swatch", 1, k, f"{np.mean(errs):.4f}"])
                        w.writerow([name, "swatch-baseline", 0, k, f"{np.mean(base):.4f}"])
            f.flush()
            print(name, "done", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
