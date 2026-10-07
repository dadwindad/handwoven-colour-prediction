"""Extra runs requested in the Q2 revision (all 625 fabrics, the primary analysis).

    .venv/bin/python -m weavecolor.evaluate_revision repeated      # random 5-fold repeated with 10 seeds
    .venv/bin/python -m weavecolor.evaluate_revision uncertainty   # per-fabric error and uncertainty
    .venv/bin/python -m weavecolor.evaluate_revision params        # selected hyper-parameters per fold

Outputs go to results/: repeated_cv.csv, uncertainty_oof.csv, selected_params.csv.
The analysis of these files (and of oof_errors.csv) is in scripts/make_paper_figures.py.
"""
import csv
import sys
import time

import numpy as np

from . import data as D
from .colorlib import delta_e_2000
from .evaluate import RESULTS, SCHEMES, inner_cv, split_random, subsets
from .models import (LEARNERS, HybridEnsemble, HybridModel, MeanLab, MLModel, StearnsNoechel,
                     YuleNielsen)

KEY_MODELS = {
    "เฉลี่ย L*a*b*": lambda: MeanLab(),
    "S-N (Stearns-Noechel)": lambda: StearnsNoechel(),
    "ML ล้วน: GP": lambda: MLModel("GP"),
    "ML ล้วน: MLP (BPNN)": lambda: MLModel("MLP (BPNN)"),
    "ML ล้วน: XGBoost": lambda: MLModel("XGBoost"),
    "Hybrid: S-N + GP": lambda: HybridModel(StearnsNoechel, "GP"),
    "Hybrid: S-N + MLP (BPNN)": lambda: HybridModel(StearnsNoechel, "MLP (BPNN)"),
    "Hybrid: S-N + XGBoost": lambda: HybridModel(StearnsNoechel, "XGBoost"),
    "Hybrid: S-N + XGBoost ensemble×10": lambda: HybridEnsemble(StearnsNoechel, "XGBoost", 10),
}


def repeated(seeds=range(10)):
    ds = D.load()
    idx = subsets(ds)["all"]
    out = RESULTS / "repeated_cv.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "mean_de00", "share_le3"])
        for seed in seeds:
            for name, make in KEY_MODELS.items():
                t0 = time.time()
                err = np.full(len(ds.lab), np.nan)
                for tr, te in split_random(ds, idx, seed=seed):
                    err[te] = delta_e_2000(make().fit(ds, tr).predict(ds, te), ds.lab[te])
                e = err[idx]
                w.writerow([name, seed, f"{e.mean():.4f}", f"{np.mean(e <= 3):.4f}"])
                f.flush()
                print(f"seed {seed} {name:36} {e.mean():5.2f} ({time.time() - t0:4.0f}s)", flush=True)
    print("wrote", out)


def uncertainty():
    ds = D.load()
    idx = subsets(ds)["all"]
    models = {
        "S-N + XGBoost, bagged": lambda: HybridEnsemble(StearnsNoechel, "XGBoost", 10),
        "S-N + GP": lambda: HybridModel(StearnsNoechel, "GP"),
    }
    out = RESULTS / "uncertainty_oof.csv"
    single = open(RESULTS / "uncertainty_folds.csv", "w", newline="", encoding="utf-8")
    ws = csv.writer(single)     # every single prediction (leave-one-yarn-out predicts most fabrics twice)
    ws.writerow(["model", "scheme", "fold", "row", "de00", "unc"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scheme", "row", "de00", "unc"])
        for scheme, splitter in SCHEMES.items():
            for name, make in models.items():
                # a fabric held out in two leave-one-yarn-out folds gets the mean of both
                s_err, s_unc, n = np.zeros(len(ds.lab)), np.zeros(len(ds.lab)), np.zeros(len(ds.lab))
                for k, (tr, te) in enumerate(splitter(ds, idx)):
                    pred, std = make().fit(ds, tr, cv=inner_cv(ds, tr, scheme)).predict(ds, te, return_std=True)
                    e, u = delta_e_2000(pred, ds.lab[te]), np.sqrt((np.asarray(std) ** 2).sum(-1))
                    s_err[te] += e
                    s_unc[te] += u
                    n[te] += 1
                    for i, ei, ui in zip(te, e, u):
                        ws.writerow([name, scheme, k, i, f"{ei:.4f}", f"{ui:.5f}"])
                err = np.where(n > 0, s_err / np.maximum(n, 1), np.nan)
                unc = np.where(n > 0, s_unc / np.maximum(n, 1), np.nan)
                for i in idx:
                    w.writerow([name, scheme, i, f"{err[i]:.4f}", f"{unc[i]:.5f}"])
                f.flush()
                single.flush()
                print(scheme, name, f"{np.nanmean(err[idx]):.2f}", flush=True)
    single.close()
    print("wrote", out)


def params():
    """Hyper-parameters chosen by the inner CV in every outer fold, and the fitted physics parameters."""
    ds = D.load()
    idx = subsets(ds)["all"]
    out = RESULTS / "selected_params.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "scheme", "fold", "n_train", "params"])
        for scheme, splitter in SCHEMES.items():
            for fold, (tr, te) in enumerate(splitter(ds, idx)):
                sn = StearnsNoechel().fit(ds, tr)
                w.writerow(["S-N", scheme, fold, len(tr), f"w={sn.w:.3f}; b={np.exp(sn.extra[0]):.4f}"])
                for learner in ("XGBoost", "MLP (BPNN)", "GP"):
                    for model in (MLModel(learner), HybridModel(StearnsNoechel, learner)):
                        est = model.fit(ds, tr, cv=inner_cv(ds, tr, scheme)).est
                        if hasattr(est, "best_params_"):
                            p = "; ".join(f"{k.split('__')[-1]}={v}" for k, v in est.best_params_.items())
                        else:
                            p = str(est.kernel_)
                        w.writerow([model.name, scheme, fold, len(tr), p])
                f.flush()
                print(scheme, fold, flush=True)
        yn = YuleNielsen().fit(ds, idx)
        sn = StearnsNoechel().fit(ds, idx)
        w.writerow(["S-N", "all 625", "-", len(idx), f"w={sn.w:.3f}; b={np.exp(sn.extra[0]):.4f}"])
        w.writerow(["Yule-Nielsen", "all 625", "-", len(idx),
                    "; ".join(f"{k}={v:.3f}" for k, v in yn.params.items())])
    print("wrote", out)
    for name, (est, grid) in LEARNERS.items():
        print(name, grid)


if __name__ == "__main__":
    {"repeated": repeated, "uncertainty": uncertainty, "params": params}[sys.argv[1]]()
