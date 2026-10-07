"""Superseded by `python -m weavecolor.evaluate_revision uncertainty` (all 625 fabrics, inner CV that
mirrors each scheme), which produced the results in the paper. Kept for reference.

Does the model's uncertainty tell us which predictions will be wrong?

Compares the bagged S-N + XGBoost ensemble with S-N + GP (whose uncertainty is the
GP posterior standard deviation). Uses the cleaned data and the same split schemes
as the main comparison.

    .venv/bin/python -m weavecolor.evaluate_uncertainty
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import data as D
from .colorlib import delta_e_2000
from .evaluate import RESULTS, SCHEMES
from .models import HybridEnsemble, HybridModel, StearnsNoechel

BIG = 3.0          # an error above this is "worth a test weave"
FLAG_RATE = 0.30   # the app would warn on the 30 % least certain predictions


def oof(ds, make, splitter, idx):
    err = np.full(len(ds.lab), np.nan)
    unc = np.full(len(ds.lab), np.nan)
    for tr, te in splitter(ds, idx):
        if len(te):
            pred, std = make().fit(ds, tr).predict(ds, te, return_std=True)
            err[te] = delta_e_2000(pred, ds.lab[te])
            unc[te] = np.sqrt((np.asarray(std) ** 2).sum(-1)) if np.ndim(std) == 2 else std
    return err[idx], unc[idx]


def metrics(err, unc):
    order = np.argsort(unc)
    half = len(err) // 2
    flagged = unc >= np.quantile(unc, 1 - FLAG_RATE)
    big = err > BIG
    return {
        "mean ΔE00": err.mean(),
        "Spearman ρ (ความไม่แน่นอน, error)": spearmanr(unc, err).statistic,
        f"AUROC จับ error > {BIG:g}": roc_auc_score(big, unc) if 0 < big.sum() < len(big) else np.nan,
        "ΔE00 ครึ่งที่มั่นใจ": err[order[:half]].mean(),
        "ΔE00 ครึ่งที่ไม่มั่นใจ": err[order[half:]].mean(),
        f"เตือน {FLAG_RATE:.0%} แรก จับ error ใหญ่ได้": (flagged & big).sum() / max(big.sum(), 1),
    }


def main():
    ds = D.load()
    idx = np.flatnonzero(~ds.flag)
    models = {
        "S-N + XGBoost ensemble×10": lambda: HybridEnsemble(StearnsNoechel, "XGBoost", 10),
        "S-N + GP": lambda: HybridModel(StearnsNoechel, "GP"),
    }
    lines = ["# ความไม่แน่นอนของการทำนาย: บอกได้ไหมว่าคู่ไหนจะผิดมาก", "",
             f"ข้อมูลตัดแถว * ออก · error ใหญ่ = ΔE00 > {BIG:g} · "
             "Spearman ρ และ AUROC ยิ่งสูงยิ่งดี (AUROC 0.5 = เดาสุ่ม, 1.0 = แยกได้สมบูรณ์)", ""]
    for scheme, splitter in SCHEMES.items():
        lines += [f"## {scheme}", ""]
        rows = {}
        for name, make in models.items():
            rows[name] = metrics(*oof(ds, make, splitter, idx))
            print(scheme, name, {k: round(float(v), 3) for k, v in rows[name].items()}, flush=True)
        keys = list(next(iter(rows.values())))
        lines += ["| ตัวชี้วัด | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
        for k in keys:
            fmt = (lambda v: f"{v:.0%}") if k.startswith("เตือน") else (lambda v: f"{v:.2f}")
            lines.append(f"| {k} | " + " | ".join(fmt(rows[n][k]) for n in rows) + " |")
        lines.append("")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "uncertainty.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RESULTS / "uncertainty.md")


if __name__ == "__main__":
    main()
