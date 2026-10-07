"""Comparison experiment (plan section 7.2): every model, same splits, nested tuning.

Run from research/:
    .venv/bin/python -m weavecolor.evaluate                 # everything, resumable
    .venv/bin/python -m weavecolor.evaluate --only "Yule"   # models whose name contains "Yule"
    .venv/bin/python -m weavecolor.evaluate --report        # rebuild the report from saved results

Out-of-fold errors for every fabric are stored in results/oof_errors.csv, one model
at a time, so a long run can be stopped and resumed. The report compares every
model with the reference hybrid using a paired Wilcoxon signed-rank test
(Holm-corrected within each split scheme).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from . import data as D
from .colorlib import delta_e_2000
from .models import (LEARNERS, PHYSICS, CNNYarnPhotos, DimitrovskiGabrijelcic, HybridEnsemble, HybridModel,
                     MeanLab, MLModel,
                     StearnsNoechel, WarburtonOliver, YuleNielsen)

RESULTS = Path(__file__).resolve().parents[1] / "results"
OOF = RESULTS / "oof_errors.csv"
REFERENCE = "Hybrid: Yule-Nielsen + GP"


# ---------------------------------------------------------------- splits
def split_random(ds, idx, k=5, seed=0):
    perm = np.random.default_rng(seed).permutation(idx)
    for fold in np.array_split(perm, k):
        yield np.setdiff1d(idx, fold), fold


def split_leave_one_yarn_out(ds, idx):
    """Hold out every fabric that contains yarn c, as if c were a newly bought colour."""
    for c in ds.codes:
        has_c = (ds.warp[idx] == c) | (ds.weft[idx] == c)
        yield idx[~has_c], idx[has_c]


def split_leave_one_loom_out(ds, idx):
    """Hold out every fabric woven on loom k (warp group): a new warp set-up."""
    loom = ds.loom[idx]
    for k in sorted(set(loom)):
        yield idx[loom != k], idx[loom == k]


SCHEMES = {
    "random 5-fold": split_random,
    "leave-one-yarn-out": split_leave_one_yarn_out,
    "leave-one-loom-out": split_leave_one_loom_out,
}


def subsets(ds):
    return {
        "clean": np.flatnonzero(~ds.flag),        # flagged '*' rows removed (main result)
        "all": np.arange(len(ds.lab)),
    }


# ---------------------------------------------------------------- model registry
def model_factories():
    f = [lambda: MeanLab()]
    f += [lambda cls=cls: cls() for cls in PHYSICS]
    f += [lambda name=name: MLModel(name) for name in LEARNERS]
    for phys in (YuleNielsen, StearnsNoechel, WarburtonOliver, DimitrovskiGabrijelcic):
        for learner in ("GP", "XGBoost", "MLP (BPNN)"):
            f.append(lambda p=phys, l=learner: HybridModel(p, l))
    f.append(lambda: HybridEnsemble(StearnsNoechel, "XGBoost", 10))
    if CNNYarnPhotos.available():
        f += [lambda: CNNYarnPhotos("Ridge"), lambda: CNNYarnPhotos("GP")]
    else:
        print("CNN-A skipped: run `python -m weavecolor.cnn` first to build the photo features")
    return f


# ---------------------------------------------------------------- run
def out_of_fold(ds, make, splitter, idx):
    err = np.full(len(ds.lab), np.nan)
    for tr, te in splitter(ds, idx):
        if len(te):
            model = make().fit(ds, tr)
            err[te] = delta_e_2000(model.predict(ds, te), ds.lab[te])
    return err


def load_done():
    if not OOF.exists():
        return set()
    with open(OOF, newline="", encoding="utf-8") as f:
        return {(r["model"], r["subset"], r["scheme"]) for r in csv.DictReader(f)}


def run(only=None, subset_names=("clean", "all")):
    RESULTS.mkdir(exist_ok=True)
    ds = D.load()
    done = load_done()
    new_file = not OOF.exists()
    with open(OOF, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model", "family", "subset", "scheme", "row", "warp", "weft", "de00"])
        for make in model_factories():
            proto = make()
            if only and only.lower() not in proto.name.lower():
                continue
            for sname in subset_names:
                idx = subsets(ds)[sname]
                for scheme, splitter in SCHEMES.items():
                    if (proto.name, sname, scheme) in done:
                        continue
                    t0 = time.time()
                    err = out_of_fold(ds, make, splitter, idx)
                    for i in idx:
                        w.writerow([proto.name, proto.family, sname, scheme, i, ds.warp[i], ds.weft[i],
                                    f"{err[i]:.4f}"])
                    f.flush()
                    print(f"{proto.name:42} {sname:5} {scheme:20} mean {np.nanmean(err[idx]):5.2f}"
                          f"  ({time.time() - t0:5.0f}s)", flush=True)


# ---------------------------------------------------------------- report
def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def report():
    rows = list(csv.DictReader(open(OOF, newline="", encoding="utf-8")))
    errs, family = {}, {}
    for r in rows:
        errs.setdefault((r["model"], r["subset"], r["scheme"]), {})[int(r["row"])] = float(r["de00"])
        family[r["model"]] = r["family"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    fam_order = {"baseline": 0, "physics": 1, "ml": 2, "hybrid": 3, "cnn": 4}
    models.sort(key=lambda m: fam_order.get(family[m], 9))
    fam_th = {"baseline": "ฐาน", "physics": "ฟิสิกส์", "ml": "ML ล้วน", "hybrid": "Hybrid", "cnn": "CNN"}

    lines = [
        "# ผลเปรียบเทียบแบบจำลองทั้งหมด",
        "",
        "- ค่าคือ CIEDE2000 เฉลี่ยของการทำนายนอกชุดฝึก (out-of-fold) ยิ่งต่ำยิ่งดี; ในวงเล็บ = ร้อยละที่คลาดไม่เกิน 3",
        "- ทุกแบบจำลองใช้การแบ่งข้อมูลชุดเดียวกัน ML จูนพารามิเตอร์ด้วย inner 3-fold CV ภายในชุดฝึก (nested CV)",
        f"- p = Wilcoxon signed-rank เทียบกับ **{REFERENCE}** แบบจับคู่รายผ้า ปรับด้วย Holm ภายในแต่ละแบบการแบ่ง; "
        "▲ = ดีกว่าตัวอ้างอิงอย่างมีนัยสำคัญ (p < 0.05), ▼ = แย่กว่าอย่างมีนัยสำคัญ, – = ไม่ต่างอย่างมีนัยสำคัญ",
        "- แบบจำลองฟิสิกส์ใช้ค่า X/Xn, Y/Yn, Z/Zn แทนสเปกตรัม R(λ) (ข้อมูลมีแค่ L*a*b*) จึงเป็นการประมาณของสูตรต้นฉบับ",
        "- CNN-A ใช้ภาพถ่ายด้ายจากเล่มงานวิจัย (ไม่ได้ถ่ายในแสงควบคุม) ผ่าน ResNet-18 ที่ฝึกจาก ImageNet",
        "",
    ]
    for sname, title in (("clean", "ตัดแถว * (ผลหลัก)"), ("all", "ข้อมูลทั้งหมด 625 คู่")):
        present = [m for m in models if (m, sname, list(SCHEMES)[0]) in errs]
        if not present:
            continue
        stats = {}
        for scheme in SCHEMES:
            ref = errs.get((REFERENCE, sname, scheme))
            others = [m for m in present if m != REFERENCE and (m, sname, scheme) in errs]
            pv = []
            for m in others:
                e = errs[(m, sname, scheme)]
                if ref is None:
                    pv.append(np.nan)
                    continue
                keys = sorted(set(e) & set(ref))
                a, b = np.array([e[k] for k in keys]), np.array([ref[k] for k in keys])
                pv.append(wilcoxon(a, b).pvalue if np.any(a != b) else 1.0)
            adj = holm(np.nan_to_num(np.array(pv), nan=1.0)) if pv else []
            for m, p in zip(others, adj):
                stats[(m, scheme)] = p
        lines += [f"## {title}", "", "| กลุ่ม | แบบจำลอง | " + " | ".join(SCHEMES) + " |",
                  "|---|---|" + "---|" * len(SCHEMES)]
        for m in present:
            cells = []
            for scheme in SCHEMES:
                e = errs.get((m, sname, scheme))
                if e is None:
                    cells.append("")
                    continue
                v = np.array(list(e.values()))
                mark = ""
                if m != REFERENCE and (m, scheme) in stats and REFERENCE in present:
                    ref_mean = np.mean(list(errs[(REFERENCE, sname, scheme)].values()))
                    p = stats[(m, scheme)]
                    mark = " –" if p >= 0.05 else (" ▲" if v.mean() < ref_mean else " ▼")
                cells.append(f"{v.mean():.2f} ({np.mean(v <= 3) * 100:.0f}%){mark}")
            name = f"**{m}**" if m == REFERENCE else m
            lines.append(f"| {fam_th.get(family[m], family[m])} | {name} | " + " | ".join(cells) + " |")
        lines.append("")
    (RESULTS / "comparison_all_models.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RESULTS / "comparison_all_models.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run only models whose name contains this text")
    ap.add_argument("--subset", choices=["clean", "all", "both"], default="both")
    ap.add_argument("--report", action="store_true", help="only rebuild the report")
    a = ap.parse_args()
    if not a.report:
        run(a.only, ("clean", "all") if a.subset == "both" else (a.subset,))
    report()
