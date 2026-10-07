"""Figures, tables and summary numbers for the manuscript (paper/).

Primary analysis: all 625 fabrics (no fabric is excluded on the basis of its measured colour).
The screened 596-fabric subset is a sensitivity analysis.

Reads results/oof_errors.csv (and, when present, repeated_cv.csv, uncertainty_oof.csv and
selected_params.csv from weavecolor.evaluate_revision) and writes to paper/figures/:
    fig_design.pdf      the 25 x 25 factorial, fabric colours grouped by loom
    fig_schemes.pdf     mean ΔE00 per split scheme, key models
    fig_looms.pdf       leave-one-loom-out ΔE00 per held-out loom
    fig_distance.pdf    error versus distance from the training inputs
    table_*.tex         table bodies included by manuscript.tex
    numbers.txt         every number quoted in the text

Inference is yarn-cluster aware: the 25 yarn colours are resampled with replacement and each
fabric is weighted by the product of the multiplicities of its warp and weft yarn (pigeonhole
bootstrap; Owen, 2007). Fabric-level Wilcoxon tests are reported only as descriptive.

Run from research/:
    .venv/bin/python scripts/make_paper_figures.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weavecolor import data as D  # noqa: E402
from weavecolor.colorlib import lab_to_xyz  # noqa: E402
from weavecolor.evaluate import SCHEMES as SPLITTERS, holm, subsets  # noqa: E402
from weavecolor.models import StearnsNoechel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "research" / "results"
OOF = RES / "oof_errors.csv"
OUT = ROOT / "paper" / "figures"
PRIMARY = "all"
B = 10000            # bootstrap resamples

SCHEMES = {"random 5-fold": "Random 5-fold", "leave-one-yarn-out": "Leave-one-yarn-out",
           "leave-one-loom-out": "Leave-one-loom-out"}
SCHEME_SHORT = {"random 5-fold": "Random", "leave-one-yarn-out": "Unseen yarn",
                "leave-one-loom-out": "Unseen loom"}
# internal model name -> label used in the paper
MODELS = {
    "เฉลี่ย L*a*b*": "Mean CIELAB",
    "S-N (Stearns-Noechel)": "S-N (physics)",
    "ML ล้วน: GP": "GP (pure ML)",
    "ML ล้วน: MLP (BPNN)": "MLP (pure ML)",
    "ML ล้วน: XGBoost": "XGBoost (pure ML)",
    "Hybrid: S-N + GP": "S-N + GP (hybrid)",
    "Hybrid: S-N + MLP (BPNN)": "S-N + MLP (hybrid)",
    "Hybrid: S-N + XGBoost": "S-N + XGBoost (hybrid)",
    "Hybrid: S-N + XGBoost ensemble×10": "S-N + XGBoost, bagged (hybrid)",
}
LEARNERS = ("GP", "MLP (BPNN)", "XGBoost")
REF = "Hybrid: S-N + XGBoost"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, validated for CVD
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

FAMILY_EN = {"baseline": "Baseline", "physics": "Physics", "ml": "Pure ML", "hybrid": "Hybrid",
             "cnn": "Image"}
NAME_EN = {
    "เฉลี่ย L*a*b*": "Mean of yarn CIELAB",
    "D-G (L*a*b* ถ่วงน้ำหนัก)": "D-G (weighted CIELAB)",
    "modified S-N (ประมาณ λ)": "Modified S-N",
    "CNN-A: ResNet-18 ภาพด้าย + Ridge": "ResNet-18 photo + Ridge",
    "CNN-A: ResNet-18 ภาพด้าย + GP": "ResNet-18 photo + GP",
}


def name_en(m):
    return NAME_EN.get(m, m.replace("ML ล้วน: ", "").replace("Hybrid: ", "").replace(" (BPNN)", "")
                       .replace(" ensemble×10", ", bagged"))


# ---------------------------------------------------------------- data
DS = D.load()
N = len(DS.lab)


def load_errors():
    """errs[subset][(model, scheme)] -> length-625 array (nan outside the subset); family; model order."""
    errs, family, order = defaultdict(dict), {}, []
    with open(OOF, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["scheme"])
            e = errs[r["subset"]].setdefault(key, np.full(N, np.nan))
            e[int(r["row"])] = float(r["de00"])
            family[r["model"]] = r["family"]
            if r["model"] not in order:
                order.append(r["model"])
    rank = {"baseline": 0, "physics": 1, "ml": 2, "hybrid": 3, "cnn": 4}
    order.sort(key=lambda m: rank[family[m]])
    return errs, family, order


def folds(idx):
    """For every scheme: fold label per fabric (the fold whose model made its out-of-fold prediction)
    and the training rows of each fold. Reproduces the overwrite order of evaluate.out_of_fold."""
    out = {}
    for scheme, splitter in SPLITTERS.items():
        lab, train = np.full(N, -1), []
        for k, (tr, te) in enumerate(splitter(DS, idx)):
            lab[te] = k
            train.append(tr)
        out[scheme] = (lab, train)
    return out


def yarn_weights(idx, seed=0):
    """(B, len(idx)) pigeonhole-bootstrap weights: product of yarn multiplicities of warp and weft."""
    pos = {c: i for i, c in enumerate(DS.codes)}
    wi = np.array([pos[c] for c in DS.warp[idx]])
    fi = np.array([pos[c] for c in DS.weft[idx]])
    cnt = np.random.default_rng(seed).multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=B)
    return (cnt[:, wi] * cnt[:, fi]).astype(float)


def boot_means(W, e):
    """Bootstrap distribution of the weighted mean; resamples with zero total weight are dropped."""
    s = W.sum(1)
    ok = s > 0
    return (W[ok] @ e) / s[ok]


def boot_ci(W, e):
    m = boot_means(W, e)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_p(W, d):
    """Two-sided bootstrap p for mean(d) = 0 (percentile method, floored at 1/B)."""
    m = boot_means(W, d)
    return max(2 * min(np.mean(m <= 0), np.mean(m >= 0)), 1 / len(m))


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    mant, ex = f"{p:.0e}".split("e")
    return f"${mant}\\times10^{{{int(ex)}}}$"


# ---------------------------------------------------------------- style
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def lab_to_srgb(lab):
    xyz = lab_to_xyz(np.atleast_2d(lab), "D65_2") / 100.0
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(xyz @ M.T, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- figures
def fig_design():
    """25 x 25 factorial: rows = warp yarn grouped by loom, columns = weft yarn; cell = fabric colour."""
    order = [c for L in range(1, 6) for c in sorted(D.LOOM_OF_WARP) if D.LOOM_OF_WARP[c] == L]
    pos = {c: i for i, c in enumerate(order)}
    img = np.ones((25, 25, 3))
    for i in range(N):
        img[pos[DS.warp[i]], pos[DS.weft[i]]] = lab_to_srgb(DS.lab[i])[0]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.imshow(img, interpolation="nearest", extent=(0, 25, 25, 0))
    for i in np.flatnonzero(DS.flag):
        ax.plot(pos[DS.weft[i]] + 0.5, pos[DS.warp[i]] + 0.5, marker="x", ms=4, mew=0.9, color="white")
        ax.plot(pos[DS.weft[i]] + 0.5, pos[DS.warp[i]] + 0.5, marker="x", ms=3, mew=0.5, color=INK)
    for L in range(1, 5):
        ax.axhline(5 * L, color="white", lw=2.0)
    yarn = lab_to_srgb(DS.yarn_lab_array(order))
    for j, c in enumerate(yarn):
        ax.add_patch(plt.Rectangle((j, -1.3), 1, 1, color=c, clip_on=False))
        ax.add_patch(plt.Rectangle((-1.3, j), 1, 1, color=c, clip_on=False))
    for L in range(1, 6):
        ax.text(-1.8, 5 * L - 2.5, f"L{L}", ha="right", va="center", fontsize=8, color=INK)
    ax.set_xticks(np.arange(25) + 0.5, order, fontsize=5.5, rotation=90, color=MUTED)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=14)
    ax.set_xlabel("Weft yarn colour (code)", fontsize=8, color=INK)
    ax.set_ylabel("Warp yarn colour, grouped by loom set-up", fontsize=8, color=INK, labelpad=26)
    ax.set_xlim(-1.4, 25)
    ax.set_ylim(25, -1.4)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_design.pdf")
    fig.savefig(OUT / "fig_design.png", dpi=300)


def fig_schemes(errs):
    names = [m for m in MODELS if "ensemble" not in m]
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for j, (scheme, label) in enumerate(SCHEMES.items()):
        means = [np.nanmean(errs[(m, scheme)]) for m in names]
        bars = ax.bar(x + (j - 1) * width, means, width * 0.92, color=COLORS[j], label=label)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6, color=INK)
    style(ax)
    ax.set_xticks(x, [MODELS[m] for m in names], rotation=25, ha="right")
    ax.set_ylabel("Mean CIEDE2000 (out-of-fold)", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.set_ylim(0, 10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_schemes.pdf")
    fig.savefig(OUT / "fig_schemes.png", dpi=300)


def fig_looms(errs):
    loom = DS.loom
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    looms = np.arange(1, 6)
    for ax, learner in zip(axes, LEARNERS):
        pure, hyb = f"ML ล้วน: {learner}", f"Hybrid: S-N + {learner}"
        for k, (m, c) in enumerate(((pure, COLORS[1]), (hyb, COLORS[0]))):
            e = errs[(m, "leave-one-loom-out")]
            means = [np.nanmean(e[loom == L]) for L in looms]
            ax.bar(looms + (k - 0.5) * 0.38, means, 0.36, color=c,
                   label="pure ML" if k == 0 else "hybrid (S-N + learner)")
        style(ax)
        ax.set_title(MODELS[pure].split(" (")[0], fontsize=9, color=INK)
        ax.set_xticks(looms, [f"L{L}" for L in looms])
        ax.set_xlabel("Held-out loom", fontsize=8, color=INK)
    axes[0].set_ylabel("Mean CIEDE2000", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_looms.pdf")
    fig.savefig(OUT / "fig_looms.png", dpi=300)


def input_distance(idx, fl):
    """Distance (CIELAB units, 6-D warp+weft) from each fabric's input to the nearest training input."""
    X = np.hstack([DS.warp_lab, DS.weft_lab])
    out = {}
    for scheme, (lab, train) in fl.items():
        d = np.full(N, np.nan)
        for k, tr in enumerate(train):
            te = np.flatnonzero(lab == k)
            if len(te):
                d[te] = np.sqrt(((X[te, None, :] - X[None, tr, :]) ** 2).sum(-1)).min(1)
        out[scheme] = d
    return out


def fig_distance(errs, dist, idx):
    """Mean error by distance quartile, pooled over the two shift schemes."""
    shift = ("leave-one-yarn-out", "leave-one-loom-out")
    d = np.concatenate([dist[s][idx] for s in shift])
    edges = np.quantile(d, [0, 0.25, 0.5, 0.75, 1])
    q = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, 3)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    rows = []
    for ax, learner in zip(axes, LEARNERS):
        for k, (m, c, lbl) in enumerate(((f"ML ล้วน: {learner}", COLORS[1], "pure ML"),
                                          (f"Hybrid: S-N + {learner}", COLORS[0], "hybrid (S-N + learner)"))):
            e = np.concatenate([errs[(m, s)][idx] for s in shift])
            means = [e[q == j].mean() for j in range(4)]
            rows.append((m, means, spearmanr(d, e).statistic))
            ax.plot(np.arange(4), means, color=c, lw=2, marker="o", ms=5, label=lbl)
        style(ax)
        ax.set_title(learner.replace(" (BPNN)", ""), fontsize=9, color=INK)
        ax.set_xticks(np.arange(4), ["Q1\nnear", "Q2", "Q3", "Q4\nfar"])
        ax.set_xlabel("Distance to nearest training input", fontsize=8, color=INK)
    axes[0].set_ylabel("Mean CIEDE2000", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_distance.pdf")
    fig.savefig(OUT / "fig_distance.png", dpi=300)
    return edges, rows


# ---------------------------------------------------------------- tables
def table_all(errs, family, order, idx, subset):
    """LaTeX rows: mean (share <= 3); markers vs the S-N + XGBoost hybrid by the yarn-cluster
    bootstrap test, Holm-corrected within each scheme."""
    W = yarn_weights(idx)
    marks = {}
    for scheme in SCHEMES:
        ref = errs[(REF, scheme)][idx]
        others = [m for m in order if m != REF]
        p = [boot_p(W, errs[(m, scheme)][idx] - ref) for m in others]
        for m, q in zip(others, holm(np.array(p))):
            worse = np.mean(errs[(m, scheme)][idx]) > ref.mean()
            marks[(m, scheme)] = "" if q >= 0.05 else ("$^{\\dagger}$" if worse else "$^{\\ast}$")
    lines = []
    for m in order:
        cells = []
        for scheme in SCHEMES:
            e = errs[(m, scheme)][idx]
            cell = f"{e.mean():.2f} ({np.mean(e <= 3) * 100:.0f})" + marks[(m, scheme)] if m != REF else \
                f"{e.mean():.2f} ({np.mean(e <= 3) * 100:.0f})"
            cells.append(f"\\textbf{{{cell}}}" if m == REF else cell)
        lines.append(f"{FAMILY_EN[family[m]]} & {name_en(m)} & " + " & ".join(cells) + r" \\")
    (OUT / f"table_all_{subset}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def held_out_groups(scheme, idx, fl):
    """Boolean masks over idx: random folds, the 49 fabrics that contain yarn c, or the fabrics of loom k."""
    if scheme == "leave-one-yarn-out":
        return [(DS.warp[idx] == c) | (DS.weft[idx] == c) for c in DS.codes]
    lab = fl[scheme][0][idx]
    return [lab == g for g in np.unique(lab)]


def table_shares(errs, idx):
    """Share of fabrics within 1, 2 and 3 CIEDE2000 units, key models, every scheme."""
    lines = []
    for m, label in MODELS.items():
        cells = []
        for scheme in SCHEMES:
            e = errs[(m, scheme)][idx]
            cells += [f"{np.mean(e <= t) * 100:.0f}" for t in (1, 2, 3)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    (OUT / "table_shares.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gain_tests(errs, idx, fl):
    """Same learner with and without the S-N layer: fabric-level and yarn-cluster inference."""
    W = yarn_weights(idx)
    loom = DS.loom
    out = []
    for learner in LEARNERS:
        for scheme in SCHEMES:
            a = errs[(f"ML ล้วน: {learner}", scheme)][idx]
            b = errs[(f"Hybrid: S-N + {learner}", scheme)][idx]
            d = a - b
            lo, hi = boot_ci(W, d)
            groups = held_out_groups(scheme, idx, fl)
            g_win = sum(a[g].mean() > b[g].mean() for g in groups)
            l_win = sum(a[loom[idx] == L].mean() > b[loom[idx] == L].mean() for L in range(1, 6))
            out.append(dict(learner=learner, scheme=scheme, pure=a.mean(), hyb=b.mean(), gain=d.mean() / a.mean(),
                            share=np.mean(b < a), lo=lo, hi=hi, p_clu=boot_p(W, d), p_fab=wilcoxon(a, b).pvalue,
                            folds=f"{g_win}/{len(groups)}", looms=f"{l_win}/5"))
    for key in ("p_clu", "p_fab"):
        for t, q in zip(out, holm(np.array([t[key] for t in out]))):
            t[key + "_holm"] = q
    return out


def table_gain(tests):
    lines = []
    for i, t in enumerate(tests):
        first = i % 3 == 0
        lname = t["learner"].replace(" (BPNN)", "")
        lines.append((f"\\multirow{{3}}{{*}}{{{lname}}}" if first else "") +
                     f" & {SCHEME_SHORT[t['scheme']]} & {t['pure']:.2f} & {t['hyb']:.2f} & {t['gain']:.0%}".replace("%", "\\%") +
                     f" & {t['share']:.0%}".replace("%", "\\%") +
                     f" & {t['pure'] - t['hyb']:.2f} [{t['lo']:.2f}, {t['hi']:.2f}]"
                     f" & {fmt_p(t['p_clu_holm'])} & {t['folds']} & {fmt_p(t['p_fab_holm'])} \\\\")
        if i % 3 == 2 and i < len(tests) - 1:
            lines.append("\\midrule")
    (OUT / "table_gain.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bounds_and_coverage(errs, idx, fl):
    """Empirical error quantiles per scenario, and their coverage on held-out groups: for each fold
    (loom, yarn fold or random fold) the quantile is computed from the other folds' errors only."""
    rows = []
    for scheme in SCHEMES:
        e = errs[(REF, scheme)][idx]
        groups = held_out_groups(scheme, idx, fl)
        qs = np.percentile(e, [50, 80, 90])
        cov = {}
        for level in (80, 90):
            c = [np.mean(e[g] <= np.percentile(e[~g], level)) for g in groups]
            cov[level] = (np.mean(c), np.min(c), np.max(c))
        rows.append((scheme, qs, cov))
    lines = []
    names = {"random 5-fold": "Known yarns and loom (random)",
             "leave-one-yarn-out": "New yarn colour (unseen yarn)",
             "leave-one-loom-out": "New warp set-up (unseen loom)"}
    for scheme, qs, cov in rows:
        lines.append(f"{names[scheme]} & " + " & ".join(f"{v:.2f}" for v in qs) + " & " +
                     " & ".join(f"{cov[L][0]:.0%} ({cov[L][1]:.0%}--{cov[L][2]:.0%})".replace("%", "\\%")
                                for L in (80, 90)) + r" \\")
    (OUT / "table_bounds.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def residual_analysis(idx, fl):
    """Why the residual form helps: spread of the target versus the S-N residual, and how often a
    test fabric's target (or residual) falls outside the range seen in its training fold."""
    sn = StearnsNoechel().fit(DS, idx)
    y = DS.lab[idx]
    r = y - sn.predict(DS, idx)
    lines = ["S-N fitted on all fabrics: w = %.3f, b = %.4f" % (sn.w, np.exp(sn.extra[0])),
             "channel | SD target | SD residual | range target | range residual"]
    for j, ch in enumerate("Lab"):
        lines.append(f"{ch} | {y[:, j].std():.2f} | {r[:, j].std():.2f} | {np.ptp(y[:, j]):.1f} | {np.ptp(r[:, j]):.1f}")
    lines.append(f"total variance target {y.var(0).sum():.1f}, residual {r.var(0).sum():.1f}, "
                 f"ratio {r.var(0).sum() / y.var(0).sum():.3f}")
    lines.append("share of test fabrics with L* outside the training range: target | S-N residual")
    out_rows = []
    for scheme, (lab, train) in fl.items():
        out_t = out_r = n = 0
        for k, tr in enumerate(train):
            te = np.flatnonzero(lab == k)
            if not len(te):
                continue
            law = StearnsNoechel().fit(DS, tr)
            rt, rte = DS.lab[tr] - law.predict(DS, tr), DS.lab[te] - law.predict(DS, te)
            yt, yte = DS.lab[tr], DS.lab[te]
            out_t += np.sum((yte[:, 0] < yt[:, 0].min()) | (yte[:, 0] > yt[:, 0].max()))
            out_r += np.sum((rte[:, 0] < rt[:, 0].min()) | (rte[:, 0] > rt[:, 0].max()))
            n += len(te)
        out_rows.append((scheme, out_t / n, out_r / n))
        lines.append(f"{scheme} | {out_t / n:.1%} | {out_r / n:.1%}")
    # warp lightness outside the training warps' range (the leave-one-loom-out L1 case)
    lab, train = fl["leave-one-loom-out"]
    wl = DS.warp_lab[:, 0]
    outside = np.zeros(N, bool)
    for k, tr in enumerate(train):
        te = np.flatnonzero(lab == k)
        outside[te] = (wl[te] > wl[tr].max()) | (wl[te] < wl[tr].min())
    return lines, outside, (y.var(0).sum(), r.var(0).sum(), r.var(0).sum() / y.var(0).sum(), y.std(0), r.std(0))


def uncertainty_tables():
    path = RES / "uncertainty_oof.csv"
    if not path.exists():
        return ["(uncertainty_oof.csv missing: run weavecolor.evaluate_revision uncertainty)"]
    rows = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        e, u = rows[(r["model"], r["scheme"])]
        e.append(float(r["de00"]))
        u.append(float(r["unc"]))
    lines, tex = [], []
    if not all((m, s) in rows for m in ("S-N + XGBoost, bagged", "S-N + GP") for s in SCHEMES):
        return [f"(uncertainty_oof.csv incomplete: {len(rows)}/6 model-scheme pairs)"]
    for model in ("S-N + XGBoost, bagged", "S-N + GP"):
        cells = []
        for scheme in SCHEMES:
            e, u = map(np.array, rows[(model, scheme)])
            auc = {t: roc_auc_score(e > t, u) for t in (2, 3, 4)}
            rho = spearmanr(u, e).statistic
            order = np.argsort(u)
            half = len(e) // 2
            lines.append(f"{model} | {scheme} | mean {e.mean():.2f} | rho {rho:.2f} | AUROC>2 {auc[2]:.2f} | "
                         f"AUROC>3 {auc[3]:.2f} | AUROC>4 {auc[4]:.2f} | certain half {e[order[:half]].mean():.2f} | "
                         f"uncertain half {e[order[half:]].mean():.2f}")
            cells += [f"{rho:.2f}".replace("-", "$-$"), f"{auc[3]:.2f}"]
        label = model.replace("S-N + GP", "S-N + GP (posterior SD)")
        tex.append(f"{label} & " + " & ".join(cells) + r" \\")
    (OUT / "table_unc.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return lines


def repeated_cv():
    path = RES / "repeated_cv.csv"
    if not path.exists():
        return ["(repeated_cv.csv missing: run weavecolor.evaluate_revision repeated)"]
    vals = defaultdict(dict)
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        vals[r["model"]][int(r["seed"])] = float(r["mean_de00"])
    seeds = sorted(set.intersection(*(set(v) for v in vals.values())))
    lines = [f"random 5-fold repeated with seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} repetitions), all fabrics",
             "model | mean of means | SD | min | max | rank among listed (per seed)"]
    models = list(vals)
    M = np.array([[vals[m][s] for s in seeds] for m in models])
    ranks = M.argsort(0).argsort(0) + 1
    tex = []
    for i, m in enumerate(models):
        rk = sorted(set(int(x) for x in ranks[i]))
        lines.append(f"{MODELS.get(m, m)} | {M[i].mean():.2f} | {M[i].std(ddof=1):.3f} | {M[i].min():.2f} | "
                     f"{M[i].max():.2f} | {rk}")
        tex.append(f"{MODELS.get(m, m)} & {M[i].mean():.2f} & {M[i].std(ddof=1):.2f} & "
                   f"{M[i].min():.2f}--{M[i].max():.2f} & " + "--".join(str(x) for x in (rk[0], rk[-1]))
                   .replace(f"{rk[0]}--{rk[0]}", str(rk[0])) + r" \\")
    ref = models.index(REF)
    xgb = models.index("ML ล้วน: XGBoost")
    lines.append(f"S-N + XGBoost better than pure XGBoost in {np.sum(M[ref] < M[xgb])}/{len(seeds)} seeds")
    (OUT / "table_repeated.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return lines


def selected_params():
    path = RES / "selected_params.csv"
    if not path.exists():
        return ["(selected_params.csv missing: run weavecolor.evaluate_revision params)"]
    count = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if "GP" in r["model"] or r["scheme"] == "all 625":
            continue
        count[(r["model"], r["scheme"])][r["params"]] += 1
    lines = ["most frequently selected hyper-parameters per outer fold (count / folds)"]
    for (m, s), c in count.items():
        top = max(c, key=c.get)
        lines.append(f"{m} | {s} | {top} ({c[top]}/{sum(c.values())})")
    return lines


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allerrs, family, order = load_errors()
    idx = subsets(DS)[PRIMARY]
    errs = allerrs[PRIMARY]
    fl = folds(idx)
    W = yarn_weights(idx)
    loom = DS.loom

    fig_design()
    fig_schemes(errs)
    fig_looms(errs)
    dist = input_distance(idx, fl)
    edges, dist_rows = fig_distance(errs, dist, idx)
    for subset in ("all", "clean"):
        table_all(allerrs[subset], family, order, subsets(DS)[subset], subset)
    table_shares(errs, idx)
    tests = gain_tests(errs, idx, fl)
    table_gain(tests)
    bounds = bounds_and_coverage(errs, idx, fl)
    res_lines, outside, _ = residual_analysis(idx, fl)

    L = [f"PRIMARY ANALYSIS: {PRIMARY} ({len(idx)} fabrics). CI: yarn-cluster (pigeonhole) bootstrap, "
         f"{B} resamples; fabric bootstrap CI shown for comparison",
         "model | scheme | mean [yarn-cluster 95% CI] | [fabric 95% CI] | share<=1 | share<=2 | share<=3 | "
         "median | P80 | P90"]
    rng = np.random.default_rng(0)
    for m, label in MODELS.items():
        for scheme in SCHEMES:
            e = errs[(m, scheme)][idx]
            lo, hi = boot_ci(W, e)
            fb = rng.choice(e, (2000, len(e))).mean(1)
            L.append(f"{label} | {scheme} | {e.mean():.2f} [{lo:.2f}, {hi:.2f}] | "
                     f"[{np.percentile(fb, 2.5):.2f}, {np.percentile(fb, 97.5):.2f}] | "
                     f"{np.mean(e <= 1):.0%} | {np.mean(e <= 2):.0%} | {np.mean(e <= 3):.0%} | {np.median(e):.2f} | "
                     f"{np.percentile(e, 80):.2f} | {np.percentile(e, 90):.2f}")
    L += ["", "pure learners worse than mean CIELAB (yarn-cluster bootstrap, Holm over the pure learners)"]
    pure = [m for m in order if family[m] == "ml"]
    for scheme in SCHEMES:
        base = errs[("เฉลี่ย L*a*b*", scheme)][idx]
        ps = holm(np.array([boot_p(W, errs[(m, scheme)][idx] - base) for m in pure]))
        L.append(scheme + ": " + "; ".join(f"{name_en(m)} {errs[(m, scheme)][idx].mean():.2f} p={p:.3g}"
                                          for m, p in zip(pure, ps)))
    L += ["", "S-N + XGBoost single vs bagged (bagged - single): mean diff [yarn-cluster CI], p"]
    for scheme in SCHEMES:
        d = errs[("Hybrid: S-N + XGBoost ensemble×10", scheme)][idx] - errs[(REF, scheme)][idx]
        lo, hi = boot_ci(W, d)
        L.append(f"{scheme}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] p={boot_p(W, d):.3g}")
    L += ["", "best model per scheme (mean)"]
    for scheme in SCHEMES:
        best = sorted(order, key=lambda m: errs[(m, scheme)][idx].mean())[:3]
        L.append(scheme + ": " + ", ".join(f"{name_en(m)} {errs[(m, scheme)][idx].mean():.2f}" for m in best))
    L += ["", "leave-one-loom-out mean per held-out loom (L1..L5)"]
    for m, label in MODELS.items():
        e = errs[(m, "leave-one-loom-out")]
        L.append(f"{label}: " + " ".join(f"{np.nanmean(e[idx][loom[idx] == k]):.2f}" for k in range(1, 6)))
    L += ["", "pure ML vs S-N hybrid, same learner | pure | hybrid | gain | fabrics improved | "
          "diff [yarn-cluster CI] | p cluster (Holm, 9) | p fabric Wilcoxon (Holm, 9) | folds improved | looms improved"]
    for t in tests:
        L.append(f"{t['learner']} | {t['scheme']} | {t['pure']:.2f} | {t['hyb']:.2f} | {t['gain']:.0%} | "
                 f"{t['share']:.0%} | {t['pure'] - t['hyb']:.2f} [{t['lo']:.2f}, {t['hi']:.2f}] | "
                 f"{t['p_clu_holm']:.1e} | {t['p_fab_holm']:.1e} | {t['folds']} | {t['looms']}")
    L += ["", "scenario error quantiles of S-N + XGBoost | 50 80 90 | held-out-group coverage of the "
          "80% and 90% quantiles: mean (min-max)"]
    for scheme, qs, cov in bounds:
        L.append(f"{scheme} | " + " ".join(f"{v:.2f}" for v in qs) + " | " +
                 " | ".join(f"{k}%: {v[0]:.0%} ({v[1]:.0%}-{v[2]:.0%})" for k, v in cov.items()))
    L += ["", "per-loom coverage of the leave-one-loom-out 90% quantile (computed on the other looms)"]
    e = errs[(REF, "leave-one-loom-out")][idx]
    lab = fl["leave-one-loom-out"][0][idx]
    L.append(" ".join(f"L{g + 1}: {np.mean(e[lab == g] <= np.percentile(e[lab != g], 90)):.0%}"
                      for g in np.unique(lab)))
    L += ["", "RESIDUAL TARGET ANALYSIS"] + res_lines
    L += ["", f"leave-one-loom-out fabrics whose warp L* lies outside the training warps' range: "
          f"{outside[idx].sum()} (looms {sorted(set(int(x) for x in loom[idx][outside[idx]]))})",
          "mean error on those / on the rest:"]
    for m in ("ML ล้วน: GP", "ML ล้วน: MLP (BPNN)", "ML ล้วน: XGBoost", "Hybrid: S-N + GP",
              "Hybrid: S-N + MLP (BPNN)", "Hybrid: S-N + XGBoost"):
        e = errs[(m, "leave-one-loom-out")]
        L.append(f"{MODELS[m]}: {np.nanmean(e[idx][outside[idx]]):.2f} / {np.nanmean(e[idx][~outside[idx]]):.2f}")
    L += ["", "DISTANCE TO NEAREST TRAINING INPUT (6-D CIELAB), pooled unseen-yarn + unseen-loom",
          "quartile edges: " + ", ".join(f"{v:.1f}" for v in edges),
          "model | mean error Q1..Q4 | Spearman(distance, error)"]
    for m, means, rho in dist_rows:
        L.append(f"{MODELS[m]} | " + " ".join(f"{v:.2f}" for v in means) + f" | {rho:.2f}")
    L += ["", "median distance per scheme: " + ", ".join(f"{s} {np.median(dist[s][idx]):.1f}" for s in SCHEMES)]
    L += ["", "UNCERTAINTY"] + uncertainty_tables()
    L += ["", "REPEATED RANDOM CV"] + repeated_cv()
    L += ["", "SELECTED HYPER-PARAMETERS"] + selected_params()

    # sensitivity: screened subset
    cidx = subsets(DS)["clean"]
    ce = allerrs["clean"]
    L += ["", f"SENSITIVITY: screened subset ({len(cidx)} fabrics)"]
    for m in ("ML ล้วน: GP", "ML ล้วน: XGBoost", "ML ล้วน: MLP (BPNN)", REF, "Hybrid: S-N + XGBoost ensemble×10"):
        L.append(f"{MODELS[m]}: " + ", ".join(f"{ce[(m, s)][cidx].mean():.2f}" for s in SCHEMES))
    Wc = yarn_weights(cidx)
    for learner in LEARNERS:
        for s in SCHEMES:
            d = ce[(f"ML ล้วน: {learner}", s)][cidx] - ce[(f"Hybrid: S-N + {learner}", s)][cidx]
            lo, hi = boot_ci(Wc, d)
            L.append(f"  gain {learner} {s}: {d.mean():.2f} [{lo:.2f}, {hi:.2f}]")
    (OUT / "numbers.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
