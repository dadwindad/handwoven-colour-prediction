# Data and code: colour prediction of handwoven fabrics under unseen yarns and looms

This repository contains the data set, analysis code and results for the paper

> Vongpramate, D., Boonket, K., Hobanthad, S., Khuntong, P., Siapthaisong, W. and Saenkham, T.
> *Shift-aware evaluation and physics-guided residual learning for colour prediction of handwoven fabrics.*
> Manuscript submitted to Expert Systems with Applications.

**Version 1.1.0.** Archived on Zenodo: [doi:10.5281/zenodo.23205538](https://doi.org/10.5281/zenodo.23205538)
(concept DOI, all versions). Version 1.1.0 changes the evaluation protocol (see *Changes in 1.1.0*
below) and adds the few-shot loom-calibration experiment; versions 1.0.x contain the earlier protocol.

It also contains a small offline web app (and a macOS wrapper) that lets weavers look up the
woven colour of any pair of yarns.

## Data

The fabrics come from a community weaving project in Samet sub-district, Buriram, Thailand
(Khuntong & Siapthaisong, 2021; supervised by S. Hobanthad, Program in Mathematics, Faculty of
Science, Buriram Rajabhat University). Twenty-five pre-dyed polyester-blend ("Toray") yarn
colours were split into five groups of five; each group was the warp of one hand-loom set-up
(10 m per loom), and every one of the 25 colours was woven as weft on every loom. The result is
a full 25 × 25 factorial of 625 plain-weave fabrics.

Yarns and fabrics were measured with a Datacolor 800-series spectrophotometer (xenon source,
d/8° geometry) and reported as CIELAB under illuminant D65. Only CIELAB values were recorded.
The specular setting (SCI/SCE), aperture, number of readings and standard observer were not
documented; the analysis uses the 10° observer (the 2° observer changes mean errors by < 0.1
CIEDE2000).

### `data/table4_7_lab.csv`

650 rows: 25 yarns and 625 fabrics.

| column | meaning |
|---|---|
| `warp` | yarn colour code (two digits, e.g. `01`) of the warp |
| `weft` | yarn colour code of the weft; **empty for the 25 yarn rows** (the row is the yarn itself) |
| `L`, `a`, `b` | measured CIELAB (D65) |
| `flag` | `*` if the row failed a consistency check, otherwise empty |
| `flag_reason` | which check: `yarn vs 38+38` (yarn differs from its own same-colour fabric) or `asym X+Y/Y+X` (fabric and its warp/weft-swapped pair differ), with the CIEDE2000 value |

Loom set-ups (warp colour codes): L1 01 02 03 05 07 · L2 12 13 15 17 18 · L3 19 20 21 22 23 ·
L4 24 25 27 30 37 · L5 38 40 41 43 50. `X+Y` always means warp X, weft Y.

The 29 flagged fabrics are **kept** in the primary analysis; excluding them is reported only as
a sensitivity analysis, because the flags were derived from the measured fabric colours.

### Other data

| path | content |
|---|---|
| `data/images/yarn/`, `data/images/fabric/` | photographs of the yarns and fabrics from the project report (simple light box, **not colour-calibrated**); `index.csv` maps files to codes |
| `data/images/yarn_embeddings_resnet18.npz` | ResNet-18 (ImageNet) features of the yarn photographs, used by the image-feature baseline |
| `data/yarn_specs.csv` | retail specification and price of the yarn line (retrieved online, used by the app's yarn calculator; not confirmed to match the lot used) |

## Results

`research/results/` holds the out-of-fold predictions behind every table and figure:

| file | content |
|---|---|
| `oof_errors.csv` | out-of-fold CIEDE2000 of every model (`model`), split scheme (`scheme`: random 5-fold, leave-one-yarn-out, leave-one-loom-out) and subset (`subset`: `all` = 625 fabrics, the primary analysis; `clean` = 596 unflagged fabrics), per fabric (`row` = fabric index in data order) |
| `uncertainty_oof.csv` | error and predictive uncertainty of the bagged S-N + XGBoost and S-N + GP hybrids |
| `repeated_cv.csv` | random 5-fold CV repeated with seeds 0–9 |
| `selected_params.csv` | hyper-parameters chosen by the inner CV in every outer fold, and fitted mixing-law parameters |
| `calibration.csv` | few-shot calibration of a new loom: mean error after an offset from n calibration fabrics, per model, strategy and held-out loom |
| `observer_check.csv` | key physics and hybrid models with the 2° instead of the 10° observer |
| `archive/` | results of the earlier protocol (versions 1.0.x), kept for the comparison in the paper's appendix |

Model names in these files are those used in the code; some are in Thai:
`เฉลี่ย L*a*b*` = mean of the two yarn colours, `ML ล้วน: X` = pure learner X,
`Hybrid: LAW + X` = learner X on the residual of mixing law LAW,
`D-G (L*a*b* ถ่วงน้ำหนัก)` = weighted CIELAB average, `modified S-N (ประมาณ λ)` = modified Stearns–Noechel,
`CNN-A: ResNet-18 ภาพด้าย + X` = yarn-photo features + X.

## Reproducing the paper

Python 3.11 or later; the results were produced with Python 3.14.7 on macOS (Apple silicon) and
the exact package versions in `research/requirements-lock.txt` (use it instead of
`requirements.txt` to reproduce the numbers exactly). On macOS, XGBoost needs OpenMP
(`brew install libomp`).

```sh
cd research
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests                        # CIEDE2000 against Sharma et al. (2005), models

# tables, figures and every number quoted in the paper, from the saved results (seconds)
.venv/bin/python scripts/make_paper_figures.py          # -> ../paper/figures/ (numbers.txt, table_*.tex, fig_*.pdf)

# rerun the experiments (hours; resumable; overwrite/extend results/)
.venv/bin/python -m weavecolor.evaluate                  # all 31 models x 3 schemes x 2 subsets
.venv/bin/python -m weavecolor.evaluate_revision uncertainty
.venv/bin/python -m weavecolor.evaluate_revision repeated
.venv/bin/python -m weavecolor.evaluate_revision params
.venv/bin/python -m weavecolor.evaluate_calibration      # few-shot loom calibration
.venv/bin/python -m weavecolor.evaluate_observer         # 2-degree observer check
```

Inner (tuning) cross-validation mirrors the outer scheme: shuffled 3-fold for random 5-fold, three
groups of held-out yarns for leave-one-yarn-out, and leave-one-training-loom-out for
leave-one-loom-out (`weavecolor.evaluate.inner_cv`). Under leave-one-yarn-out a fabric of two
different yarns is held out in two folds and its error is the mean of both.

### Changes in 1.1.0

- Inner CV now mirrors the outer scheme (1.0.x used shuffled inner folds in every scheme).
- Leave-one-yarn-out averages the two folds of each fabric (1.0.x kept the fold of the
  higher-coded yarn only).
- New experiment `weavecolor/evaluate_calibration.py`; bootstrap p-values are reported as bounds
  when no resample crosses zero.
- Random 5-fold results are unchanged; `results/archive/` keeps the 1.0.x results.

Main modules: `weavecolor/colorlib.py` (colour conversions, CIEDE2000), `weavecolor/models.py`
(mixing laws, learners with their hyper-parameter grids, hybrids), `weavecolor/evaluate.py`
(split schemes, nested cross-validation). All random seeds are fixed in the code.

The image models need extra packages (`pip install -r requirements-optional.txt`):
`python -m weavecolor.cnn` rebuilds the yarn-photo features and
`python -m weavecolor.evaluate_cnn_b` runs the fabric-photo CNN. `scripts/extract_images.py`
extracted the photographs from the original report, which is not redistributed here.

## Web app

`app/` is a static, offline-capable web app (no build step): `cd app && python3 -m http.server 8000`.
Tests: `cd app && npm test` (Node 20+). `macos/build.sh` packages it as a macOS app.

## Licence

Code: MIT (`LICENSE`). Data, images and results: CC BY 4.0 (`data/LICENSE`).
Please cite the paper and this repository (see `CITATION.cff`).

## Acknowledgements

We thank the weaving community of Samet sub-district, Buriram, for weaving the fabrics, and the
Faculty of Science, Buriram Rajabhat University, for kindly providing the colour measurements.

## References

Khuntong, P., Siapthaisong, W., 2021. *The influence of Toray yarn colours on hand-woven fabric
patterns* (in Thai). Technical report, Faculty of Science, Buriram Rajabhat University.

Sharma, G., Wu, W., Dalal, E.N., 2005. The CIEDE2000 color-difference formula: Implementation
notes, supplementary test data, and mathematical observations. *Color Research & Application* 30, 21–30.
