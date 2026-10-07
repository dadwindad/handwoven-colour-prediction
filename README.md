# Data and code: colour prediction of handwoven fabrics under unseen yarns and looms

This repository contains the data set, analysis code and results for the paper

> Vongpramate, D., Boonket, K., Hobanthad, S., Khuntong, P., Siapthaisong, W. and Saenkham, T.
> *Shift-aware evaluation and physics-guided residual learning for colour prediction of handwoven fabrics.*
> Manuscript submitted to Expert Systems with Applications.

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

Model names in these files are those used in the code; some are in Thai:
`เฉลี่ย L*a*b*` = mean of the two yarn colours, `ML ล้วน: X` = pure learner X,
`Hybrid: LAW + X` = learner X on the residual of mixing law LAW,
`D-G (L*a*b* ถ่วงน้ำหนัก)` = weighted CIELAB average, `modified S-N (ประมาณ λ)` = modified Stearns–Noechel,
`CNN-A: ResNet-18 ภาพด้าย + X` = yarn-photo features + X.

## Reproducing the paper

Python 3.11 or later. On macOS, XGBoost needs OpenMP (`brew install libomp`).

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
```

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
