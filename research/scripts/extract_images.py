"""Extract fabric and yarn photos from Tables 4.2-4.6 of the source report.

Layout of every table page: a left column with the weft code and a weft-yarn
photo, then one row of five fabric photos per weft. The first page of each
table also has a header row (warp codes + warp-yarn photos). Continuation
pages carry over the warp order of the last header seen.

Writes:
  data/images/fabric/<warp>+<weft>.<ext>    625 fabric photos
  data/images/yarn/<code>_<role>_p<page>.<ext>  yarn photos (warp header / weft row)
  data/images/index.csv
"""
import csv
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "images"
FIRST_PAGE, LAST_PAGE = 32, 65   # 1-based PDF pages holding Tables 4.2-4.6
LEFT_COL_X = 150                 # images/labels left of this belong to the weft column
CODE = re.compile(r"\d{2}")


def clusters(images, tol=40):
    """Group images whose top edges are within tol points into rows."""
    rows = []
    for im in sorted(images, key=lambda i: i["bbox"][1]):
        if rows and abs(im["bbox"][1] - rows[-1][0]["bbox"][1]) <= tol:
            rows[-1].append(im)
        else:
            rows.append([im])
    return rows


def main():
    pdf = next((ROOT / "source").glob("*.pdf"))
    doc = pymupdf.open(pdf)
    (OUT / "fabric").mkdir(parents=True, exist_ok=True)
    (OUT / "yarn").mkdir(parents=True, exist_ok=True)

    def save(xref, name):
        img = doc.extract_image(xref)
        path = OUT / f"{name}.{img['ext']}"
        path.write_bytes(img["image"])
        return path

    warp_order = None
    index, seen = [], set()
    for pno in range(FIRST_PAGE - 1, LAST_PAGE):
        page = doc[pno]
        words = [w for w in page.get_text("words") if CODE.fullmatch(w[4]) and w[1] > 60]
        images = page.get_image_info(xrefs=True)

        right = [i for i in images if i["bbox"][0] >= LEFT_COL_X - 25 and i["bbox"][2] > LEFT_COL_X + 20]
        left = [i for i in images if i not in right]
        labels = [w for w in words if w[0] < LEFT_COL_X - 20]

        for row in clusters(right):
            row.sort(key=lambda i: i["bbox"][0])
            top, bottom = min(i["bbox"][1] for i in row), max(i["bbox"][3] for i in row)
            # the weft label is printed just above its row; take the nearest one above
            above = [w for w in labels if top - 150 <= w[1] <= top + 25]
            label = [max(above, key=lambda w: w[1])] if above else []
            header = [w for w in words if w[0] >= LEFT_COL_X - 20 and top - 90 <= w[1] < top]
            if len(header) == 5:  # header first: the first weft label can sit just below it
                warp_order = [w[4] for w in sorted(header, key=lambda w: w[0])]
                for code, im in zip(warp_order, row):
                    index.append({"kind": "yarn", "role": "warp", "warp": code, "weft": "", "page": pno + 1,
                                  "file": save(im["xref"], f"yarn/{code}_warp_p{pno + 1}").relative_to(ROOT)})
                continue
            if len(row) != 5 or len(label) != 1 or warp_order is None:
                sys.exit(f"page {pno + 1}: unexpected row (images={len(row)}, labels={len(label)})")
            weft = label[0][4]
            for warp, im in zip(warp_order, row):
                if (warp, weft) in seen:
                    sys.exit(f"duplicate pair {warp}+{weft} on page {pno + 1}")
                seen.add((warp, weft))
                index.append({"kind": "fabric", "role": "", "warp": warp, "weft": weft, "page": pno + 1,
                              "file": save(im["xref"], f"fabric/{warp}+{weft}").relative_to(ROOT)})
            # weft yarn photo: the left-column image closest to this row
            near = [i for i in left if i["bbox"][1] < bottom + 40 and i["bbox"][3] > top - 70]
            if near:
                im = min(near, key=lambda i: abs(i["bbox"][1] - top))
                index.append({"kind": "yarn", "role": "weft", "warp": "", "weft": weft, "page": pno + 1,
                              "file": save(im["xref"], f"yarn/{weft}_weft_p{pno + 1}").relative_to(ROOT)})

    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "role", "warp", "weft", "page", "file"])
        w.writeheader()
        w.writerows(index)
    n_fab = sum(r["kind"] == "fabric" for r in index)
    n_yarn = sum(r["kind"] == "yarn" for r in index)
    print(f"fabric photos: {n_fab} (unique pairs {len(seen)}), yarn photos: {n_yarn}")


if __name__ == "__main__":
    main()
