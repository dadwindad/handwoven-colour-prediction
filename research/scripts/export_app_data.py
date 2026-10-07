"""Build app/data/dataset.json from the CSV files in data/.

The CSV files stay the single source of truth; rerun this after editing them:
    .venv/bin/python scripts/export_app_data.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weavecolor import data as D  # noqa: E402

ROOT = D.ROOT
OUT = ROOT / "app" / "data" / "dataset.json"


def main():
    reasons = {}
    with open(D.TABLE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reasons[(row["warp"], row["weft"])] = row.get("flag_reason", "")

    ds = D.load()
    yarns = [
        {
            "code": c,
            "name": D.YARN_NAMES[c],
            "loom": D.LOOM_OF_WARP[c],
            "lab": [round(v, 2) for v in ds.yarn_lab[c]],
            "flag": bool(ds.yarn_flag[c]),
            "flagReason": reasons[(c, "")],
        }
        for c in ds.codes
    ]
    fabrics = [
        {
            "warp": w,
            "weft": f,
            "lab": [round(v, 2) for v in lab],
            "flag": bool(flag),
            "flagReason": reasons[(w, f)],
        }
        for w, f, lab, flag in zip(ds.warp, ds.weft, ds.lab, ds.flag)
    ]
    with open(ROOT / "data" / "yarn_specs.csv", newline="", encoding="utf-8") as f:
        specs = []
        for row in csv.DictReader(f):
            specs.append({
                "id": row["yarn_id"],
                "name": row["name"],
                "composition": row["composition"],
                "count": row["count"],
                "tex": float(row["tex"]),
                "gramsPerSkein": float(row["grams_per_skein"]),
                "pricePerSkein": float(row["price_thb_per_skein"]),
                "pricePerKg": float(row["price_thb_per_kg"]),
                "sourceUrl": row["source_url"],
                "retrieved": row["retrieved"],
                "notes": row["notes"],
            })

    payload = {
        "source": "พนมพร ขุนทอง และ วทัญญู เสียบไธสง (2564). อิทธิพลสีด้ายโทเรที่เกิดกับลายผ้าทอมือ. "
                  "มหาวิทยาลัยราชภัฏบุรีรัมย์. ตาราง 4.7 (ใช้โดยได้รับอนุญาตจากคณะผู้วิจัย)",
        "measurement": "วัดด้วย spectrophotometer Datacolor 800 family ของสาขาสิ่งทอ แหล่งแสงซีนอน คำนวณที่ D65 เรขาคณิต d/8° (ยังไม่ทราบ observer 2°/10° และ SCI/SCE)",
        "pairConvention": "warp+weft",
        "yarns": yarns,
        "fabrics": fabrics,
        "yarnSpecs": specs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(yarns)} yarns, {len(fabrics)} fabrics, "
          f"{sum(x['flag'] for x in fabrics)} flagged")


if __name__ == "__main__":
    main()
