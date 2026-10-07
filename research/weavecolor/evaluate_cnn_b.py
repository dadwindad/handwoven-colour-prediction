"""CNN-B: read fabric L*a*b* from a photo of the woven fabric (fine-tuned ResNet-18).

This is a different task from the main comparison: the fabric already exists and
we only want its colour from a phone-style photo instead of a spectrophotometer.
Baselines: the plain average colour of the photo, and that average after a
linear calibration fitted on the training fold.

    .venv/bin/python -m weavecolor.evaluate_cnn_b [--epochs 20]
"""
import argparse
import time

import numpy as np
from sklearn.linear_model import LinearRegression

from . import data as D
from .cnn import fabric_photo_paths, photo_mean_lab, train_predict_cnn_b
from .colorlib import delta_e_2000
from .evaluate import RESULTS, SCHEMES


def main(epochs):
    ds = D.load()
    idx = np.flatnonzero(~ds.flag)
    paths = fabric_photo_paths(ds)
    mean_lab = photo_mean_lab(paths)
    y = ds.lab

    lines = [
        "# CNN-B: อ่านสีผ้าจากภาพถ่ายผ้า (ภาพจากเล่มงานวิจัย)",
        "",
        "งานนี้ต่างจากตารางหลัก: ผ้าทอเสร็จแล้ว และใช้ภาพถ่ายแทนเครื่องวัดสี "
        "ค่าคือ CIEDE2000 เฉลี่ยนอกชุดฝึก (ร้อยละที่คลาดไม่เกิน 3) ใช้ข้อมูลที่ตัดแถว * ออก",
        "",
        f"- ResNet-18 (ImageNet) fine-tune {epochs} epochs, augmentation เฉพาะเชิงเรขาคณิต (ไม่ปรับสีภาพ), "
        "loss = MSE บน L*a*b*/100, test-time augmentation 3 แบบ",
        "",
        "| แบบจำลอง | " + " | ".join(SCHEMES) + " |",
        "|---|" + "---|" * len(SCHEMES),
    ]
    results = {"สีเฉลี่ยของภาพ (ไม่เรียนรู้)": {}, "สีเฉลี่ย + ปรับเทียบเชิงเส้น": {}, "CNN-B: ResNet-18 fine-tune": {}}
    for scheme, splitter in SCHEMES.items():
        errs = {k: np.full(len(y), np.nan) for k in results}
        t0 = time.time()
        for tr, te in splitter(ds, idx):
            errs["สีเฉลี่ยของภาพ (ไม่เรียนรู้)"][te] = delta_e_2000(mean_lab[te], y[te])
            lin = LinearRegression().fit(mean_lab[tr], y[tr])
            errs["สีเฉลี่ย + ปรับเทียบเชิงเส้น"][te] = delta_e_2000(lin.predict(mean_lab[te]), y[te])
            pred = train_predict_cnn_b(paths, y, tr, te, epochs=epochs)
            errs["CNN-B: ResNet-18 fine-tune"][te] = delta_e_2000(pred, y[te])
        for k, e in errs.items():
            results[k][scheme] = e[idx]
        print(scheme, {k: round(float(np.nanmean(e[idx])), 2) for k, e in errs.items()},
              f"{time.time() - t0:.0f}s", flush=True)

    for k, per in results.items():
        lines.append(f"| {k} | " + " | ".join(
            f"{per[s].mean():.2f} ({np.mean(per[s] <= 3) * 100:.0f}%)" for s in SCHEMES) + " |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cnn_b_fabric_photo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", RESULTS / "cnn_b_fabric_photo.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    main(ap.parse_args().epochs)
