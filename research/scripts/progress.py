"""Show progress of the running experiments, with a rough time remaining.

    .venv/bin/python scripts/progress.py            # print once
    .venv/bin/python scripts/progress.py --watch    # refresh every 15 s (Ctrl+C to stop)

Reads results/run_all.log (weavecolor.evaluate) and results/run_cnn_b.log
(weavecolor.evaluate_cnn_b). Time remaining is estimated from how long similar
finished steps took, so it gets better as the run goes on.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
LINE = re.compile(r"^(?P<model>.+?)\s+(?P<subset>clean|all)\s+(?P<scheme>random 5-fold|leave-one-yarn-out|"
                  r"leave-one-loom-out)\s+mean\s+(?P<mean>[\d.]+)\s+\(\s*(?P<secs>\d+)s\)")
SCHEMES = ["random 5-fold", "leave-one-yarn-out", "leave-one-loom-out"]
FOLDS = {"random 5-fold": 5, "leave-one-yarn-out": 25, "leave-one-loom-out": 5}
LEARNER_KEYS = ["GP", "XGBoost", "MLP", "SVR", "Random Forest", "KNN", "Polynomial Ridge"]


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def fmt_time(s):
    s = int(max(s, 0))
    return f"{s // 3600} ชม. {s % 3600 // 60} นาที" if s >= 3600 else f"{s // 60} นาที {s % 60} วิ"


def bar(done, total, width=30):
    n = int(width * done / total) if total else 0
    return "█" * n + "░" * (width - n) + f" {done}/{total} ({done / total * 100:.0f}%)" if total else ""


def key_of(model):
    """Group models whose steps take similar time: by learner, else 'fast'."""
    for k in LEARNER_KEYS:
        if k in model:
            return k
    return "fast"


def main_run():
    log = RESULTS / "run_all.log"
    if not log.exists():
        return ["การเปรียบเทียบแบบจำลอง: ยังไม่เริ่ม"]
    from weavecolor.evaluate import model_factories
    names = [make().name for make in model_factories()]
    units = [(m, s, sc) for m in names for s in ("clean", "all") for sc in SCHEMES]

    done = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line.split("%")[-1])
        if m:
            done[(m["model"].strip(), m["subset"], m["scheme"])] = int(m["secs"])
    todo = [u for u in units if u not in done]

    # estimate each remaining step from finished steps with the same learner and scheme
    def estimate(u):
        same = [t for d, t in done.items() if key_of(d[0]) == key_of(u[0]) and d[2] == u[2]]
        if same:
            return sum(same) / len(same) * (2.5 if u[0].startswith("Hybrid") and key_of(u[0]) == "GP" else 1)
        any_scheme = [t for d, t in done.items() if key_of(d[0]) == key_of(u[0])]
        return (sum(any_scheme) / len(any_scheme) if any_scheme else 30) * FOLDS[u[2]] / 5

    is_running = running("m weavecolor.evaluate$") or running("weavecolor.evaluate --")
    lines = ["การเปรียบเทียบแบบจำลอง (CPU)", "  " + bar(len(done), len(units))]
    if todo and is_running:
        since_last = time.time() - os.path.getmtime(log)
        cur = todo[0]
        eta = sum(estimate(u) for u in todo) - min(since_last, estimate(cur))
        upcoming = [m for m in dict.fromkeys(u[0] for u in todo) if m != cur[0]][:3]
        lines += [f"  กำลังทำ : {cur[0]}  [{cur[1]} / {cur[2]}]  (ทำมาแล้ว {fmt_time(since_last)})",
                  f"  ถัดไป   : {', '.join(upcoming) or '-'}",
                  f"  เหลือประมาณ: {fmt_time(eta)}  (ประมาณจากขั้นที่คล้ายกันที่ทำเสร็จแล้ว)"]
    elif todo:
        lines.append("  ⚠️ ไม่ได้รันอยู่ แต่ยังไม่ครบ — รันคำสั่งเดิมซ้ำเพื่อทำต่อจากจุดที่ค้าง")
    else:
        lines.append("  ✅ เสร็จแล้ว → results/comparison_all_models.md")
    return lines


def cnn_b_run():
    log = RESULTS / "run_cnn_b.log"
    if not log.exists():
        return ["CNN-B: ยังไม่เริ่ม"]
    text = log.read_text(encoding="utf-8")
    finished = [(s, int(t)) for s, t in re.findall(r"^(random 5-fold|leave-one-yarn-out|leave-one-loom-out) \{.*\} (\d+)s",
                                                    text, re.M)]
    folds_done = sum(FOLDS[s] for s, _ in finished)
    total = sum(FOLDS.values())
    lines = ["CNN-B ภาพถ่ายผ้า → สี (GPU)"]
    if "wrote" in text:
        return lines + ["  " + bar(total, total), "  ✅ เสร็จแล้ว → results/cnn_b_fabric_photo.md"]
    if not running("evaluate_cnn_b"):
        return lines + ["  " + bar(folds_done, total), "  ⚠️ ไม่ได้รันอยู่" + (" (มี error ดู results/run_cnn_b.log)" if "Traceback" in text else "")]
    per_fold = (sum(t for _, t in finished) / folds_done) if folds_done else 95
    since_last = time.time() - os.path.getmtime(log) if finished else None
    cur = SCHEMES[len(finished)]
    in_scheme = min(int((since_last or 0) / per_fold), FOLDS[cur] - 1) if finished else 0
    remaining = (total - folds_done) * per_fold - (since_last or 0)
    return lines + ["  " + bar(folds_done + in_scheme, total) + "  (รอบย่อยในช่วงปัจจุบันเป็นค่าประมาณ)",
                    f"  กำลังทำ : {cur}  ประมาณรอบที่ {in_scheme + 1}/{FOLDS[cur]}",
                    f"  เหลือประมาณ: {fmt_time(remaining)}  (~{per_fold:.0f} วิ/รอบ)"]


def show():
    print(time.strftime("%H:%M:%S"), "— ความคืบหน้าการทดลอง\n")
    print("\n".join(main_run()))
    print()
    print("\n".join(cnn_b_run()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh every 15 seconds")
    ap.add_argument("--every", type=int, default=15)
    a = ap.parse_args()
    if not a.watch:
        show()
    else:
        try:
            while True:
                print("\033[2J\033[H", end="")
                show()
                print("\n(Ctrl+C เพื่อออก)")
                time.sleep(a.every)
        except KeyboardInterrupt:
            pass
