"""Load the Table 4.7 dataset (data/table4_7_lab.csv)."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLE_CSV = ROOT / "data" / "table4_7_lab.csv"

# Thai names from Table 4.1 of the source report (spelling kept as printed).
YARN_NAMES = {
    "01": "ขาว", "02": "เหลืองสด", "03": "เหลืองทอง", "05": "ชมพู", "07": "ม่วงแดง",
    "12": "ม่วง", "13": "ฟ้าน้ำทะเล", "15": "น้ำเงิน", "17": "เหลืองเข", "18": "เขียวก้านมะลิ",
    "19": "เขียวหัวเป็ดแก่", "20": "แดงสด", "21": "มะขามอ่อน", "22": "เขียวกลาง", "23": "กรม",
    "24": "ดำ", "25": "ขี้ม้าแก่นขนุนแก่", "27": "มะขามแก่", "30": "แดงแก่", "37": "เทา",
    "38": "อิฐ", "40": "ขี้ม้าสด", "41": "ขี้ม้าแก่", "43": "กะปิอ่อน", "50": "โอรส",
}

# Each loom (ฟืม) carried five warp colours (section 4.3.6 of the source report).
LOOM_OF_WARP = {
    code: loom
    for loom, codes in {
        1: ["01", "02", "03", "05", "07"],
        2: ["12", "13", "15", "17", "18"],
        3: ["19", "20", "21", "22", "23"],
        4: ["24", "25", "27", "30", "37"],
        5: ["38", "40", "41", "43", "50"],
    }.items()
    for code in codes
}


@dataclass
class Dataset:
    codes: list            # 25 yarn codes, sorted
    yarn_lab: dict         # code -> np.array(3) measured yarn colour
    yarn_flag: dict        # code -> bool
    warp: np.ndarray       # (625,) warp code per fabric
    weft: np.ndarray       # (625,) weft code per fabric
    lab: np.ndarray        # (625, 3) measured fabric colour
    flag: np.ndarray       # (625,) bool, True when marked '*'

    def yarn_lab_array(self, codes):
        return np.stack([self.yarn_lab[c] for c in codes])

    @property
    def warp_lab(self):
        return self.yarn_lab_array(self.warp)

    @property
    def weft_lab(self):
        return self.yarn_lab_array(self.weft)

    @property
    def loom(self):
        return np.array([LOOM_OF_WARP[c] for c in self.warp])


def load(path=TABLE_CSV):
    yarn_lab, yarn_flag = {}, {}
    warp, weft, lab, flag = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = np.array([float(row["L"]), float(row["a"]), float(row["b"])])
            flagged = row.get("flag", "") == "*"
            if row["weft"]:
                warp.append(row["warp"])
                weft.append(row["weft"])
                lab.append(value)
                flag.append(flagged)
            else:
                yarn_lab[row["warp"]] = value
                yarn_flag[row["warp"]] = flagged
    return Dataset(
        codes=sorted(yarn_lab),
        yarn_lab=yarn_lab,
        yarn_flag=yarn_flag,
        warp=np.array(warp),
        weft=np.array(weft),
        lab=np.stack(lab),
        flag=np.array(flag),
    )
