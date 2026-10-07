import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { deltaE2000, hexToLab, labToHex, labToRgb } from "../js/color.js";

const sharma = JSON.parse(
  readFileSync(new URL("../../research/tests/sharma2005_ciede2000_testdata.json", import.meta.url)),
);

test("CIEDE2000 matches all 34 pairs of Sharma et al. (2005)", () => {
  assert.equal(sharma.length, 34);
  for (const { pair, lab1, lab2, de00 } of sharma) {
    assert.ok(Math.abs(deltaE2000(lab1, lab2) - de00) < 1e-4, `pair ${pair}`);
    assert.ok(Math.abs(deltaE2000(lab2, lab1) - de00) < 1e-4, `pair ${pair} reversed`);
  }
});

test("hex -> Lab -> hex round trip", () => {
  for (const hex of ["#ffffff", "#000000", "#c0392b", "#2e86c1", "#7d6608"]) {
    assert.equal(labToHex(hexToLab(hex)), hex);
  }
});

test("sRGB white is L*=100", () => {
  const [L, a, b] = hexToLab("#ffffff");
  assert.ok(Math.abs(L - 100) < 0.01 && Math.abs(a) < 0.01 && Math.abs(b) < 0.01);
});

test("very saturated yellow from the dataset is reported out of gamut", () => {
  assert.equal(labToRgb([72.65, 13.07, 85.43]).inGamut, false); // yarn 02
  assert.equal(labToRgb([50, 0, 0]).inGamut, true);
});
