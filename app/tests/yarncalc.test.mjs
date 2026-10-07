import { test } from "node:test";
import assert from "node:assert/strict";
import { computeUsage, distributeStripes } from "../js/yarncalc.js";

const spec = { tex: 29.5, gramsPerSkein: 70, pricePerSkein: 32 };

test("stripes: full repeats plus a partial repeat filled in order", () => {
  const m = distributeStripes(25, [
    { code: "01", count: 4 },
    { code: "24", count: 6 },
  ]);
  // 2 full repeats (20) + 5 left: 4 to "01", 1 to "24"
  assert.deepEqual([...m], [["01", 12], ["24", 13]]);
});

test("stripes: same colour twice in a repeat is merged", () => {
  const m = distributeStripes(10, [
    { code: "01", count: 2 },
    { code: "24", count: 1 },
    { code: "01", count: 2 },
  ]);
  assert.equal(m.get("01") + m.get("24"), 10);
  assert.equal(m.get("24"), 2);
});

test("single-colour cloth: hand-checked numbers", () => {
  const r = computeUsage({
    lengthM: 2, widthCm: 100, warpPerCm: 20, weftPerCm: 18,
    warpCrimpPct: 10, weftCrimpPct: 5, loomWasteM: 0.5, selvedgeEnds: 0,
    warpStripes: [{ code: "01", count: 1 }],
    weftStripes: [{ code: "24", count: 1 }],
    spec,
  });
  assert.equal(r.warpEnds, 2000);
  assert.equal(r.weftPicks, 3600);
  const warp = r.rows.find((x) => x.role === "warp");
  const weft = r.rows.find((x) => x.role === "weft");
  // warp: 2000 ends x (2 x 1.10 + 0.5) m = 5400 m -> 5400 x 29.5 / 1000 = 159.3 g -> 3 skeins
  assert.ok(Math.abs(warp.meters - 5400) < 1e-9);
  assert.ok(Math.abs(warp.grams - 159.3) < 1e-9);
  assert.equal(warp.skeins, 3);
  // weft: 3600 picks x 1.05 m = 3780 m -> 111.51 g -> 2 skeins
  assert.ok(Math.abs(weft.meters - 3780) < 1e-9);
  assert.equal(weft.skeins, 2);
  assert.equal(r.totals.skeins, 5);
  assert.equal(r.totals.cost, 160);
});

test("selvedge ends go to the first warp colour", () => {
  const r = computeUsage({
    selvedgeEnds: 8,
    warpStripes: [{ code: "05", count: 1 }],
    weftStripes: [{ code: "05", count: 1 }],
    spec,
  });
  assert.equal(r.rows.find((x) => x.role === "warp").threads, 2008);
});

test("colour used in both warp and weft is bought once", () => {
  const r = computeUsage({
    warpStripes: [{ code: "05", count: 1 }],
    weftStripes: [{ code: "05", count: 1 }],
    spec,
  });
  assert.equal(r.buy.length, 1);
  const grams = r.rows.reduce((s, x) => s + x.grams, 0);
  assert.equal(r.buy[0].skeins, Math.ceil(grams / 70));
});
