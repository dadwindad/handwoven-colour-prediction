// Yarn usage for a plain-weave cloth with optional colour stripes (plan, section 3).

// Spread `total` threads over a repeating stripe sequence [{code, count}].
// Returns Map(code -> threads). A partial repeat at the end is filled in order.
export function distributeStripes(total, stripes) {
  const out = new Map();
  const valid = stripes.filter((s) => s.code && s.count > 0);
  if (valid.length === 0 || total <= 0) return out;
  const repeat = valid.reduce((sum, s) => sum + s.count, 0);
  const full = Math.floor(total / repeat);
  let rest = total - full * repeat;
  for (const s of valid) {
    const extra = Math.min(s.count, rest);
    rest -= extra;
    out.set(s.code, (out.get(s.code) || 0) + full * s.count + extra);
  }
  return out;
}

export const DEFAULTS = {
  lengthM: 2, // finished cloth length
  widthCm: 100, // cloth width in the reed
  warpPerCm: 20, // ends per cm (sample value; count on real cloth)
  weftPerCm: 18, // picks per cm (sample value)
  warpCrimpPct: 8,
  weftCrimpPct: 6,
  loomWasteM: 0.5, // extra warp length per end lost on the loom
  selvedgeEnds: 8, // added to the first warp colour
};

// params: DEFAULTS fields + warpStripes, weftStripes, spec {tex, gramsPerSkein, pricePerSkein}
// Returns {warpEnds, weftPicks, rows: [{code, role, threads, meters, grams, skeins, cost}], totals}
export function computeUsage(params) {
  const p = { ...DEFAULTS, ...params };
  const { tex, gramsPerSkein, pricePerSkein } = p.spec;

  const bodyEnds = Math.round(p.widthCm * p.warpPerCm);
  const warpEnds = bodyEnds + p.selvedgeEnds;
  const weftPicks = Math.round(p.lengthM * 100 * p.weftPerCm);
  const warpLenPerEnd = p.lengthM * (1 + p.warpCrimpPct / 100) + p.loomWasteM;
  const weftLenPerPick = (p.widthCm / 100) * (1 + p.weftCrimpPct / 100);

  const warpCounts = distributeStripes(bodyEnds, p.warpStripes);
  const firstWarp = p.warpStripes.find((s) => s.code && s.count > 0);
  if (firstWarp && p.selvedgeEnds > 0) {
    warpCounts.set(firstWarp.code, warpCounts.get(firstWarp.code) + p.selvedgeEnds);
  }
  const weftCounts = distributeStripes(weftPicks, p.weftStripes);

  const row = (code, role, threads, lenEach) => {
    const meters = threads * lenEach;
    const grams = (meters * tex) / 1000;
    const skeins = Math.ceil(grams / gramsPerSkein - 1e-9);
    return { code, role, threads, meters, grams, skeins, cost: skeins * pricePerSkein };
  };
  const rows = [
    ...[...warpCounts].map(([c, n]) => row(c, "warp", n, warpLenPerEnd)),
    ...[...weftCounts].map(([c, n]) => row(c, "weft", n, weftLenPerPick)),
  ];

  // Skeins per colour across warp and weft, since one skein can serve both.
  const byColour = new Map();
  for (const r of rows) byColour.set(r.code, (byColour.get(r.code) || 0) + r.grams);
  const buy = [...byColour].map(([code, grams]) => {
    const skeins = Math.ceil(grams / gramsPerSkein - 1e-9);
    return { code, grams, skeins, cost: skeins * pricePerSkein };
  });

  return {
    warpEnds,
    weftPicks,
    warpLenPerEnd,
    weftLenPerPick,
    rows,
    buy,
    totals: {
      grams: rows.reduce((s, r) => s + r.grams, 0),
      skeins: buy.reduce((s, b) => s + b.skeins, 0),
      cost: buy.reduce((s, b) => s + b.cost, 0),
    },
  };
}
