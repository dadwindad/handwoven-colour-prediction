// Colour maths for display and search.
// Screen colours use sRGB with a D65 (2°) white, which is what displays expect.

const WHITE = [95.047, 100.0, 108.883];
const EPS = 216 / 24389;
const KAPPA = 24389 / 27;

export function labToXyz([L, a, b]) {
  const fy = (L + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - b / 200;
  const inv = (f) => (f ** 3 > EPS ? f ** 3 : (116 * f - 16) / KAPPA);
  const yr = L > KAPPA * EPS ? fy ** 3 : L / KAPPA;
  return [inv(fx) * WHITE[0], yr * WHITE[1], inv(fz) * WHITE[2]];
}

export function xyzToLab([X, Y, Z]) {
  const f = (t) => (t > EPS ? Math.cbrt(t) : (KAPPA * t + 16) / 116);
  const fx = f(X / WHITE[0]);
  const fy = f(Y / WHITE[1]);
  const fz = f(Z / WHITE[2]);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

const gammaEncode = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const gammaDecode = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

// Returns {rgb: [0-255 x3], inGamut: bool}
export function labToRgb(lab) {
  const [X, Y, Z] = labToXyz(lab).map((v) => v / 100);
  const lin = [
    3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
    -0.969266 * X + 1.8760108 * Y + 0.041556 * Z,
    0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z,
  ];
  const tol = 0.002;
  const inGamut = lin.every((c) => c >= -tol && c <= 1 + tol);
  const rgb = lin.map((c) => Math.round(255 * gammaEncode(Math.min(1, Math.max(0, c)))));
  return { rgb, inGamut };
}

export function labToHex(lab) {
  return "#" + labToRgb(lab).rgb.map((v) => v.toString(16).padStart(2, "0")).join("");
}

export function hexToLab(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) throw new Error(`invalid hex colour: ${hex}`);
  const n = parseInt(m[1], 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => gammaDecode(v / 255));
  const X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
  const Y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const Z = 0.0193339 * r + 0.119192 * g + 0.9503041 * b;
  return xyzToLab([X * 100, Y * 100, Z * 100]);
}

const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;

// CIEDE2000, following Sharma, Wu & Dalal (2005).
export function deltaE2000([L1, a1, b1], [L2, a2, b2]) {
  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const Cbar7 = ((C1 + C2) / 2) ** 7;
  const G = 0.5 * (1 - Math.sqrt(Cbar7 / (Cbar7 + 25 ** 7)));
  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);
  const hue = (a, b) => (a === 0 && b === 0 ? 0 : (deg(Math.atan2(b, a)) + 360) % 360);
  const h1p = hue(a1p, b1);
  const h2p = hue(a2p, b2);

  const dLp = L2 - L1;
  const dCp = C2p - C1p;
  const zeroC = C1p * C2p === 0;
  let dhp = h2p - h1p;
  if (dhp > 180) dhp -= 360;
  else if (dhp < -180) dhp += 360;
  if (zeroC) dhp = 0;
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(rad(dhp / 2));

  const Lbarp = (L1 + L2) / 2;
  const Cbarp = (C1p + C2p) / 2;
  let hbarp;
  if (zeroC) hbarp = h1p + h2p;
  else if (Math.abs(h1p - h2p) <= 180) hbarp = (h1p + h2p) / 2;
  else if (h1p + h2p < 360) hbarp = (h1p + h2p + 360) / 2;
  else hbarp = (h1p + h2p - 360) / 2;

  const T =
    1 -
    0.17 * Math.cos(rad(hbarp - 30)) +
    0.24 * Math.cos(rad(2 * hbarp)) +
    0.32 * Math.cos(rad(3 * hbarp + 6)) -
    0.2 * Math.cos(rad(4 * hbarp - 63));
  const SL = 1 + (0.015 * (Lbarp - 50) ** 2) / Math.sqrt(20 + (Lbarp - 50) ** 2);
  const SC = 1 + 0.045 * Cbarp;
  const SH = 1 + 0.015 * Cbarp * T;
  const dTheta = 30 * Math.exp(-(((hbarp - 275) / 25) ** 2));
  const Cbarp7 = Cbarp ** 7;
  const RT = -2 * Math.sqrt(Cbarp7 / (Cbarp7 + 25 ** 7)) * Math.sin(rad(2 * dTheta));

  const tL = dLp / SL;
  const tC = dCp / SC;
  const tH = dHp / SH;
  return Math.sqrt(tL * tL + tC * tC + tH * tH + RT * tC * tH);
}

// Plain-language label for a colour difference, for the weaver view.
export function closenessLabel(de) {
  if (de <= 1) return "แทบเหมือนกัน";
  if (de <= 3) return "ใกล้มาก";
  if (de <= 6) return "ใกล้เคียง";
  return "ต่างกันชัดเจน";
}
