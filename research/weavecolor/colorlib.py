"""Colour conversions and CIEDE2000, vectorised with numpy.

All Lab values are CIE 1976 L*a*b*. Table 4.7 was measured on a Datacolor 800
family instrument (xenon, D65, d/8° geometry). The standard observer (2° or 10°)
is not confirmed; D65/10° is the default, and WHITE_POINTS lets experiments
switch so the sensitivity can be reported (it was < 0.1 ΔE00 in the first run).
"""
import numpy as np

# Reference white XYZ (Y = 100)
WHITE_POINTS = {
    "D65_2": np.array([95.047, 100.0, 108.883]),
    "D65_10": np.array([94.811, 100.0, 107.304]),
}
DEFAULT_WHITE = "D65_10"

_EPS = 216 / 24389
_KAPPA = 24389 / 27


def lab_to_xyz(lab, white=DEFAULT_WHITE):
    lab = np.asarray(lab, dtype=float)
    wp = WHITE_POINTS[white]
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(f):
        f3 = f**3
        return np.where(f3 > _EPS, f3, (116 * f - 16) / _KAPPA)

    xr, zr = finv(fx), finv(fz)
    yr = np.where(L > _KAPPA * _EPS, fy**3, L / _KAPPA)
    return np.stack([xr * wp[0], yr * wp[1], zr * wp[2]], axis=-1)


def xyz_to_lab(xyz, white=DEFAULT_WHITE):
    xyz = np.asarray(xyz, dtype=float)
    r = xyz / WHITE_POINTS[white]
    f = np.where(r > _EPS, np.cbrt(r), (_KAPPA * r + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 following Sharma, Wu & Dalal (2005)."""
    lab1 = np.asarray(lab1, dtype=float)
    lab2 = np.asarray(lab2, dtype=float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cbar7 / (Cbar7 + 25.0**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, h1p)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p
    zero_c = (C1p * C2p) == 0
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(zero_c, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbarp = np.where(
        np.abs(h1p - h2p) <= 180,
        hsum / 2,
        np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
    )
    hbarp = np.where(zero_c, hsum, hbarp)

    T = (
        1
        - 0.17 * np.cos(np.radians(hbarp - 30))
        + 0.24 * np.cos(np.radians(2 * hbarp))
        + 0.32 * np.cos(np.radians(3 * hbarp + 6))
        - 0.20 * np.cos(np.radians(4 * hbarp - 63))
    )
    SL = 1 + 0.015 * (Lbarp - 50) ** 2 / np.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    dtheta = 30 * np.exp(-(((hbarp - 275) / 25) ** 2))
    Cbarp7 = Cbarp**7
    RT = -2 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0**7)) * np.sin(np.radians(2 * dtheta))

    tL = dLp / (kL * SL)
    tC = dCp / (kC * SC)
    tH = dHp / (kH * SH)
    return np.sqrt(tL**2 + tC**2 + tH**2 + RT * tC * tH)
