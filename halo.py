#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

# Halo measurements in physical units, as in Mancera Pina et al. (2026, arXiv:2609.25220, Sect. 3.3)

import math
import numpy as np
import torch

from universe import HBAR, G, EV, C, MSUN, KPC, MPC

# Cosmology of Mancera Pina et al.: H0 = 70, Omega_m = 0.3, Omega_L = 0.7
H0 = 70.0
RHO_CRIT = 3 * (H0 * 1e3 / MPC) ** 2 / (8 * math.pi * G) / MSUN * KPC ** 3     # [Msun/kpc^3], z = 0

class Units:
    # Static box with hbar/m = 1 and G = G_code; the code length [kpc] is free
    def __init__(self, m_boson_eV, length_kpc, G_code=1.0):
        m = m_boson_eV * EV / C ** 2
        self.length = length_kpc                                            # [kpc]
        self.time = m * (length_kpc * KPC) ** 2 / HBAR                      # [s]
        self.mass = G_code * (length_kpc * KPC) ** 3 / (G * self.time ** 2) / MSUN   # [Msun]
        self.density = self.mass / length_kpc ** 3                          # [Msun/kpc^3]
        self.velocity = length_kpc * KPC / self.time / 1e3                  # [km/s]

def core_radius_schive(m_boson_eV, M_halo):
    # Schive et al. (2014, PRL 113, 261302) at z = 0 [kpc], with M200 in place of M_vir
    return 1.6 / (m_boson_eV / 1e-22) / (M_halo / 1e9) ** (1 / 3)

# The field around a centre
def periodic_distance(n, L, centre):
    # Distance of each cell from centre, nearest periodic image
    x = (np.arange(n) + 0.5) * L / n - L / 2
    d = [(x - c + L / 2) % L - L / 2 for c in centre]
    return np.sqrt(d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2)

def shells(rho, r, dx, edges):
    # Mass, mean density and mean radius in spherical shells
    which = np.digitize(r.ravel(), edges) - 1
    keep = (which >= 0) & (which < len(edges) - 1)
    bins = len(edges) - 1
    count = np.bincount(which[keep], minlength=bins)
    mass = np.bincount(which[keep], weights=rho.ravel()[keep], minlength=bins) * dx ** 3
    r_mean = np.bincount(which[keep], weights=r.ravel()[keep], minlength=bins) / np.maximum(count, 1)
    return mass, mass / np.maximum(count, 1) / dx ** 3, r_mean

def zoom(psi_k, L, centre, half_width, m=64):
    # psi on an m^3 cube around centre, by Fourier interpolation: psi(x) = (1/n^3) sum_k psi_k exp(i k (x - x_0))
    # Interpolate psi, not rho (rho has twice the bandwidth of the grid)
    n = psi_k.shape[0]
    dx = L / n
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64).to(psi_k.device)
    s = torch.linspace(-half_width, half_width, m, dtype=torch.float64, device=psi_k.device)
    x0 = dx / 2 - L / 2
    E = [torch.exp(1j * (c + s[:, None] - x0) * k[None, :]).to(psi_k.dtype) / n for c in centre]
    out = torch.tensordot(E[0], psi_k, dims=([1], [0]))            # (m, n, n)
    out = torch.einsum("bj,ajk->abk", E[1], out)                    # (m, m, n)
    out = torch.einsum("ck,abk->abc", E[2], out)                    # (m, m, m)
    return out.cpu().numpy(), s.cpu().numpy()

def peak(psi_k, L, centre, m=33):
    # Density maximum near centre, to 1/16 of a cell
    dx = L / psi_k.shape[0]
    cube, s = zoom(psi_k, L, centre, dx, m)
    i = np.unravel_index(np.argmax(np.abs(cube) ** 2), cube.shape)
    return np.array([centre[a] + s[i[a]] for a in range(3)])

def at_radius(psi_k, L, centre, r, m=64):
    # Density and log slope at r (Sect. 3.3): linear fit of log rho against log r over 0.9 r < r' < 1.1 r
    cube, s = zoom(psi_k, L, centre, 1.15 * r, m)
    rho = np.abs(cube).ravel() ** 2
    rr = np.sqrt(s[:, None, None] ** 2 + s[None, :, None] ** 2 + s[None, None, :] ** 2).ravel()
    inside = (rr > 0.9 * r) & (rr < 1.1 * r)
    edges = np.arange(0.9 * r, 1.1 * r + 1e-12 * r, s[1] - s[0])
    which = np.digitize(rr[inside], edges) - 1
    count = np.bincount(which, minlength=len(edges) - 1)
    good = count > 0
    r_bin = np.bincount(which, weights=rr[inside])[good] / count[good]
    rho_bin = np.bincount(which, weights=rho[inside])[good] / count[good]
    slope = np.polyfit(np.log(r_bin), np.log(rho_bin), 1)[0]
    return rho[inside].mean(), slope

def enclosed(rho, r, L):
    # Enclosed mass up to L/2, in shells of a quarter cell
    dx = L / rho.shape[0]
    edges = np.arange(0, L / 2 + 1e-9, dx / 4)
    mass, _, _ = shells(rho, r, dx, edges)
    return edges[1:], np.cumsum(mass)

def r200(r, M, units):
    # R200 [kpc] and M200 [Msun]; nan if beyond the box
    r, M = r[M > 0], M[M > 0]
    mean = M * units.mass / (4 / 3 * math.pi * (r * units.length) ** 3)
    below = np.nonzero(mean < 200 * RHO_CRIT)[0]
    if len(below) == 0 or below[0] == 0:
        return math.nan, math.nan
    i = below[0]
    # Log-log interpolation across the crossing
    t = (math.log(200 * RHO_CRIT) - math.log(mean[i - 1])) / (math.log(mean[i]) - math.log(mean[i - 1]))
    R = math.exp(math.log(r[i - 1]) + t * (math.log(r[i]) - math.log(r[i - 1])))
    return R * units.length, math.exp(math.log(M[i - 1]) + t * (math.log(M[i]) - math.log(M[i - 1]))) * units.mass

def scale_for(m_boson_eV, M200_target, r, M, G_code=1.0, lo=-3.0, hi=3.0, points=600):
    # Code lengths [kpc] in [10^lo, 10^hi] that give M200 = M200_target for this boson mass
    from scipy.optimize import brentq

    def mismatch(log_l):
        _, M200 = r200(r, M, Units(m_boson_eV, 10 ** log_l, G_code))
        return math.log(M200 / M200_target) if M200 > 0 else math.nan

    grid = np.linspace(lo, hi, points)
    values = np.array([mismatch(q) for q in grid])
    roots = []
    for i in range(points - 1):
        if np.isfinite(values[i]) and np.isfinite(values[i + 1]) and values[i] * values[i + 1] < 0:
            roots.append(10 ** brentq(mismatch, grid[i], grid[i + 1], xtol=1e-10))
    return roots

def core_mass_chan(m_boson_eV, M_halo):
    # Chan et al. (2022, MNRAS 511, 943), Eq. 15 at z = 0: (central, low, high) core mass [Msun]
    # Band: their range at M_h = 1e9 Msun, m = 8e-23 eV (3e7 to 1e8 Msun), as fixed factors
    q = m_boson_eV / 8e-23
    alpha = 0.515
    central = 8.0e6 * q ** -1.5 + (M_halo / 10 ** -5.73) ** alpha * q ** (1.5 * (alpha - 1))
    at_1e9 = 8.0e6 + (1e9 / 10 ** -5.73) ** alpha
    return central, central * 3e7 / at_1e9, central * 1e8 / at_1e9

# NFW and coreNFW (Eqs. 2-5): r [kpc], M200 [Msun], rho [Msun/kpc^3]
def R200_of(M200):
    return (3 * M200 / (4 * math.pi * 200 * RHO_CRIT)) ** (1 / 3)

def _g(x):
    return np.log1p(x) - x / (1 + x)

def nfw_mass(r, M200, c):
    return M200 * _g(np.asarray(r) * c / R200_of(M200)) / _g(c)

def nfw_density(r, M200, c):
    rs = R200_of(M200) / c
    x = np.asarray(r) / rs
    return M200 / (4 * math.pi * rs ** 3 * _g(c)) / (x * (1 + x) ** 2)

def corenfw_density(r, M200, c, r_c, n):
    # Eq. 4: rho = f^n rho_NFW + n f^(n-1) (1 - f^2) M_NFW / (4 pi r^2 r_c), f = tanh(r / r_c)
    r = np.asarray(r)
    f = np.tanh(r / r_c)
    return f ** n * nfw_density(r, M200, c) + n * f ** (n - 1) * (1 - f ** 2) * nfw_mass(r, M200, c) / (4 * math.pi * r ** 2 * r_c)

def nfw_reference(r_scale, M200, c):
    # Eqs. 13-14: NFW density and log slope at r_scale
    x = r_scale * c / R200_of(M200)
    return 200 / 3 * RHO_CRIT * c ** 3 / _g(c) / (x * (1 + x) ** 2), -1 - 2 * x / (1 + x)

def fit_corenfw(r, rho, M200, r_c_guess):
    # coreNFW fit with M200 fixed: c200, r_c and 0 <= n <= 1, least squares in log rho
    from scipy.optimize import least_squares
    R = R200_of(M200)

    def residual(q):
        return np.log(corenfw_density(r, M200, 10 ** q[0], 10 ** q[1], q[2])) - np.log(rho)

    lower, upper = [0.0, math.log10(1e-3 * R), 0.0], [2.0, math.log10(R), 1.0]
    best = None
    for c0 in (5.0, 15.0):
        for rc0 in (0.3, 1.0, 3.0):
            for n0 in (0.2, 0.8):
                q0 = np.clip([math.log10(c0), math.log10(rc0 * r_c_guess), n0], np.add(lower, 1e-6), np.subtract(upper, 1e-6))
                fit = least_squares(residual, q0, bounds=(lower, upper))
                if best is None or fit.cost < best.cost:
                    best = fit
    rms = math.sqrt(2 * best.cost / len(r))
    return 10 ** best.x[0], 10 ** best.x[1], best.x[2], rms

def concentration_dj19(M200):
    # Mean c200 of Diemer & Joyce (2019) at z = 0, from colossus; nan without colossus
    try:
        from colossus.cosmology import cosmology
        from colossus.halo import concentration
    except ImportError:
        return math.nan
    cosmology.setCosmology("mp26", dict(flat=True, H0=H0, Om0=0.3, Ob0=0.049, sigma8=0.81, ns=0.965))
    h = H0 / 100
    return float(concentration.concentration(M200 * h, "200c", 0.0, model="diemer19"))

_trapezoid = getattr(np, "trapezoid", None) or np.trapz     # numpy < 2

# Soliton + NFW, no simulation
def soliton_density(r, m_boson_eV, r_c):
    # Soliton [Msun/kpc^3] with half-density radius r_c [kpc]
    from wave import soliton_shape, soliton_profile
    r_half = soliton_shape()[2]
    rho_c = r_half ** 4 / (4 * math.pi) * Units(m_boson_eV, 1.0).density / r_c ** 4
    return soliton_profile(r, rho_c, r_c)

def soliton_radius_for_core_mass(m_boson_eV, M_core):
    # r_c [kpc] of the soliton with M_core [Msun] inside r_c
    x = np.linspace(0, 1, 2001)
    M_at_1kpc = _trapezoid(4 * math.pi * x ** 2 * soliton_density(x, m_boson_eV, 1.0), x)
    return M_at_1kpc / M_core

def soliton_nfw_density(r, m_boson_eV, r_c, M200, c):
    # Soliton inside r_t, NFW outside (Schive et al. 2014; Mocz et al. 2017); r_t = nan if no core shows
    grid = np.geomspace(r_c, 50 * r_c, 4000)
    above = np.nonzero(soliton_density(grid, m_boson_eV, r_c) >= nfw_density(grid, M200, c))[0]
    r_t = grid[above[-1]] if len(above) else math.nan
    r = np.asarray(r, dtype=float)
    return np.where(r < r_t, soliton_density(r, m_boson_eV, r_c), nfw_density(r, M200, c)), r_t

def profile_at_radius(density, r, points=41):
    # Sect. 3.3 on a smooth profile: mean density and log slope over 0.9 r < r' < 1.1 r
    rr = np.linspace(0.9 * r, 1.1 * r, points)
    d = density(rr)
    return _trapezoid(d * rr ** 2, rr) / _trapezoid(rr ** 2, rr), np.polyfit(np.log(rr), np.log(d), 1)[0]

def profile_shells(density, edges, points=64):
    # Volume-averaged density and mean radius in spherical shells between edges
    rho, r_mean = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        rr = np.linspace(a, b, points)
        w = rr ** 2
        rho.append(_trapezoid(density(rr) * w, rr) / _trapezoid(w, rr))
        r_mean.append(_trapezoid(rr * w, rr) / _trapezoid(w, rr))
    return np.array(rho), np.array(r_mean)
