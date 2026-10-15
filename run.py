#!/usr/bin/env python3
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import os
import sys
import json
import time
import math

import numpy as np
import torch

from universe import Universe, load_power_spectrum, initial_wave, KPC
import halo
from wave import Box, blobs, cloud, ground_state, radial_profile, fit_core, soliton_profile, power_spectrum, point_mass, pull, separation

HERE = os.path.dirname(os.path.abspath(__file__))

COSMO = dict(
    H0=70.0, Omega_m=0.27, Omega_L=0.73,     # background cosmology (flat)
    m_boson_eV=8.0e-23,                      # boson mass
    L_Mpc=4.0,                               # comoving box side
    n=128,                                   # cells per side
    z_start=1000.0, z_end=10.0,              # redshift range
    pk_file="data/psidm_pk_z1000.txt",       # linear P(k) at z_start: k [1/Mpc], P [Mpc^3]
    seed=1,
    c_kin=0.265, c_pot=1.885,                # time step: dt <= c_kin dx^2 and dt <= c_pot / (a max|V|) (Schive et al. 2014)
    dlna_max=0.1,                            # ... and ln a changes by at most this per step
    snapshot_every=50,                       # steps between saved snapshots
    device="auto", dtype="complex64",        # auto: the GPU (mps or cuda) if there is one
    out="results/cosmo",
)

SOLITON = dict(
    n=128, L=30.0, G=1.0,                    # static box, code units with hbar = m = 1
    blobs=8, blob_mass=0.6, blob_width=1.5, spread=5.0, seed=3,
    t_end=120.0, fits=40,                    # run time, and how many times the core is fitted
    c_kin=0.25, c_pot=0.5,
    ground_n=64, ground_L=10.0, ground_mass=3.0,   # the reference ground state
    device="cpu", dtype="complex128",        # double precision: the mass is conserved to 1e-12
    out="results/soliton",
)

BLACKHOLE = dict(
    n=80, L=10.0, G=1.0,                     # static box, code units with hbar = m = 1
    mass=3.0,                                # soliton mass (core radius ~0.9 for G = 1)
    softening=2.0,                           # black hole potential softened over this many cells (1 conserves momentum poorly)
    ratios="0,0.1,0.25,0.5,1",               # M_bh / M_soliton for the ground states
    tolerance=1e-7,                          # ground state: stop when mu changes less than this per 100 steps
    sink_ratio=0.1,                          # M_bh / M_soliton for the black hole that sinks
    sink_offset=1.0,                         # its starting distance from the centre, released at rest
    t_end=40.0, frames=400,                  # run time of the sinking black hole, and how many records
    c_kin=0.25, c_pot=0.5,
    device="cpu", dtype="complex128",
    out="results/blackhole",
)

CORENFW = dict(
    G=1.0, L=24.0,                           # static box, code units with hbar = m = 1
    shape="iso", speeds="flat", cloud_mass=16.0, cloud_radius=8.0, cloud_core=0.5, concentration=10.0, seed=1,   # wave.cloud
    resolutions="128,160",                   # the same halo on two grids: a measurement counts only if both agree
    t_end=320.0, samples=8,                  # measured at `samples` times from t_end/2 to t_end
    masses="8e-23,1e-22,2e-22,4e-22,1e-21",  # boson masses [eV]
    halo_masses="1e9,3e9,1e10",              # M200 [Msun]: each (boson mass, M200) is one scaling of the same run
    c_kin=0.25, c_pot=0.5,
    checkpoint_min=10.0,                     # wall minutes between saves of the running state (a stopped run resumes)
    device="cpu", dtype="complex128",
    out="results/corenfw/iso",
)

SOLITON_NFW = dict(
    masses="8e-23,1e-22,1.5e-22,2e-22,3e-22,4e-22,6e-22,1e-21,2e-21,5e-21,1e-20,3e-20,1e-19",   # boson masses [eV]
    halo_masses="5e8,1e9,2e9",               # M200 [Msun]
    fit_from=3.2,                            # coreNFW fit from r_c / fit_from to R200 (SPmini at 640^3: r_sol/dx = 3.2)
    validate="results/corenfw/iso_hpc/summary_320_384.txt:384,results/corenfw/heavy_M300/summary_512_640.txt:640",
    out="results/solitonnfw",
)

FIGURES = dict(
    M200=1e9,                                # halo mass [Msun] of the figure
    seeds="results/corenfw/iso_hpc/summary.txt:384,results/corenfw/iso_seed2/summary.txt:384,"
          "results/corenfw/iso_seed3/summary.txt:384,results/corenfw/iso_seed4/summary.txt:384",   # one SPmini halo, 4 seeds
    heavy="results/corenfw/heavy_M300/summary_512_640.txt:640",   # the heavier SPmini halo (one seed)
    pina="data/pavel",                       # comparison sample of Mancera Pina et al. (2026), not included
    profile="results/corenfw/iso_hpc/n384:1e-22:1e9",   # the SPmini halo whose profile is shown: run folder, m [eV], M200 [Msun]
    pina_logM_max=10.0,                      # Piña's objects below this log M200 set the CDM + feedback range
    m_allowed=2e-20,                         # [eV] lower bound on m from the Lyman-alpha forest (Rogers & Peiris 2021)
    out="results/figures",
)

# Agreement between the two grids for a measurement to count (time-averaged values)
TOLERANCE = dict(log_rho300=0.05, alpha300=0.05, log_rho015=0.05, alpha015=0.05, log_rc=0.1, n=0.1)

# Command line
USAGE = """usage:
    python run.py cosmo        a cosmological box from z_start to z_end   -> results/cosmo/
    python run.py soliton      blobs collapse into a soliton               -> results/soliton/
    python run.py blackhole    a black hole in a soliton                   -> results/blackhole/
    python run.py corenfw      a halo measured like the paper              -> results/corenfw/
    python run.py solitonnfw   soliton + NFW computed directly, measured the same way -> results/solitonnfw/
    python run.py figures      figures for the draft: SPmini, soliton + NFW, Piña's sample -> results/figures/
    python run.py check       tests of the physics against exact answers

change any setting from the command line, e.g.   python run.py soliton t_end=60 blobs=12"""

def device_of(settings):
    name = settings["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and settings["dtype"] == "complex128":
        sys.exit("the Apple GPU (mps) has no double precision: use dtype=complex64 or device=cpu")
    return torch.device(name)

def outdir(settings):
    path = os.path.join(HERE, settings["out"])
    os.makedirs(path, exist_ok=True)
    return path

# Cosmological box
def cosmo(s):
    out = outdir(s)
    log = open(os.path.join(out, "log.txt"), "w")

    def say(line):
        print(line, flush=True)
        log.write(line + "\n")

    universe = Universe(s["H0"], s["Omega_m"], s["Omega_L"], s["m_boson_eV"])
    a, a_end = 1.0 / (1.0 + s["z_start"]), 1.0 / (1.0 + s["z_end"])
    P_of_k = load_power_spectrum(os.path.join(HERE, s["pk_file"]))
    psi = initial_wave(s["n"], s["L_Mpc"], universe, P_of_k, a, s["seed"])
    device = device_of(s)
    box = Box(torch.tensor(psi, dtype=getattr(torch, s["dtype"]), device=device), universe.code_length(s["L_Mpc"]))

    cell_kpc = s["L_Mpc"] * 1e3 / s["n"]
    say(f"box {s['L_Mpc']} Mpc, {s['n']}^3 cells of {cell_kpc:.1f} kpc (comoving), m = {s['m_boson_eV']:.1e} eV, "
        f"code length {universe.L0 / KPC:.2f} kpc, device {device}")
    save_snapshot(box, a, 0, out, s["L_Mpc"])
    k, P_start = power_spectrum(box.density().cpu().numpy(), s["L_Mpc"])

    step, tau, t0 = 0, 0.0, time.time()
    while a < a_end:
        rate = universe.da_dtau(a)
        dt = min(box.dt_max(a, s["c_kin"], s["c_pot"]), s["dlna_max"] * a / rate)
        if a + rate * dt > a_end:                          # do not overshoot z_end
            dt = (a_end - a) / rate
        a_mid = universe.advance(a, dt / 2)                # the kick uses a at mid-step: second order in dt
        box.step(dt, a_mid)
        a = universe.advance(a, dt)
        tau, step = tau + dt, step + 1
        if step % s["snapshot_every"] == 0 or a >= a_end:
            say(f"step {step:6d}   z = {1 / a - 1:8.3f}   max density {box.density().max().item():9.2f}   "
                f"under-resolved {under_resolved(box):5.1%}   wall {time.time() - t0:7.1f} s")
            save_snapshot(box, a, step, out, s["L_Mpc"])

    k, P_end = power_spectrum(box.density().cpu().numpy(), s["L_Mpc"])
    np.savetxt(os.path.join(out, "power_spectrum.txt"), np.column_stack([k, P_start, P_end]),
               header=f"k [1/Mpc]   P(k) [Mpc^3] at z = {s['z_start']:g}   and at z = {1 / a - 1:g}")
    plot_cosmo(out, k, P_start, P_end, s, a)
    say(f"done: {step} steps; results in {out}")

# Unresolved regions
def under_resolved(box):
    """Fraction of cells where the wave turns by more than pi/2 between neighbours:
    i.e. the de Broglie wavelength is shorter than 4 cells.

    The phase jump from cell i to i+1 is the angle of psi_{i+1} conj(psi_i)
    .: a jump of pi means 2 cells per wavelength, where the grid gives up.
    """
    worst = torch.zeros_like(box.psi.real)
    for axis in range(3):
        turn = torch.roll(box.psi, -1, axis) * box.psi.conj()
        worst = torch.maximum(worst, torch.atan2(turn.imag, turn.real).abs())
    return (worst > math.pi / 2).float().mean().item()

# Soliton formation
def soliton(s):
    """Blobs with random phases collapse, the core that forms should be the ground state.

    1.- All solitons are one shape rescaled (psi(x) -> lambda^2 psi(lambda x))
    2.- rho_c r_c^4 is the same number for every soliton
    3.- I compute it once from the ground state
    4.- Then check that the core that forms by itself has r_c = (that number / rho_c)^(1/4), with no free parameter.
    """
    out = outdir(s)
    device = device_of(s)
    print("reference ground state ...", flush=True)
    ref, converged = ground_state(s["ground_n"], s["ground_L"], s["G"], s["ground_mass"])
    r, rho = radial_profile(ref.density().numpy(), ref.dx, s["ground_L"] / 2)
    rho_c, r_c = fit_core(r, rho)
    invariant = rho_c * r_c ** 4
    print(f"  converged {converged}, r_c = {r_c:.3f} ({r_c / ref.dx:.1f} cells), rho_c r_c^4 = {invariant:.4e}")

    psi = blobs(s["n"], s["L"], s["blobs"], s["blob_mass"], s["blob_width"], s["spread"], s["seed"])
    box = Box(torch.tensor(psi, dtype=getattr(torch, s["dtype"]), device=device), s["L"], 4 * math.pi * s["G"])
    M0, E0 = box.mass(), box.energy()
    t, rows = 0.0, []
    for t_fit in np.linspace(0, s["t_end"], s["fits"] + 1)[1:]:
        while t < t_fit - 1e-12:
            dt = min(box.dt_max(1.0, s["c_kin"], s["c_pot"]), t_fit - t)
            box.step(dt)
            t += dt
        r, rho = radial_profile(box.density().cpu().numpy(), box.dx, s["L"] / 2)
        rho_c, r_c = fit_core(r, rho)
        predicted = (invariant / rho_c) ** 0.25
        rows.append((t, box.mass() / M0 - 1, box.energy() / E0 - 1, rho_c, r_c, predicted))
        print(f"  t = {t:6.1f}   mass drift {rows[-1][1]:+.1e}   energy drift {rows[-1][2]:+.1e}   "
              f"r_c = {r_c:.3f} ({r_c / box.dx:.1f} cells)   predicted {predicted:.3f}   ratio {r_c / predicted:.3f}",
              flush=True)
    rows = np.array(rows)
    np.savetxt(os.path.join(out, "cores.txt"), rows, header="t  mass_drift  energy_drift  rho_c  r_c  r_c_predicted")
    late = rows[rows[:, 0] >= s["t_end"] / 2]
    print(f"late times: r_c / predicted = {np.mean(late[:, 4] / late[:, 5]):.3f} +- {np.std(late[:, 4] / late[:, 5]):.3f}")
    plot_soliton(out, rows, box, r, rho, rho_c, r_c)

# Black hole in a soliton
def blackhole(s):
    """A (Newtonian) point mass in a soliton, softened over a couple of cells.

    1.- Ground states with the black hole at the centre, as M_bh / M_soliton grows: the core contracts. When the black
        hole dominates, the field is the hydrogen-like ground state of -G M_bh / r, with Bohr radius a0 = 1 / (G M_bh)
        (hbar = m = 1) and half its mass inside 1.337 a0.
    2.- A black hole released at rest off-centre: it falls through the soliton, which recoils (the total momentum
        stays zero). Does it sink to the centre, and how fast?
    Physics at the horizon (accretion, superradiance) needs general relativity: out of these equations.
    """
    out = outdir(s)
    device = device_of(s)
    G, M = s["G"], s["mass"]
    eps = s["softening"] * s["L"] / s["n"]
    ratios = [float(q) for q in s["ratios"].split(",")]

    # 1. Ground states with the black hole at the centre
    profiles, rows = [], []
    for q in ratios:
        t0 = time.time()
        box, converged = ground_state(s["n"], s["L"], G, M, tolerance=s["tolerance"],
                                      external=lambda b: point_mass(b, G, q * M, (0, 0, 0), eps))
        rho = box.density().numpy()
        dx, dy, dz = separation(box, (0, 0, 0))
        r = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2).numpy().ravel()
        order = np.argsort(r)
        r_half = r[order][np.searchsorted(np.cumsum(rho.ravel()[order]) * box.dx ** 3, M / 2)]
        rr, prof = radial_profile(rho, box.dx, s["L"] / 2)
        profiles.append((q, rr, prof))
        a0 = 1 / (G * q * M) if q > 0 else math.inf
        rows.append((q, rho.max(), r_half, a0))
        print(f"  M_bh / M = {q:5.2f}: converged {converged}, peak density {rho.max():8.3f}, half-mass radius {r_half:.3f}"
              f"{f', Bohr radius {a0:.3f} ({a0 / box.dx:.1f} cells)' if q > 0 else ''}   [{time.time() - t0:.0f} s]", flush=True)
        if q == 0:
            soliton = box
    rows = np.array(rows)
    np.savetxt(os.path.join(out, "ground_states.txt"), rows, header="M_bh/M  peak_density  half_mass_radius  bohr_radius")

    # 2. A black hole released at rest off-centre, both advanced kick-drift-kick
    box = Box(soliton.psi.to(getattr(torch, s["dtype"])).to(device), s["L"], 4 * math.pi * G)
    M_bh = s["sink_ratio"] * M
    x_bh, v_bh = np.array([s["sink_offset"], 0.0, 0.0]), np.zeros(3)
    box.V_ext = point_mass(box, G, M_bh, x_bh, eps)
    acc = pull(box, G, x_bh, eps)
    energy = lambda: box.energy() + 0.5 * M_bh * v_bh @ v_bh
    E0, P0 = energy(), box.momentum() + M_bh * v_bh
    print(f"sinking: M_bh = {M_bh:.3f} ({s['sink_ratio']} of the soliton), released at rest at x = {s['sink_offset']}", flush=True)

    t, track = 0.0, []
    for t_frame in np.linspace(0, s["t_end"], s["frames"] + 1):
        while t < t_frame - 1e-12:
            dt = min(box.dt_max(1.0, s["c_kin"], s["c_pot"]), t_frame - t)
            v_bh += acc * dt / 2
            x_bh = (x_bh + v_bh * dt + s["L"] / 2) % s["L"] - s["L"] / 2
            box.step(dt, V_ext_end=point_mass(box, G, M_bh, x_bh, eps))
            acc = pull(box, G, x_bh, eps)
            v_bh += acc * dt / 2
            t += dt
        centre = core_centre(box)
        d = (x_bh - centre + s["L"] / 2) % s["L"] - s["L"] / 2
        track.append((t, *x_bh, *centre, np.linalg.norm(d)))
        if (len(track) - 1) % max(s["frames"] // 20, 1) == 0:
            P = box.momentum() + M_bh * v_bh
            print(f"  t = {t:6.2f}   distance to the core {np.linalg.norm(d):.3f}   speed {np.linalg.norm(v_bh):.3f}   "
                  f"energy drift {energy() / E0 - 1:+.1e}   momentum {np.linalg.norm(P - P0):.1e} (of {M_bh * np.linalg.norm(v_bh):.1e})",
                  flush=True)
    track = np.array(track)
    np.savetxt(os.path.join(out, "sinking.txt"), track, header="t  x_bh y_bh z_bh  x_core y_core z_core  distance")
    plot_blackhole(out, s, profiles, rows, track, box, x_bh)
    print(f"figures in {out}")

def core_centre(box, radius=1.0, iterations=20):
    # Centre of the core: the centre of mass of the field under a Gaussian window of this radius, starting from the
    # densest cell and re-centring the window until it settles (a smooth window, since a sharp sphere on the grid can
    # lock off-centre). A mean over the whole periodic box drifts as the tail and the matter thrown out wrap around
    # the edges, even with the momentum conserved
    rho = box.density().cpu().double()
    peak = np.unravel_index(torch.argmax(rho).item(), rho.shape)
    centre = (np.array(peak) + 0.5) * box.dx - box.L / 2
    for _ in range(iterations):
        d = [q.cpu().double() for q in separation(box, centre)]
        weight = rho * torch.exp(-(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / (2 * radius ** 2))
        offset = np.array([(weight * q).sum().item() for q in d]) / weight.sum().item()
        centre = (centre + offset + box.L / 2) % box.L - box.L / 2
    return centre

def softened_half_radius(GM, softening, N=20000):
    # Half-mass radius of the ground state of -GM / sqrt(r^2 + eps^2) alone (no self-gravity), by a 1D radial solve
    from scipy.linalg import eigh_tridiagonal
    h = (20 / GM + 5 * softening) / N
    r = h * np.arange(1, N + 1)
    u = eigh_tridiagonal(1 / h ** 2 - GM / np.sqrt(r ** 2 + softening ** 2), -0.5 / h ** 2 * np.ones(N - 1),
                         select="i", select_range=(0, 0))[1][:, 0]
    return r[np.searchsorted(np.cumsum(u ** 2) / np.sum(u ** 2), 0.5)]

# The soliton halo measured like the paper
def corenfw(s):
    """Halo grown from wave.cloud, measured like Mancera Pina et al. (2026).

    1.- Simulate the cloud on each grid, save psi at `samples` late times
    2.- For each (m, M200) find the code length that gives that M200
    3.- Measure rho and slope at 300 pc and 0.015 R200, the coreNFW fit and the NFW reference
    4.- Average over time, compare the grids (TOLERANCE)
    """
    out = outdir(s)
    pairs = [(float(m), float(M)) for m in s["masses"].split(",") for M in s["halo_masses"].split(",")]
    resolutions = [int(q) for q in s["resolutions"].split(",")]
    times = np.linspace(s["t_end"] / 2, s["t_end"], s["samples"])
    summary = {}
    for n in resolutions:
        files = simulate_halo(s, n, times, out)
        rows = []
        for t, f in zip(times, files):
            t0 = time.time()
            last = measure_halo(np.load(f)["psi"], s["L"], s["G"], pairs, t)
            rows += last
            print(f"  n = {n}, t = {t:6.1f} measured [{time.time() - t0:.0f} s]", flush=True)
        names = list(rows[0])
        table = np.array([[row[k] for k in names] for row in rows])
        np.savetxt(os.path.join(out, f"n{n}", "measurements.txt"), table, header="  ".join(names))
        summary[n] = {p: average([row for row in rows if (row["m_eV"], row["M_target"]) == p]) for p in pairs}
        plot_corenfw(out, n, np.load(files[-1])["psi"], s, last)
    report_corenfw(out, summary, resolutions, pairs)

def simulate_halo(s, n, times, out):
    # psi at the sample times; reuses a finished run, resumes a stopped one
    folder = os.path.join(out, f"n{n}")
    os.makedirs(folder, exist_ok=True)
    keys = ("G", "L", "shape", "speeds", "cloud_mass", "cloud_radius", "cloud_core", "concentration", "seed", "t_end", "samples",
            "c_kin", "c_pot", "dtype")
    setup = {k: s[k] for k in keys} | {"n": n}
    files = [os.path.join(folder, f"psi_t{t:07.2f}.npz") for t in times]
    stamp, state = os.path.join(folder, "settings.json"), os.path.join(folder, "checkpoint.npz")
    if os.path.exists(stamp):
        if json.load(open(stamp)) != setup:
            sys.exit(f"{folder} holds a run with other settings: use another out= or delete it")
        if all(os.path.exists(f) for f in files):
            print(f"n = {n}: using the saved run in {folder}", flush=True)
            return files
    json.dump(setup, open(stamp, "w"), indent=1)

    dtype, device = getattr(torch, s["dtype"]), device_of(s)
    if os.path.exists(state):
        saved = np.load(state)
        box = Box(torch.tensor(saved["psi"], dtype=dtype, device=device), s["L"], 4 * math.pi * s["G"])
        t, M0, E0 = float(saved["t"]), float(saved["M0"]), float(saved["E0"])
        fix = float(saved["fix"]) if "fix" in saved else 1.0
        print(f"n = {n}: resuming at t = {t:.2f} of {s['t_end']}", flush=True)
    else:
        psi, sigma = cloud(n, s["L"], s["cloud_mass"], s["cloud_radius"], s["cloud_core"], s["shape"],
                           s["concentration"], s["G"], s["seed"], s["speeds"])
        box = Box(torch.tensor(psi, dtype=dtype, device=device), s["L"], 4 * math.pi * s["G"])
        t, M0, E0, fix = 0.0, box.mass(), box.energy(), 1.0
        print(f"n = {n}: {s['shape']} cloud ({s['speeds']} speeds), mass {s['cloud_mass']}, largest speed dispersion {sigma:.3f} "
              f"(shortest wave about {2 * math.pi / (3 * sigma) / box.dx:.1f} cells), under-resolved {under_resolved(box):.1%}; "
              f"simulating to t = {s['t_end']}", flush=True)

    def report(tag):
        print(f"  t = {t:7.2f} {tag:10s} mass drift {box.mass() / M0 - 1:+.1e} (mass fix {fix - 1:+.1e})   energy drift {box.energy() / E0 - 1:+.1e}   "
              f"under-resolved {under_resolved(box):5.1%}   peak density {box.density().max().item():8.3f}   "
              f"wall {time.time() - t0:7.0f} s", flush=True)

    t0 = last_save = time.time()
    single, steps = s["dtype"] == "complex64", 0
    for t_save, f in zip(times, files):
        if os.path.exists(f) and t >= t_save - 1e-9:
            continue
        while t < t_save - 1e-12:
            dt = min(box.dt_max(1.0, s["c_kin"], s["c_pot"]), t_save - t)
            box.step(dt)
            t += dt
            steps += 1
            if single and steps % 100 == 0:
                # complex64: rounding drifts the mass, rescale psi to M0
                factor = math.sqrt(M0 / box.mass())
                box.psi = box.psi * factor
                box.V = box.potential(box.density())
                fix *= factor ** 2
            if time.time() - last_save > 60 * s["checkpoint_min"]:
                np.savez(state, psi=box.psi.cpu().numpy(), t=t, M0=M0, E0=E0, fix=fix)
                last_save = time.time()
                report("checkpoint")
        np.savez(f, psi=box.psi.cpu().numpy(), t=t)
        np.savez(state, psi=box.psi.cpu().numpy(), t=t, M0=M0, E0=E0, fix=fix)
        last_save = time.time()
        report("sample")
    return files

def measure_halo(psi, L, G, pairs, t):
    # One row per (m, M200); kpc, Msun, Msun/kpc^3
    n = psi.shape[0]
    dx = L / n
    rho = np.abs(psi) ** 2
    psi_t = torch.tensor(psi)
    psi_k = torch.fft.fftn(psi_t)
    cell = np.unravel_index(np.argmax(rho), rho.shape)
    centre = halo.peak(psi_k, L, (np.array(cell) + 0.5) * dx - L / 2)
    r_cells = halo.periodic_distance(n, L, centre)
    r_enc, M_enc = halo.enclosed(rho, r_cells, L)
    r_c_code = fit_core(*radial_profile(rho, dx, L / 2))[1]
    Mc_code = np.interp(r_c_code, r_enc, M_enc)
    under = under_resolved(Box(psi_t, L))

    rows = []
    for m, M_target in pairs:
        row = dict(t=t, m_eV=m, M_target=M_target, n_grid=n, under_resolved=under, rc_code=r_c_code, found=0)
        roots = halo.scale_for(m, M_target, r_enc, M_enc, G)
        if not roots:                           # R200 does not fall inside the box
            rows.append(row | {k: math.nan for k in ROW})
            continue
        row["found"] = len(roots)
        units = halo.Units(m, roots[0], G)
        R200, M200 = halo.r200(r_enc, M_enc, units)
        to_code = 1 / units.length
        Mc = Mc_code * units.mass
        Mc_chan, Mc_low, Mc_high = halo.core_mass_chan(m, M200)
        row |= dict(length=units.length, M200=M200, R200=R200, box_R200=L * units.length / R200,
                    dx_pc=dx * units.length * 1e3, r_sol=r_c_code * units.length, rsol_cells=r_c_code / dx,
                    rsol_schive=r_c_code * units.length / halo.core_radius_schive(m, M200),
                    Mc=Mc, Mc_chan=Mc_chan, Mc_band=float(Mc_low <= Mc <= Mc_high))
        for tag, r_scale in (("300", 0.3), ("015", 0.015 * R200)):
            d, a = halo.at_radius(psi_k, L, centre, r_scale * to_code)
            row[f"rho{tag}"], row[f"alpha{tag}"] = d * units.density, a
        edges = np.geomspace(dx, R200 * to_code, 26)
        _, d, r_bin = halo.shells(rho, r_cells, dx, edges)
        d, r_bin = d[r_bin > 0], r_bin[r_bin > 0]
        c, r_c, n_core, rms = halo.fit_corenfw(r_bin * units.length, d * units.density, M200, row["r_sol"])
        row |= dict(c200=c, rc=r_c, n=n_core, fit_rms=rms)
        _, d, r_bin = halo.shells(rho, r_cells, dx, np.geomspace(0.3, 1.0, 9) * R200 * to_code)
        row["alpha_out"] = np.polyfit(np.log(r_bin), np.log(d), 1)[0]
        c_ref = halo.concentration_dj19(M200)
        row["c_dj19"] = c_ref
        for tag, r_scale in (("300", 0.3), ("015", 0.015 * R200)):
            row[f"rho{tag}_cdm"], row[f"alpha{tag}_cdm"] = halo.nfw_reference(r_scale, M200, c_ref)
        rows.append(row)
    return rows

# The columns of a measured row, after t, m_eV, M_target, n_grid, under_resolved, rc_code, found
ROW = ("length", "M200", "R200", "box_R200", "dx_pc", "r_sol", "rsol_cells", "rsol_schive", "Mc", "Mc_chan", "Mc_band",
       "rho300", "alpha300", "rho015", "alpha015", "c200", "rc", "n", "fit_rms", "alpha_out", "c_dj19",
       "rho300_cdm", "alpha300_cdm", "rho015_cdm", "alpha015_cdm")

def average(rows):
    # Mean and scatter over the time samples; masses, densities and radii in log10
    logs = {"M200", "R200", "r_sol", "Mc", "Mc_chan", "rho300", "rho015", "rc", "rho300_cdm", "rho015_cdm"}
    result = {}
    for k in rows[0]:
        v = np.array([row[k] for row in rows], dtype=float)
        v = np.log10(v) if k in logs else v
        result[k] = (np.mean(v), np.std(v))
    return result

def report_corenfw(out, summary, resolutions, pairs):
    # Time averages per grid, and convergence between the first two grids
    shown = [("box_R200", "box/R200", 1), ("r_sol", "log r_sol", 2), ("rsol_cells", "r_sol/dx", 1),
             ("Mc", "log Mc", 2), ("Mc_chan", "log Mc Chan", 2), ("Mc_band", "in band", 2),
             ("rho300", "log rho300", 2), ("alpha300", "a300", 2), ("rho015", "log rho015", 2), ("alpha015", "a015", 2),
             ("rc", "log rc", 2), ("n", "n", 2), ("c200", "c200", 1), ("alpha_out", "a_out", 2),
             ("rho300_cdm", "CDM rho300", 2), ("alpha300_cdm", "CDM a300", 2),
             ("rho015_cdm", "CDM rho015", 2), ("alpha015_cdm", "CDM a015", 2)]
    lines = ["corenfw: time averages (mean, scatter over the samples in brackets); kpc, Msun, Msun/kpc^3",
             "in band: fraction of samples whose core mass lies in the cosmological range (Chan et al. 2022)", ""]
    for n in resolutions:
        lines.append(f"grid {n}^3")
        lines.append("  m [eV]   M200    " + "".join(f"{label:>15s}" for _, label, _ in shown))
        for p in pairs:
            a = summary[n][p]
            lines.append(f"  {p[0]:7.1e}  {p[1]:7.1e}" + "".join(
                f"{a[k][0]:>8.{d}f} ({a[k][1]:.{d}f})" if np.isfinite(a[k][0]) else f"{'beyond box':>15s}" for k, _, d in shown))
        lines.append("")
    if len(resolutions) > 1:
        lo, hi = resolutions[:2]
        lines.append(f"convergence {lo}^3 -> {hi}^3 (difference of the time averages; tolerance in brackets)")
        keys = dict(log_rho300="rho300", alpha300="alpha300", log_rho015="rho015", alpha015="alpha015", log_rc="rc", n="n")
        for p in pairs:
            diffs = {q: summary[hi][p][k][0] - summary[lo][p][k][0] for q, k in keys.items()}
            ok = all(abs(diffs[q]) <= TOLERANCE[q] for q in keys)
            lines.append(f"  {p[0]:7.1e}  {p[1]:7.1e}  " + "  ".join(f"{q} {diffs[q]:+.3f} ({TOLERANCE[q]})" for q in keys)
                         + f"   -> {'CONVERGED' if ok else 'not converged'}")
    text = "\n".join(lines)
    print("\n" + text)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")

def plot_corenfw(out, n, psi, s, rows):
    # Profiles at the last sample, with the coreNFW fit and NFW
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle
    plotstyle.use()
    L, G = s["L"], s["G"]
    masses = [float(q) for q in s["masses"].split(",")]
    halo_masses = [float(q) for q in s["halo_masses"].split(",")]
    dx = L / n
    rho = np.abs(psi) ** 2
    cell = np.unravel_index(np.argmax(rho), rho.shape)
    centre = halo.peak(torch.fft.fftn(torch.tensor(psi)), L, (np.array(cell) + 0.5) * dx - L / 2)
    r_cells = halo.periodic_distance(n, L, centre)
    _, d_code, r_code = halo.shells(rho, r_cells, dx, np.geomspace(dx / 2, L / 2, 40))
    d_code, r_code = d_code[r_code > 0], r_code[r_code > 0]
    fig, axes = plt.subplots(len(halo_masses), len(masses), figsize=(3.6 * len(masses), 3.2 * len(halo_masses)),
                             squeeze=False)
    for row in rows:
        ax = axes[halo_masses.index(row["M_target"])][masses.index(row["m_eV"])]
        ax.set_title(f"m = {row['m_eV']:.0e} eV, M200 = {row['M_target']:.0e}", fontsize=9)
        if not row["found"]:
            ax.text(0.5, 0.5, "R200 beyond the box", ha="center", transform=ax.transAxes)
            continue
        r_kpc = r_code * row["length"]
        ax.loglog(r_kpc, d_code * halo.Units(row["m_eV"], row["length"], G).density, "o", ms=2.5, label="SP mini")
        rr = np.geomspace(r_kpc[0], row["R200"], 200)
        ax.loglog(rr, halo.corenfw_density(rr, row["M200"], row["c200"], row["rc"], row["n"]),
                  label=f"coreNFW n={row['n']:.2f}, r_c={row['rc']:.2f}")
        if np.isfinite(row["c_dj19"]):
            ax.loglog(rr, halo.nfw_density(rr, row["M200"], row["c_dj19"]), "--", label="NFW, DJ19 c")
        for r_scale in (0.3, 0.015 * row["R200"]):
            ax.axvline(r_scale, lw=0.8, color="0.6")
        ax.axvline(row["R200"], lw=0.8, color="0.3")
        ax.legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("r [kpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\rho$ [M$_\odot$/kpc$^3$]")
    plt.savefig(os.path.join(out, f"n{n}", "profiles.png"), dpi=110, bbox_inches="tight")
    plt.close()

# Soliton + NFW computed directly
SHOWN = ("box_R200", "r_sol", "rsol_cells", "Mc", "Mc_chan", "Mc_band", "rho300", "alpha300", "rho015", "alpha015",
         "rc", "n", "c200", "alpha_out", "rho300_cdm", "alpha300_cdm", "rho015_cdm", "alpha015_cdm")

def read_summary(path, n):
    # Time averages of one grid from a corenfw summary: {(m, M200): {column: value}}
    import re
    rows, inside = {}, False
    for line in open(os.path.join(HERE, path)):
        if line.startswith("grid "):
            inside = line.split()[1] == f"{n}^3"
        elif inside and line.strip() and not line.lstrip().startswith("m [eV]"):
            words = line.split()
            values = re.findall(r"(-?[0-9.]+(?:e[+-]?[0-9]+)?) \(", line[19:])
            if len(values) == len(SHOWN):
                rows[(float(words[0]), float(words[1]))] = dict(zip(SHOWN, map(float, values)))
        elif inside and not line.strip():
            inside = False
    return rows

def measure_profile(density, M200, r_c, r_min):
    # Same measurements on a smooth profile, coreNFW fit from r_min to R200
    R200 = halo.R200_of(M200)
    row = dict(R200=R200)
    for tag, r_scale in (("300", 0.3), ("015", 0.015 * R200)):
        row[f"rho{tag}"], row[f"alpha{tag}"] = halo.profile_at_radius(density, r_scale)
    d, r = halo.profile_shells(density, np.geomspace(r_min, R200, 26))
    row["c200"], row["rc"], row["n"], row["fit_rms"] = halo.fit_corenfw(r, d, M200, r_c)
    return row

def soliton_nfw(s):
    out = outdir(s)
    keys = (("rho300", "log rho300", True), ("alpha300", "a300", False), ("rho015", "log rho015", True),
            ("alpha015", "a015", False), ("rc", "log rc", True), ("n", "n", False))
    lines = ["solitonnfw: soliton + NFW computed directly, measured with Piña's recipe; kpc, Msun, Msun/kpc^3", ""]

    # 1. Validation against the SPmini haloes
    lines.append("validation: soliton of SPmini's core mass on an NFW of SPmini's c200, fit from SPmini's cell size")
    lines.append("  each entry: soliton+NFW / SPmini (difference)")
    worst = {k: 0.0 for k, _, _ in keys}
    for item in s["validate"].split(","):
        path, n = item.rsplit(":", 1)
        lines.append(f"  {path} ({n}^3)")
        for (m, M200), sp in read_summary(path, int(n)).items():
            r_c = halo.soliton_radius_for_core_mass(m, 10 ** sp["Mc"])
            density = lambda r: halo.soliton_nfw_density(r, m, r_c, M200, sp["c200"])[0]
            dx = 10 ** sp["r_sol"] / sp["rsol_cells"]
            row = measure_profile(density, M200, r_c, dx)
            text = []
            for k, label, log in keys:
                a = math.log10(row[k]) if log else row[k]
                d = a - sp[k]
                worst[k] = max(worst[k], abs(d))
                text.append(f"{label} {a:6.2f}/{sp[k]:6.2f} ({d:+.2f})")
            lines.append(f"    {m:7.1e} {M200:7.1e}  r_c {r_c:5.2f}/{10 ** sp['r_sol']:5.2f}  " + "  ".join(text))
    lines.append("  largest |difference|: " + "  ".join(f"{label} {worst[k]:.2f}" for k, label, _ in keys))
    lines.append("")

    # 2. Core masses of Chan et al. (2022), c200 of Diemer & Joyce (2019)
    lines.append("prediction: soliton of Chan et al. (2022) core mass on an NFW with the Diemer & Joyce (2019) c200,")
    lines.append(f"  fit from r_c/{s['fit_from']}; core mass low / central / high of Chan et al.'s range; CDM = Piña's CDM line")
    lines.append("  r_t: where the soliton meets the NFW (central core mass); nan: the soliton lies below the NFW, model does not apply")
    lines.append("  m [eV]   M200     r_c [kpc]          r_t [kpc]  log rho300 (CDM)       a300 (CDM)        "
                 "log rho015 (CDM)       a015 (CDM)        log rc           n")
    rows = []
    for M200 in [float(q) for q in s["halo_masses"].split(",")]:
        c = halo.concentration_dj19(M200)
        cdm = {}
        for tag, r_scale in (("300", 0.3), ("015", 0.015 * halo.R200_of(M200))):
            cdm[f"rho{tag}"], cdm[f"alpha{tag}"] = halo.nfw_reference(r_scale, M200, c)
        for m in [float(q) for q in s["masses"].split(",")]:
            per_band = []
            band = halo.core_mass_chan(m, M200)
            for Mc in (band[1], band[0], band[2]):          # low, central, high core mass
                r_c = halo.soliton_radius_for_core_mass(m, Mc)
                r_t = halo.soliton_nfw_density(r_c, m, r_c, M200, c)[1]
                if np.isfinite(r_t):
                    density = lambda r: halo.soliton_nfw_density(r, m, r_c, M200, c)[0]
                    row = measure_profile(density, M200, r_c, r_c / s["fit_from"])
                else:                                       # no core shows
                    row = {k: math.nan for k in ("R200", "rho300", "alpha300", "rho015", "alpha015", "c200", "rc", "n", "fit_rms")}
                row |= dict(m_eV=m, M200=M200, Mc=Mc, r_sol=r_c, r_t=r_t, c_dj19=c) | {f"{k}_cdm": v for k, v in cdm.items()}
                per_band.append(row)
            rows.append(per_band)
            lo, mid, hi = per_band
            fmt = lambda k, log: "/".join(f"{(math.log10(q[k]) if log else q[k]):.2f}" for q in per_band)
            lines.append(f"  {m:7.1e}  {M200:7.1e}  {fmt('r_sol', False):17s}  {mid['r_t']:6.3f}   "
                         f"{fmt('rho300', True):14s} ({math.log10(cdm['rho300']):.2f})  {fmt('alpha300', False):17s} ({cdm['alpha300']:.2f})  "
                         f"{fmt('rho015', True):14s} ({math.log10(cdm['rho015']):.2f})  {fmt('alpha015', False):17s} ({cdm['alpha015']:.2f})  "
                         f"{fmt('rc', True):17s}  {fmt('n', False)}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
    json.dump(rows, open(os.path.join(out, "rows.json"), "w"), indent=1)
    plot_soliton_nfw(out, rows, s)

def plot_soliton_nfw(out, rows, s):
    # Slope and density at 300 pc against boson mass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle
    plotstyle.use()
    spmini = {}
    for item in s["validate"].split(","):
        path, n = item.rsplit(":", 1)
        spmini |= read_summary(path, int(n))
    halo_masses = sorted({r[0]["M200"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, M200 in enumerate(halo_masses):
        color = plotstyle.SERIES[i]
        group = [r for r in rows if r[0]["M200"] == M200]
        m = np.array([r[1]["m_eV"] for r in group])
        for ax, k, log in ((axes[0], "alpha300", False), (axes[1], "rho300", True)):
            v = np.array([[math.log10(q[k]) if log else q[k] for q in r] for r in group])
            ax.fill_between(m, v.min(1), v.max(1), color=color, alpha=0.2, lw=0)
            ax.plot(m, v[:, 1], color=color, label=f"soliton+NFW, M200 = {M200:.0e}")
            cdm = group[0][1][f"{k}_cdm"]
            ax.axhline(math.log10(cdm) if log else cdm, color=color, ls="--", lw=0.8)
            pts = [(p[0], sp[k]) for p, sp in spmini.items() if p[1] == M200]
            if pts:
                ax.plot(*zip(*pts), "o", color=color, ms=4, label=f"SPmini, M200 = {M200:.0e}")
    axes[0].set_ylabel("log slope at 300 pc")
    axes[1].set_ylabel(r"log $\rho$(300 pc) [M$_\odot$/kpc$^3$]")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("boson mass m [eV]")
    axes[0].legend(fontsize=7, title="dashed: Piña's CDM line", title_fontsize=7)
    plt.savefig(os.path.join(out, "slope_density_300pc.png"), dpi=130, bbox_inches="tight")
    plt.close()

# Figures for the draft
QUANTITIES = (("alpha300", "log slope at 300 pc", False), ("rho300", r"log $\rho$(300 pc) [M$_\odot$ kpc$^{-3}$]", True),
              ("alpha015", r"log slope at 0.015 R$_{200}$", False), ("rho015", r"log $\rho$(0.015 R$_{200}$) [M$_\odot$ kpc$^{-3}$]", True))

def soliton_nfw_point(m, M200, c):
    # rho and slope at 300 pc and 0.015 R200 of soliton + NFW, low / central / high core mass
    R200 = halo.R200_of(M200)
    out = []
    for Mc in np.array(halo.core_mass_chan(m, M200))[[1, 0, 2]]:
        r_c = halo.soliton_radius_for_core_mass(m, Mc)
        if not np.isfinite(halo.soliton_nfw_density(r_c, m, r_c, M200, c)[1]):
            out.append({k: math.nan for k, _, _ in QUANTITIES})
            continue
        density = lambda r: halo.soliton_nfw_density(r, m, r_c, M200, c)[0]
        row = {}
        for tag, r_scale in (("300", 0.3), ("015", 0.015 * R200)):
            rho, alpha = halo.profile_at_radius(density, r_scale)
            row[f"rho{tag}"], row[f"alpha{tag}"] = math.log10(rho), alpha
        out.append(row)
    return out

def spmini_points(s):
    # halo 40: (mean, min, max) over the seeds; halo 300: one value
    runs = [read_summary(path, int(n)) for path, n in (item.rsplit(":", 1) for item in s["seeds"].split(","))]
    iso = {}
    for key in runs[0]:
        v = {k: np.array([run[key][k] for run in runs]) for k, _, _ in QUANTITIES}
        iso[key] = {k: (x.mean(), x.min(), x.max()) for k, x in v.items()}
    path, n = s["heavy"].rsplit(":", 1)
    return dict(iso=iso, heavy=read_summary(path, int(n)), seeds=len(runs))

def pina_tables(folder):
    # Sample of Mancera Pina et al. (2026): (median, lo, hi) per quantity, lo/hi to the 16th/84th percentiles
    if not os.path.isdir(os.path.join(HERE, folder)):
        sys.exit(f"{folder} not found: the sample of Mancera Pina et al. (2026) is not distributed here (see README)")
    def read(name):
        lines = [l.split() for l in open(os.path.join(HERE, folder, name)) if l.strip()]
        return [dict(zip(lines[0], l)) for l in lines[1:]]
    cols = (("alpha300", "alpha_300pc"), ("rho300", "rhoDM_300pc"), ("alpha015", "alpha_015R200"), ("rho015", "rhoDM_015R200"))
    n = {r["galaxy"]: (float(r["n"]), float(r["n_err_low"]), float(r["n_err_high"])) for r in read("n_MP26.dat")}
    rows = []
    for r in read("alpha_rho_015r200_M200_MP26.dat"):
        row = dict(name=r["galaxy"], population="gas-poor" if r["flag"] == "gaspoor" else "gas-rich", logM200=float(r["logM200"]), n=n[r["galaxy"]])
        for k, c in cols:
            v, lo, hi = float(r[c]), float(r[c + "_err_low"]), float(r[c + "_err_high"])
            row[k] = (v, lo, hi) if k.startswith("alpha") else (math.log10(v), math.log10(v) - math.log10(v - lo), math.log10(v + hi) - math.log10(v))
        rows.append(row)
    for suite, name in (("FIRE-2", "alpha_rho_015r200_M200_FIRE2_MP26.dat"), ("NIHAO", "alpha_rho_015r200_M200_NIHAO_MP26.dat"),
                        ("EDGE", "alpha_rho_r015200_M200_EDGE_MP26.dat")):
        for r in read(name):
            row = dict(name=r["galaxy"], population=suite, logM200=float(r["logM200"]))
            for k, c in cols:
                v = float(r[c])
                row[k] = ((v if k.startswith("alpha") else math.log10(v)), math.nan, math.nan)
            rows.append(row)
    return rows

def figures(s):
    figure_q1(s)
    figure_q1_density(s)
    figure_q2(s)

def figure_q1(s):
    # Slope at 300 pc minus NFW against boson mass
    q1_panel(s, "alpha300")

def figure_q1_density(s):
    q1_panel(s, "rho300")

def q1_panel(s, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator
    from matplotlib.transforms import blended_transform_factory
    import plotstyle
    plotstyle.use()
    matplotlib.rcParams["hatch.linewidth"] = 0.6
    box = dict(fc=plotstyle.SURFACE, ec="none", pad=1.0, alpha=0.85)
    soliton_colour = soliton_text = plotstyle.TEAL
    halo_colour = halo_text = plotstyle.PLUM
    out = outdir(s)
    M200, logM_max, m_allowed = s["M200"], s["pina_logM_max"], s["m_allowed"]
    c = halo.concentration_dj19(M200)
    rho_cdm, alpha_cdm = halo.nfw_reference(0.3, M200, c)
    cdm = dict(alpha300=alpha_cdm, rho300=math.log10(rho_cdm))
    sp = spmini_points(s)

    # Piña's sample below logM_max, relative to the NFW of each M200
    pina, reach = {}, None
    for row in pina_tables(s["pina"]):
        if row["logM200"] >= logM_max:
            continue
        M = 10 ** row["logM200"]
        rho, alpha = halo.nfw_reference(0.3, M, halo.concentration_dj19(M))
        ref = alpha if k == "alpha300" else math.log10(rho)
        v, lo, hi = row[k]
        if math.isnan(v):
            continue
        group = "galaxies" if row["population"].startswith("gas") else "simulations"
        pina.setdefault(group, []).append(v - ref)
        if group == "galaxies":
            a, b = reach or (np.inf, -np.inf)
            reach = (min(a, v - ref - lo), max(b, v - ref + hi))

    # Soliton + NFW, and r_sol for the top axis
    j = ("alpha300", "rho300").index(k)
    m_grid = np.geomspace(6e-23, 1e-18, 180)
    curve = np.array([[p[k] - cdm[k] for p in soliton_nfw_point(m, M200, c)] for m in m_grid])
    r_sol = np.array([halo.soliton_radius_for_core_mass(m, halo.core_mass_chan(m, M200)[0]) for m in m_grid])   # kpc
    m_at = lambda r_kpc: float(np.exp(np.interp(-math.log(r_kpc), -np.log(r_sol), np.log(m_grid))))

    ylabel, ylim = {"alpha300": ("slope at 300 pc minus NFW slope\n(> 0 flatter, < 0 steeper)", (-3.4, 1.9)),
                    "rho300": ("log density at 300 pc minus NFW [dex]\n(> 0 denser)", (-1.3, 1.1))}[k]
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    tr = blended_transform_factory(ax.transData, ax.transAxes)

    # Lyman alpha bound and Piña's ranges
    ax.axvspan(m_grid[0], m_allowed, facecolor="none", edgecolor=plotstyle.GRID, hatch="////", lw=0, zorder=0)
    ax.axvline(m_allowed, color=plotstyle.INK_2, ls="--", lw=1, zorder=2)
    for group, shade in (("simulations", 0.175), ("galaxies", 0.275)):
        ax.axhspan(min(pina[group]), max(pina[group]), color=plotstyle.MUTED, alpha=shade, lw=0, zorder=1)
    for y in reach:
        ax.axhline(y, color=plotstyle.MUTED, ls=":", lw=1.2, zorder=2)
    ax.axhline(0, color=plotstyle.INK, lw=1, zorder=2)

    # Soliton + NFW and the SPmini haloes
    ax.fill_between(m_grid, np.nanmin(curve, 1), np.nanmax(curve, 1), color=soliton_colour, alpha=0.2, lw=0, zorder=3)
    ax.plot(m_grid, curve[:, 1], color=soliton_colour, lw=2, zorder=4)
    for (m, M), val in sp["iso"].items():
        if M == M200:
            mean, lo, hi = (x - cdm[k] for x in val[k])
            ax.errorbar(m, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", ms=7, color=halo_colour, mec=plotstyle.SURFACE,
                        mew=1.2, elinewidth=1.6, zorder=6)
    heavy = {m: val[k] - cdm[k] for (m, M), val in sp["heavy"].items() if M == M200}
    for m, v in heavy.items():
        ax.plot(m, v, "D", ms=6.5, color=halo_colour, mec=plotstyle.SURFACE, mew=1.2, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(m_grid[0], m_grid[-1])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("boson mass m [eV]")
    ax.grid(False)

    # Top axis: soliton radius; dotted line where it equals 300 pc
    radii = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    top = ax.secondary_xaxis("top")
    top.set_xticks([m_at(r) for r in radii], [f"{r * 1e3:g}" for r in radii])
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("soliton radius $r_{\\rm sol}$ [pc]", fontsize=10)
    ax.axvline(m_at(0.3), color=soliton_text, ls=":", lw=1, zorder=2)

    # Direct labels
    ax.text(math.sqrt(m_allowed * m_grid[-1]), {"alpha300": -0.9, "rho300": 0.55}[k], "allowed\nFDM halo\n= NFW halo",
            ha="center", va="center", fontsize=8.5, color=plotstyle.INK, weight="semibold")
    ax.text(m_allowed / 1.1, {"alpha300": 0.80, "rho300": 0.88}[k], "excluded by the\nLyman α forest", transform=tr,
            ha="right", va="center", fontsize=8, color=plotstyle.INK_2, bbox=box, zorder=7)
    ax.text(m_grid[-1] / 1.1, 0.0, "NFW", ha="right", va="bottom", fontsize=8, color=plotstyle.INK)
    x0 = sp["iso"][(8e-23, M200)][k][0] - cdm[k]
    ax.text(1.18e-22, x0 - 0.02 * (ylim[1] - ylim[0]), "halo 40", fontsize=8.5, color=halo_text, weight="semibold",
            va="center", bbox=box, zorder=7)
    m_h = min(heavy, key=heavy.get) if k == "alpha300" else min(heavy)
    ax.text(m_h / 1.18, heavy[m_h], "halo 300", fontsize=8.5, color=halo_text, weight="semibold", va="center", ha="right",
            bbox=box, zorder=7)
    if k == "alpha300":
        ax.text(6.6e-23, 0.955, "soliton larger than 300 pc, flatter than NFW", transform=tr, va="center", fontsize=8,
                color=plotstyle.INK_2, bbox=box, zorder=7)
        # Arrow to the falling branch
        i_min = int(np.nanargmin(curve[:, 1]))
        i = int(np.nanargmin(np.abs(np.where(np.arange(len(m_grid)) < i_min, curve[:, 1], np.nan) + 1.8)))
        ax.annotate("soliton edge at 300 pc\nsteeper than NFW", (m_grid[i], curve[i, 1]), xytext=(7.0e-23, -1.55),
                    fontsize=8, color=plotstyle.INK_2, va="center", bbox=box, zorder=7,
                    arrowprops=dict(arrowstyle="->", color=plotstyle.INK_2, lw=0.8, shrinkA=2, shrinkB=2))
        ax.text(m_grid[i_min] * 1.12, curve[i_min, 1] + 0.1, "soliton inside NFW", fontsize=8, color=soliton_text,
                va="center", bbox=box, zorder=7)
    else:
        i_max = int(np.nanargmax(curve[:, 1]))
        ax.text(m_grid[i_max] * 1.3, curve[i_max, 1] + 0.05, "soliton inside NFW", fontsize=8, color=soliton_text,
                va="bottom", bbox=box, zorder=7)

    handles = [Patch(color=plotstyle.MUTED, alpha=0.45, lw=0, label="observed galaxies"),
               Patch(color=plotstyle.MUTED, alpha=0.175, lw=0, label="CDM simulations with feedback"),
               Line2D([], [], color=plotstyle.MUTED, ls=":", lw=1.2, label="68% intervals of observed galaxies")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5, frameon=False)
    name = {"alpha300": "q1_recipe_vs_boson_mass.pdf", "rho300": "q1_density_vs_boson_mass.pdf"}[k]
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print(f"figures -> {out}")

def read_measurements(path):
    # Rows of a corenfw measurements.txt as dicts
    names = open(path).readline().lstrip("# ").split()
    return [dict(zip(names, row)) for row in np.atleast_2d(np.loadtxt(path))]

def figure_q2(s):
    # coreNFW n: one SPmini halo and its fit (left), n of all fits and of Piña's galaxies (right)
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle
    plotstyle.use()
    out = outdir(s)
    folder, m, M200 = s["profile"].split(":")
    m, M_target = float(m), float(M200)
    folder = os.path.join(HERE, folder)
    G = json.load(open(os.path.join(folder, "settings.json")))["G"]
    rows = [r for r in read_measurements(os.path.join(folder, "measurements.txt")) if r["m_eV"] == m and r["M_target"] == M_target]
    profiles = np.load(os.path.join(folder, "profiles_code_units.npz"))
    fit_colour = fit_text = plotstyle.TEAL
    halo_colour = halo_text = plotstyle.PLUM

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw=dict(width_ratios=[1, 1.15]))

    # Left: profile over the time samples, fit and NFW at the last one
    curves = []
    for row in rows:
        tag = f"t{row['t']:07.2f}"
        r, d = profiles[f"r_{tag}"], profiles[f"rho_{tag}"]
        keep = r > 0
        curves.append((r[keep] * row["length"], d[keep] * halo.Units(m, row["length"], G).density))
    r_kpc = curves[-1][0]
    band = np.array([np.interp(np.log(r_kpc), np.log(rr), np.log(dd)) for rr, dd in curves])
    last = rows[-1]
    inside = r_kpc <= last["R200"]
    ax.fill_between(r_kpc[inside], np.exp(band.min(0))[inside], np.exp(band.max(0))[inside], color=halo_colour, alpha=0.2, lw=0)
    ax.loglog(r_kpc[inside], curves[-1][1][inside], "o", ms=4.5, color=halo_colour, mec=plotstyle.SURFACE, mew=0.6, zorder=5)
    rr = np.geomspace(r_kpc[inside][0], last["R200"], 300)
    ax.loglog(rr, halo.corenfw_density(rr, last["M200"], last["c200"], last["rc"], last["n"]), color=fit_colour, lw=2, zorder=4)
    ax.loglog(rr, halo.nfw_density(rr, last["M200"], last["c_dj19"]), "--", color=plotstyle.INK, lw=1.2, zorder=3)
    ax.axvline(0.3, color=plotstyle.MUTED, lw=0.8)
    ax.set_ylim(5e3, 1e9)
    ax.text(0.31, 8e3, "300 pc", fontsize=8.5, color=plotstyle.MUTED)
    ax.axvline(last["rc"], color=fit_text, lw=0.8, ls=":")
    ax.text(last["rc"] * 1.05, 8e3, "r$_c$", fontsize=8.5, color=fit_text)
    i = np.searchsorted(rr, 3 * last["rc"])
    ax.annotate(f"coreNFW fit\nn = {last['n']:.2f}, r$_c$ = {last['rc']:.2f} kpc",
                (rr[i], halo.corenfw_density(rr[i], last["M200"], last["c200"], last["rc"], last["n"])),
                xytext=(1.5, 6e7), fontsize=9, color=fit_text, arrowprops=dict(arrowstyle="->", color=fit_text, lw=0.8))
    ax.text(0.1, 6e5, f"halo 40\nm = $10^{{{math.log10(m):g}}}$ eV, M$_{{200}}$ = $10^{{{math.log10(M_target):g}}}$ M$_\\odot$\n"
            f"dots at the last time\nband over {len(rows)} times", fontsize=9, color=halo_text, va="top")
    ax.text(0.13, 3.5e8, "NFW with the same M$_{200}$", fontsize=9, color=plotstyle.INK)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"dark matter density [M$_\odot$ kpc$^{-3}$]")

    # Right: n
    spmini_n = []
    for item in s["seeds"].split(",") + [s["heavy"]]:
        path, n = item.rsplit(":", 1)
        spmini_n += [v["n"] for v in read_summary(path, int(n)).values()]
    pina = [dict(population=r["population"], n=r["n"][0]) for r in pina_tables(s["pina"]) if r["population"].startswith("gas")]
    rng = np.random.default_rng(1)
    groups = ((f"FDM haloes\n({len(spmini_n)} fits)", spmini_n, halo_colour, "o"),
              ("gas rich galaxies", [float(p["n"]) for p in pina if p["population"].startswith("gas-rich")], plotstyle.GAS_RICH, "o"),
              ("gas poor galaxies", [float(p["n"]) for p in pina if p["population"] == "gas-poor"], plotstyle.GAS_POOR, "s"))
    for y, (name, values, color, marker) in enumerate(groups):
        bx.plot(values, y + rng.uniform(-0.18, 0.18, len(values)), marker, ls="none", ms=6, color=color, alpha=0.9, mec=plotstyle.SURFACE, mew=0.6)
    bx.set_yticks(range(len(groups)), [g[0] for g in groups])
    bx.set_ylim(len(groups) - 0.5, -0.5)
    bx.set_xlim(-0.03, 1.08)
    bx.axvline(1, color=plotstyle.MUTED, lw=0.8)
    bx.grid(axis="y", visible=False)
    bx.set_xlabel("coreNFW n   (0 = cusp, 1 = fully cored)")
    bx.text(0.0, 2.45, "galaxies are posterior medians (prior 0 ≤ n ≤ 1)\nFDM haloes are best fits", fontsize=8, color=plotstyle.MUTED, ha="left", va="bottom")
    bx.text(0.97, 0, f"all at n = {min(spmini_n):.2f}", fontsize=9, color=halo_text, ha="right", va="center")
    fig.savefig(os.path.join(out, "q2_coreness.pdf"))
    plt.close(fig)

# Plots and snapshots
def save_snapshot(box, a, step, out, L_Mpc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rho = box.density().cpu().numpy()
    np.savez(os.path.join(out, f"snap_{step:06d}.npz"), psi=box.psi.cpu().numpy(), a=a)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.log10(rho.mean(axis=2)).T, origin="lower", extent=[0, L_Mpc, 0, L_Mpc], cmap="inferno")
    plt.title(f"z = {1 / a - 1:.2f}   (log projected density)")
    plt.xlabel("Mpc"); plt.ylabel("Mpc")
    plt.savefig(os.path.join(out, f"snap_{step:06d}.png"), dpi=100, bbox_inches="tight")
    plt.close()

def plot_cosmo(out, k, P_start, P_end, s, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.loglog(k, P_start, label=f"z = {s['z_start']:g}")
    plt.loglog(k, P_end, label=f"z = {1 / a - 1:g}")
    plt.xlabel("k [1/Mpc]"); plt.ylabel("P(k) [Mpc$^3$]"); plt.legend()
    plt.savefig(os.path.join(out, "power_spectrum.png"), dpi=110, bbox_inches="tight")
    plt.close()

def plot_soliton(out, rows, box, r, rho, rho_c, r_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].imshow(np.log10(box.density().cpu().numpy().sum(axis=2)).T, origin="lower", cmap="inferno")
    ax[0].set_title("projected density at the end")
    ax[1].loglog(r, rho, "o", ms=3, label="measured"); ax[1].loglog(r, soliton_profile(r, rho_c, r_c), label="soliton fit")
    ax[1].set_xlabel("r"); ax[1].set_ylabel("density"); ax[1].legend()
    ax[2].plot(rows[:, 0], rows[:, 4] / rows[:, 5]); ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set_xlabel("t"); ax[2].set_ylabel("r_c / predicted")
    plt.savefig(os.path.join(out, "soliton.png"), dpi=110, bbox_inches="tight")
    plt.close()

def plot_blackhole(out, s, profiles, rows, track, box, x_bh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle
    plotstyle.use()

    # Ground states: profiles (darker = heavier black hole), half-mass radius and peak density
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    shades = plotstyle.ramp(plotstyle.SERIES[0], len(profiles))[::-1]
    for (q, r, rho), color in zip(profiles, shades):
        ax[0].loglog(r, rho, color=color, label=f"{q:g}")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("density"); ax[0].set_ylim(1e-6, None)
    ax[0].legend(title="$M_\\mathrm{bh}/M$", loc="lower left")
    ax[0].set_title("Density profiles")
    q, peak, r_half, a0 = rows.T
    ax[1].plot(q, r_half / r_half[0], "o-", color=plotstyle.SERIES[0], mec="white", mew=1.5, label="measured")
    q_line = np.linspace(q[q > 0].min() / 2, q.max(), 100)
    # The black hole alone, with the same softening as the run; the unsoftened 1.337 a0 is what a finer grid would approach
    eps = s["softening"] * s["L"] / s["n"]
    alone = [softened_half_radius(s["G"] * x * s["mass"], eps) for x in q_line]
    ax[1].plot(q_line, np.array(alone) / r_half[0], color=plotstyle.MUTED, lw=1.2,
               label=f"black hole alone, softened over {eps:g}")
    ax[1].plot(q_line, 1.337 / (s["G"] * q_line * s["mass"]) / r_half[0], color=plotstyle.MUTED, lw=1, ls="--",
               label="black hole alone, unsoftened: $1.337\\,a_0$")
    ax[1].set_ylim(0, 1.1); ax[1].set_xlabel("$M_\\mathrm{bh}/M$"); ax[1].set_ylabel("half-mass radius / without black hole")
    ax[1].legend(loc="upper right"); ax[1].set_title("The core contracts")
    ax[2].plot(q, peak / peak[0], "o-", color=plotstyle.SERIES[0], mec="white", mew=1.5)
    ax[2].set_xlabel("$M_\\mathrm{bh}/M$"); ax[2].set_ylabel("peak density / without black hole"); ax[2].set_title("and densifies")
    plt.savefig(os.path.join(out, "ground_states.png"), dpi=150)
    plt.close()

    # Sinking: distance to the core against time, and the path over the projected density at the end
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax[0].plot(track[:, 0], track[:, 7], color=plotstyle.SERIES[1])
    ax[0].set_xlabel("t"); ax[0].set_ylabel("distance from the black hole to the core"); ax[0].set_ylim(0, None)
    ax[0].set_title(f"A black hole of {s['sink_ratio']:g} soliton masses, released at rest")
    L = s["L"]
    image = ax[1].imshow(np.log10(box.density().cpu().numpy().sum(axis=2) * box.dx).T, origin="lower", cmap="Blues",
                         extent=[-L / 2, L / 2, -L / 2, L / 2])
    image.set_clim(image.get_clim()[1] - 3, image.get_clim()[1])
    fig.colorbar(image, ax=ax[1], shrink=0.8, label="log$_{10}$ projected density").outline.set_visible(False)
    ax[1].plot(track[:, 1], track[:, 2], color=plotstyle.SERIES[1], lw=1, label="black hole")
    ax[1].plot(track[:, 4], track[:, 5], color=plotstyle.INK, lw=1, label="core centre")
    ax[1].plot(*x_bh[:2], "o", color=plotstyle.SERIES[1], mec="white", mew=1.5)
    w = 2.5 * s["sink_offset"]
    ax[1].set_xlim(-w, w); ax[1].set_ylim(-w, w); ax[1].grid(False)
    ax[1].set_xlabel("x"); ax[1].set_ylabel("y"); ax[1].legend(loc="upper right"); ax[1].set_title("Paths, over the density at the end")
    plt.savefig(os.path.join(out, "sinking.png"), dpi=150)
    plt.close()

# Sanity checks
def check():
    results = []

    def report(name, passed, detail):
        results.append(passed)
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)

    # 1. Poisson: for rho = 1 + 0.5 cos(2x), lap V = rho - 1 gives V = -0.5 cos(2x) / 4
    n, L = 32, 2 * math.pi
    x = np.arange(n) * L / n
    box = Box(torch.tensor(np.sqrt(1 + 0.5 * np.cos(2 * x))[:, None, None] * np.ones((1, n, n)) + 0j), L)
    error = np.abs(box.V.numpy()[:, 0, 0] + 0.125 * np.cos(2 * x)).max() / 0.125
    report("Poisson equation, one Fourier mode", error < 1e-10, f"relative error {error:.1e}")

    # 2. A free Gaussian packet spreads as width^2 = 1 + (t/2)^2 (its initial width is 1)
    n, L, t = 64, 20.0, 2.0
    X, Y, Z = (np.arange(n)[:, None, None] * L / n - L / 2, np.arange(n)[None, :, None] * L / n - L / 2,
               np.arange(n)[None, None, :] * L / n - L / 2)
    r2 = X ** 2 + Y ** 2 + Z ** 2
    box = Box(torch.tensor(np.exp(-r2 / 4) + 0j), L, four_pi_G=0.0)
    box.step(t)
    rho = box.density().numpy()
    width = math.sqrt((rho * r2).sum() / rho.sum() / 3)
    report("free particle spreads exactly", abs(width / math.sqrt(1 + (t / 2) ** 2) - 1) < 2e-3,
           f"width {width:.4f}, exact {math.sqrt(1 + (t / 2) ** 2):.4f}")

    # 3. Mass and energy are conserved in a self-gravitating box
    rng = np.random.default_rng(3)
    box = Box(torch.tensor(1 + 0.1 * (rng.normal(size=(32, 32, 32)) + 1j * rng.normal(size=(32, 32, 32)))),
              10.0, four_pi_G=4 * math.pi * 0.05)
    M0, E0 = box.mass(), box.energy()
    for _ in range(400):
        box.step(2e-3)
    dM, dE = abs(box.mass() / M0 - 1), abs(box.energy() / E0 - 1)
    report("mass and energy conserved", dM < 1e-12 and dE < 1e-3, f"mass drift {dM:.0e}, energy drift {dE:.0e} (400 steps)")

    # 4. The time step is second order: halving dt divides the error by 4
    psi0 = torch.tensor(1 + 0.3 * (rng.normal(size=(24, 24, 24)) + 1j * rng.normal(size=(24, 24, 24))))

    def evolve(steps, T=0.2):
        box = Box(psi0.clone(), 10.0)
        for _ in range(steps):
            box.step(T / steps)
        return box.psi

    exact = evolve(2048)
    errors = [(evolve(m) - exact).abs().max().item() for m in (16, 32, 64)]
    order = math.log2(errors[0] / errors[1]) / 2 + math.log2(errors[1] / errors[2]) / 2
    report("kick-drift-kick is second order", 1.8 < order < 2.2, f"measured order {order:.2f}")

    # 5. The ground state does not move under real-time evolution
    ref, converged = ground_state(32, 10.0, 0.5, 50.0)
    rho0 = ref.density().clone()
    for _ in range(200):
        ref.step(1e-4)
    change = ((ref.density() - rho0).abs().max() / rho0.max()).item()
    report("the soliton is stationary", converged and change < 1e-2, f"density changed by {change:.1e} of its peak")

    # 6. Code units: hbar/m = 1 and 4 pi G rho_mean = 1
    from universe import HBAR, G
    u = Universe()
    kinetic, gravity = u.T0 * HBAR / (u.m * u.L0 ** 2), u.T0 ** 2 * 4 * math.pi * G * u.rho_mean
    report("code units", abs(kinetic - 1) < 1e-12 and abs(gravity - 1) < 1e-12, f"hbar/m = {kinetic:.12f}, 4 pi G rho = {gravity:.12f}")

    # 7. A small density wave grows like a (linear theory, matter era) when started in the growing mode
    n, L_Mpc, a = 32, 4.0, 1.0 / 1001.0
    L = u.code_length(L_Mpc)
    kx = 2 * math.pi / L
    x = np.arange(n) * L / n
    delta = 1e-4 * np.cos(kx * x)[:, None, None] * np.ones((1, n, n))
    S = u.growing_mode(a) * delta / kx ** 2                     # lap S = -g delta for a single mode
    box = Box(torch.tensor(np.sqrt(1 + delta) * np.exp(1j * S)), L)
    a0 = a
    while a < 0.02:
        dt = 0.1 * a / u.da_dtau(a)
        box.step(dt, u.advance(a, dt / 2))
        a = u.advance(a, dt)
    grown = np.abs(np.fft.fftn(box.density().numpy() - 1))[1, 0, 0] / np.abs(np.fft.fftn(delta))[1, 0, 0]
    ratio = grown / (a / a0)
    report("linear growth", 0.9 < ratio < 1.1, f"a grew x{a / a0:.1f}; density wave grew x{grown:.1f} (ratio {ratio:.3f})")

    # 8. A point mass alone (no self-gravity): the ground state of -1/sqrt(r^2 + eps^2), against a 1D radial solution
    from scipy.linalg import eigh_tridiagonal
    eps, h, N = 0.3, 1.5e-3, 20000
    r = h * np.arange(1, N + 1)
    exact = eigh_tridiagonal(1 / h ** 2 - 1 / np.sqrt(r ** 2 + eps ** 2), -0.5 / h ** 2 * np.ones(N - 1),
                             select="i", select_range=(0, 0))[0][0]
    box, converged = ground_state(48, 14.0, 0.0, 1.0, tolerance=1e-7, external=lambda b: point_mass(b, 1.0, 1.0, (0, 0, 0), eps))
    mu = box.energy() / box.mass()
    report("ground state of a point mass", converged and abs(mu / exact - 1) < 1e-3, f"energy {mu:.5f}, exact {exact:.5f}")

    # 9. A point mass falling through a soliton: the total momentum (field + point mass) stays zero
    ref, _ = ground_state(32, 10.0, 1.0, 3.0, tolerance=1e-6)
    box = Box(ref.psi, 10.0, 4 * math.pi)
    M_bh, x_bh, v_bh, eps, dt = 0.3, np.array([0.8, 0.3, 0.0]), np.zeros(3), 2 * box.dx, 2e-3
    box.V_ext = point_mass(box, 1.0, M_bh, x_bh, eps)
    acc = pull(box, 1.0, x_bh, eps)
    for _ in range(300):
        v_bh += acc * dt / 2
        x_bh = x_bh + v_bh * dt
        box.step(dt, V_ext_end=point_mass(box, 1.0, M_bh, x_bh, eps))
        acc = pull(box, 1.0, x_bh, eps)
        v_bh += acc * dt / 2
    P = box.momentum() + M_bh * v_bh
    report("momentum conserved with a point mass", np.linalg.norm(P) < 1e-4 * M_bh * np.linalg.norm(v_bh),
           f"total {np.linalg.norm(P):.1e}, point mass alone {M_bh * np.linalg.norm(v_bh):.1e}")

    # 10. Static-box units: hbar/m = 1 and G = G_code in code units, for any boson mass and length
    from universe import HBAR, EV, C, MSUN
    units = halo.Units(1e-22, 0.7, G_code=2.0)
    m_kg = 1e-22 * EV / C ** 2
    kinetic = HBAR * units.time / (m_kg * (units.length * KPC) ** 2)
    gravity = G * units.mass * MSUN * units.time ** 2 / (units.length * KPC) ** 3
    report("static-box units", abs(kinetic - 1) < 1e-12 and abs(gravity / 2 - 1) < 1e-12,
           f"hbar/m = {kinetic:.12f}, G = {gravity:.12f} (set 2)")

    # 11. The zoomed field between grid points, and the slope at a radius, against a Gaussian known exactly
    n, L, w = 40, 14.0, 0.8
    X, Y, Z = (np.arange(n)[:, None, None] + 0.5) * L / n - L / 2, (np.arange(n)[None, :, None] + 0.5) * L / n - L / 2, \
        (np.arange(n)[None, None, :] + 0.5) * L / n - L / 2
    c0 = np.array([0.13, -0.27, 0.05])
    psi = np.exp(-((X - c0[0]) ** 2 + (Y - c0[1]) ** 2 + (Z - c0[2]) ** 2) / (4 * w ** 2)) + 0j
    psi_k = torch.fft.fftn(torch.tensor(psi))
    cube, s = halo.zoom(psi_k, L, c0, 1.0, 17)
    exact = np.exp(-(s[:, None, None] ** 2 + s[None, :, None] ** 2 + s[None, None, :] ** 2) / (4 * w ** 2))
    error = np.abs(cube - exact).max()
    found = halo.peak(psi_k, L, np.zeros(3))
    _, slope = halo.at_radius(psi_k, L, c0, 1.0)
    report("spectral zoom", error < 1e-6 and np.abs(found - c0).max() < L / n / 16 and abs(slope + 1 / w ** 2) < 0.02,
           f"field error {error:.1e}, peak off by {np.abs(found - c0).max():.3f}, slope at r = 1: {slope:.4f} (exact {-1 / w ** 2:.4f})")

    # 12. coreNFW: the fit recovers the parameters of an exact profile, and Eqs. 13-14 match Eq. 2
    M200, c, r_c, n_core = 3e9, 12.0, 0.8, 0.6
    r = np.geomspace(0.05, halo.R200_of(M200), 30)
    fit = halo.fit_corenfw(r, halo.corenfw_density(r, M200, c, r_c, n_core), M200, 0.5)
    d, a = halo.nfw_reference(0.3, M200, c)
    h = 1e-5
    a_num = (math.log(halo.nfw_density(0.3 * (1 + h), M200, c)) - math.log(halo.nfw_density(0.3 * (1 - h), M200, c))) / \
        (math.log(1 + h) - math.log(1 - h))
    report("coreNFW fit and NFW reference",
           abs(fit[0] / c - 1) < 1e-3 and abs(fit[1] / r_c - 1) < 1e-3 and abs(fit[2] - n_core) < 1e-3
           and abs(d / halo.nfw_density(0.3, M200, c) - 1) < 1e-12 and abs(a - a_num) < 1e-6,
           f"c {fit[0]:.3f} ({c}), r_c {fit[1]:.3f} ({r_c}), n {fit[2]:.3f} ({n_core}); slope {a:.6f} vs {a_num:.6f}")

    print(f"\n  {sum(results)} of {len(results)} passed")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("cosmo", "soliton", "blackhole", "corenfw", "solitonnfw", "figures", "check"):
        sys.exit(USAGE)
    command = sys.argv[1]
    settings = {"cosmo": COSMO, "soliton": SOLITON, "blackhole": BLACKHOLE, "corenfw": CORENFW, "solitonnfw": SOLITON_NFW, "figures": FIGURES, "check": {}}[command]
    for item in sys.argv[2:]:
        key, value = item.split("=", 1)
        if key not in settings:
            sys.exit(f"unknown setting {key}; the settings are: {', '.join(settings)}")
        settings[key] = type(settings[key])(value) if not isinstance(settings[key], str) else value
    if command == "cosmo":
        cosmo(settings)
    elif command == "soliton":
        soliton(settings)
    elif command == "blackhole":
        blackhole(settings)
    elif command == "corenfw":
        corenfw(settings)
    elif command == "solitonnfw":
        soliton_nfw(settings)
    elif command == "figures":
        figures(settings)
    else:
        check()

if __name__ == "__main__":
    main()