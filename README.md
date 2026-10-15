# SPmini

A small Schrödinger–Poisson solver for wave (fuzzy) dark matter, written in Python with PyTorch. It is the code of
the paper *Fuzzy dark matter hides inside the inner density profiles of galaxies* (Fuentes Aguilar, 2026). 
SPmini is a lite version of SP (Fuentes Aguilar, Master Thesis, 2015).

## The equations

A single complex field ψ with density ρ = |ψ|², in units where ħ/m = 1:

    i ∂ψ/∂t = −½ ∇²ψ + (V + V_ext) ψ,  ∇²V = 4πG (ρ − ρ̄)            (static box, used in the paper)
    i ∂ψ/∂τ = −½ ∇²ψ + a V ψ,          ∇²V = |ψ|² − 1               (comoving, cosmological box)

Each time step is kick–drift–kick (Strang splitting):

- **kick:** ψ → ψ e^(−i V Δt/2);
- **drift:** ψ_k → ψ_k e^(−i k² Δt/2);
- **Poisson:** V_k = −4πG ρ_k / k².

Each piece is exact, so the mass is conserved to rounding error, and the step is second order in Δt.
The step is Δt = min(0.25 Δx², 0.5 / max|V|).

## Files

| File | Contents |
|---|---|
| `wave.py` | The solver (`Box`), initial conditions (`cloud`, `blobs`), ground state, soliton profile, radial profiles. |
| `halo.py` | Physical units, R200, NFW and coreNFW, the measurements of Mancera Piña et al. (2026), soliton + NFW, core mass of Chan et al. (2022). |
| `universe.py` | Flat ΛCDM background and cosmological initial conditions. |
| `run.py` | All settings and the command line. |
| `export_profile.py` | Radial profiles of the snapshots of a run, used by the coreness figure. |
| `plotstyle.py` | Figure style. |
| `data/psidm_pk_z1000.txt` | Linear power spectrum at z = 1000 with the wave dark matter cut-off (cosmological box only). |
| `results/corenfw/` | Measurements and summaries of the runs of the paper (no snapshots). |
| `results/solitonnfw/` | Soliton + NFW, measured in the same way. |

## Requirements

Python 3 with `torch`, `numpy`, `scipy`, `matplotlib` and `colossus` (tested with torch 2.6). Without `colossus` the
NFW reference is `nan`. `device=auto` uses a GPU when there is one. The paper runs use `dtype=complex128`, which Apple
GPUs do not support.

## Tests

    python run.py check

Twelve tests against exact results (about 1 min on a CPU): Poisson for one mode, spreading of a free Gaussian packet,
mass and energy conservation, second order in time, stationary soliton, code units, linear growth, ground state of a
point mass, momentum with a point mass, static-box units, Fourier interpolation, coreNFW fit.

## Reproducing the paper

Any setting of `run.py` can be changed on the command line as `key=value`. The defaults of `CORENFW` are a small test
halo, not the runs of the paper.

**Halo 40** (cloud mass 40, seeds 1 to 4, main grid 384³, check grid 320³):

    python run.py corenfw L=48 cloud_mass=40 cloud_radius=20 cloud_core=0.5 concentration=10 shape=iso speeds=flat \
        t_end=320 samples=8 masses=8e-23,1e-22,2e-22 halo_masses=5e8,1e9,2e9 dtype=complex128 \
        seed=1 resolutions=320,384 out=results/corenfw/iso_hpc

    python run.py corenfw ... seed=2 resolutions=384 out=results/corenfw/iso_seed2      # and seeds 3, 4

**Halo 300** (cloud mass 300, one seed, main grid 640³, check grid 512³):

    python run.py corenfw L=48 cloud_mass=300 cloud_radius=20 cloud_core=0.5 concentration=10 shape=iso speeds=flat \
        t_end=320 samples=8 masses=2e-22,3e-22,4e-22,6e-22 halo_masses=5e8,1e9,2e9 dtype=complex128 \
        seed=1 resolutions=512,640 out=results/corenfw/heavy_M300

Each run writes `summary.txt` (time averages per grid and the convergence between the first two grids) and, per grid,
`measurements.txt` (one row per snapshot and pair). I performed a few runs on HPC. In particular, on one V100 in double
precision, 320³ takes about 1 h and 384³ about 3 h; 640³ takes about 10 h on an H100. 

**Soliton + NFW** (no simulation; validated against the two haloes):

    python run.py solitonnfw

**Figures:**

    python export_profile.py results/corenfw/iso_hpc/n384      # needs the snapshots of the run
    python run.py figures pina=<folder>

The SPmini side of every figure comes from the summaries in `results/`, so the figures can be drawn without running the simulations again.

## Other commands

`cosmo` (cosmological box, z = 1000 to 10), `soliton` (blobs collapse into a soliton) and `blackhole` (a point mass in
a soliton) are kept from earlier work and are not used in the paper. The uniform grid cannot resolve a soliton in a
cosmological box, so `cosmo` is qualitative only.