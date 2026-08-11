# Magnetics & magnetodynamics

The `magnetodynamics` branch's focus. Two layers: **long-range dipolar solvers** (magnetostatics)
and **per-particle magnetization dynamics** (field-response models + thermal Stoner-Wohlfarth),
now unified behind one driver.

## Feature flags (`src/config/features.def`, `/* Magnetostatics */` section)
| Flag | Relation |
|---|---|
| `DIPOLES` | implies `ROTATION` |
| `DP3M` | equals `DIPOLES and FFTW` |
| `DIPOLE_FIELD_TRACKING` | implies `DIPOLES` |
| `THERMAL_STONER_WOHLFARTH` | **requires `NLOPT`**; implies `VIRTUAL_SITES_RELATIVE`, `DIPOLE_FIELD_TRACKING` |
| `LANGEVIN_MAGNETIZATION` | implies `VIRTUAL_SITES_RELATIVE`, `DIPOLE_FIELD_TRACKING` |
| `FROELICH_KENNELLY` | implies `VIRTUAL_SITES_RELATIVE`, `DIPOLE_FIELD_TRACKING` |
| `SCAFACOS_DIPOLES` | requires `SCAFACOS`; implies `DIPOLES` |

In code the macros are prefixed, e.g. `#ifdef ESPRESSO_LANGEVIN_MAGNETIZATION`,
`#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH`.

## Long-range dipolar solvers — `src/core/magnetostatics/`
`dipoles.cpp/.hpp`, `dipoles_inline.hpp`, `solver.hpp` (dispatch); backends: `dipolar_direct_sum.*`
(+ `_gpu`, `_gpu_cuda.cu`), `dp3m_heffte.*` (dipolar P3M via HeFFTe), `dlc.*` (dipolar layer correction),
`scafacos_impl.*`. Python: `src/python/espressomd/magnetostatics.py` (`DipolarP3M`, `DipolarDirectSum`,
`Scafacos`, `DLC`, base `MagnetostaticInteraction`). Script interface: `src/script_interface/magnetostatics/`
(one `AutoParameters`-derived class per Python class).

## Unified magnetodynamics driver — `src/core/magnetostatics/magnetodynamics.{hpp,cpp}`
One driver `System::integrate_magnetodynamics(bool initial_step)` gathers the external field
(`get_external_field`, sums `HomogeneousMagneticField` constraints), loops local virtual particles,
forms `total_field = ext_fld + dip_fld`, and dispatches to the per-particle model enabled on that
particle. Only the **SW branch** resolves the reference particle (`get_reference_particle`, for the
easy axis); the field-response models need only `is_virtual()`. **Phase-aware**: deterministic
field-response models run every call (incl. the pre-loop init block, `initial_step=true`);
stochastic/thermal models (SW) run **in-loop only** (`initial_step=false`). `magnetodynamics.hpp`
declares the model helpers. Called from `integrate.cpp` (pre-loop + in-loop), in both phases
**before** `update_ghosts_and_resort_particle`, so ghosts carry the moments the force calculation
uses. A one-model-per-particle guard (`Particle::enabled_magnetodynamics_models() > 1`) lives in
`integrator_sanity_checks()`.

⚠ **`dip_fld` is only written by `DipolarDirectSum`** (CPU `dipolar_direct_sum.cpp` + GPU), which
computes it from its own global `all_gather` — never from ghosts. With `DipolarP3M`/`DLC`/ScaFaCoS
it stays **zero**, so the models silently respond to the external field alone. It is also lagged one
step (refilled in `calculate_forces` after the driver) and zero on the pre-loop call, so mutual
magnetization is solved **explicitly**, converging over several steps. Documented, not enforced.

## Field-response models (deterministic, thermostat-free; χ₀ from `mag_susc_0`, M_sat from `dipm_sat`)
- `langevin_magnetization.cpp` (`ESPRESSO_LANGEVIN_MAGNETIZATION`) — **α = 3χ₀|H|/M_sat**,
  small-x expansion near zero; `m = M_sat·L(α)·Ĥ`. For large particles where kT is negligible.
- `froelich_kennelly.cpp` (`ESPRESSO_FROELICH_KENNELLY`) — `pre = χ₀·M_sat / (M_sat + χ₀·|H|)`, `m = pre·H`.
- Per-particle enable via `langevin_magnetization_is_enabled` / `froelich_kennelly_is_enabled`.
- Both short-circuit to `dipm = 0` when `|H| < 1e-5` or `dipm_sat <= 0`. Setters validate
  `dipm_sat > 0` / `mag_susc_0 >= 0` (`ParticleHandle.cpp`, `std::domain_error` → Python `ValueError`).

## Thermal Stoner-Wohlfarth — `stoner_wohlfarth_thermal.cpp` (`ESPRESSO_THERMAL_STONER_WOHLFARTH`, needs NLOPT)
Kinetic Monte-Carlo of the SW two-state model; helpers `stoner_wohlfarth_no_field` /
`stoner_wohlfarth_main` (called by the unified driver). `phi_objective` = reduced magnetic energy
`-0.5 - 0.5·cos(2(φ-θ)) - 2h·cos(φ)`; `get_phi_at_energy_min` minimizes it with **nlopt** and does a
thermally-activated MC step (attempt frequency `tau0_inv`, `1/(k_B T V)` factor).

## Per-particle seam — `src/script_interface/particle_data/ParticleHandle.cpp`
The model parameters are **particle properties**, not a magnetostatics actor:
- Field-response (`LANGEVIN_MAGNETIZATION` / `FROELICH_KENNELLY`): shared `dipm_sat` (M_sat),
  `mag_susc_0` (χ₀), plus per-model enable flags `langevin_magnetization_is_enabled` /
  `froelich_kennelly_is_enabled`.
- `ESPRESSO_THERMAL_STONER_WOHLFARTH`: `magnetodynamics` (dict: `is_enabled`, `sw_phi_0`, `sat_mag`,
  `anisotropy_field_inv`, `anisotropy_energy`, `sw_tau0_inv`, `sw_dt_incr`).

## Docs & tests
- Docs: `doc/sphinx/magnetodynamics.rst` (common requirements + Langevin + Froelich-Kennelly +
  Thermal SW), `doc/sphinx/magnetostatics.rst`, feature list in `doc/sphinx/installation.rst`.
- Tests: `testsuite/python/magnetization_models.py` (M(H) curves vs. closed form, zero-field and
  saturation limits, mutual-pair explicit fixed point, setter validation, model exclusivity;
  MAX_NUM_PROC 4), `src/core/unit_tests/magnetodynamics_test.cpp` (pure-kernel unit tests),
  `testsuite/python/thermal_stoner_wohlfarth.py`, `..._fluid.py`, `dipole_field_tracking.py`,
  `dipolar_p3m.py`, `dipolar_direct_summation.py`, `dipolar_interface.py`,
  `dipolar_mdlc_p3m_scafacos_p2nfft.py`, `scafacos_dipoles_1d_2d.py`, `constraint_homogeneous_magnetic_field.py`.

## Cross-link
`mag_susc_0` (χ₀) / `dipm_sat` (M_sat) here are the physical parameters that `pressomancy`'s magnetize
helpers set and that `SCRIPTS/pyanal`'s χ-based analysis (α = 3χH/M_sat) interprets downstream.
