# Magnetics & magnetodynamics

The `magnetodynamics` branch's focus. Two layers: **long-range dipolar solvers** (magnetostatics)
and **per-particle magnetization dynamics** (magnetize / thermal Stoner-Wohlfarth).

## Feature flags (`src/config/features.def`, lines 61-66, 101-102)
| Flag | Relation |
|---|---|
| `DIPOLES` | implies `ROTATION` |
| `DP3M` | equals `DIPOLES and FFTW` |
| `DIPOLE_FIELD_TRACKING` | implies `DIPOLES` |
| `THERMAL_STONER_WOHLFARTH` | **requires `NLOPT`**; implies `VIRTUAL_SITES_RELATIVE`, `DIPOLE_FIELD_TRACKING` |
| `MAGNETIZE` | implies `VIRTUAL_SITES_RELATIVE`, `DIPOLE_FIELD_TRACKING` |
| `SCAFACOS_DIPOLES` | requires `SCAFACOS`; implies `DIPOLES` |

In code the macros are prefixed, e.g. `#ifdef ESPRESSO_MAGNETIZE`, `#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH`.

## Long-range dipolar solvers — `src/core/magnetostatics/`
`dipoles.cpp/.hpp`, `dipoles_inline.hpp`, `solver.hpp` (dispatch); backends: `dipolar_direct_sum.*`
(+ `_gpu`, `_gpu_cuda.cu`), `dp3m_heffte.*` (dipolar P3M via HeFFTe), `dlc.*` (dipolar layer correction),
`scafacos_impl.*`. Python: `src/python/espressomd/magnetostatics.py` (`DipolarP3M`, `DipolarDirectSum`,
`Scafacos`, `DLC`, base `MagnetostaticInteraction`). Script interface: `src/script_interface/magnetostatics/`
(one `AutoParameters`-derived class per Python class).

## Magnetize models — `src/core/magnetostatics/magnetize.cpp` (`ESPRESSO_MAGNETIZE`)
Per-particle induced-moment models applied over the local + dipolar field:
- `magnetize_p_Langevin` — Langevin response with `alpha = 3 * mag_susc_0 / dipm_sat * |H|` (i.e. **α = 3χ₀H/M_sat**),
  small-x expansion near zero; saturation via `tanh_response`.
- `magnetize_p_froelich_kennelly` — Fröhlich-Kennelly: `pre = χ₀·M_sat / (M_sat + χ₀·|H|)`.
- Driver: `System::integrate_magnetodynamics_testing()`; external field from `HomogeneousMagneticField` constraint.

## Thermal Stoner-Wohlfarth — `stoner_wohlfarth_thermal.cpp` (`ESPRESSO_THERMAL_STONER_WOHLFARTH`, needs NLOPT)
Kinetic Monte-Carlo of the SW two-state model. `phi_objective` = reduced magnetic energy
`-0.5 - 0.5·cos(2(φ-θ)) - 2h·cos(φ)`; `get_phi_at_energy_min` minimizes it with **nlopt** and does a
thermally-activated MC step (attempt frequency `tau0_inv`, `1/(k_B T V)` factor).

## Per-particle seam — `src/script_interface/particle_data/ParticleHandle.cpp`
The magnetize/SW parameters are **particle properties**, not a magnetostatics actor:
- `ESPRESSO_MAGNETIZE`: `magnetize_func`, `dipm_sat` (saturation moment M_sat), `mag_susc_0` (initial susceptibility χ₀),
  and `multidomain_mag_response` (dict: `field_center`, `saturation_level`, `field_gain`, `response_gain`).
- `ESPRESSO_THERMAL_STONER_WOHLFARTH`: `magnetodynamics` (dict: `is_enabled`, `sw_phi_0`, `sat_mag`,
  `anisotropy_field_inv`, `anisotropy_energy`, `sw_tau0_inv`, `sw_dt_incr`).

## Docs & tests
- Docs: `doc/sphinx/magnetodynamics.rst` (Thermal SW), `doc/sphinx/magnetostatics.rst`.
- Tests: `testsuite/python/thermal_stoner_wohlfarth.py`, `..._fluid.py`, `dipole_field_tracking.py`,
  `dipolar_p3m.py`, `dipolar_direct_summation.py`, `dipolar_interface.py`,
  `dipolar_mdlc_p3m_scafacos_p2nfft.py`, `scafacos_dipoles_1d_2d.py`, `constraint_homogeneous_magnetic_field.py`.

## Cross-link
`mag_susc_0` (χ₀) / `dipm_sat` (M_sat) here are the physical parameters that `pressomancy`'s magnetize
helpers set and that `SCRIPTS/pyanal`'s χ-based analysis (α = 3χH/M_sat) interprets downstream.
