# Core architecture & the Python↔C++ seam

Concise map of `src/core/` and how features reach Python. Key **header pointers live in `AGENT.md`**
(System.hpp, Particle.hpp, CellStructure.hpp, integrate.hpp, forces.hpp, thermostat.hpp, …) — read those there.

## `src/core/` subsystems
| Area | Where | Notes |
|---|---|---|
| System / box | `system/System.hpp`, `BoxGeometry.hpp`, `LocalBox.hpp` | `System` owns all components |
| Particle data | `Particle.hpp`, `ParticleList.hpp`, `ParticleRange.hpp`, `BondList.hpp` | core data structures |
| Cell system | `cell_system/` (`CellStructure.hpp`, `cells.cpp`, `ghosts.cpp`) | spatial decomposition, Verlet/ghost layers |
| Integration | `integrate.cpp`, `integrators/`, `propagation.cpp`, `rattle.cpp` | velocity-Verlet, NPT, Brownian, SD |
| Forces/energy | `forces*.hpp/.cpp`, `energy_inline.hpp`, `pressure_inline.hpp` | + Cabana/AoSoA path (`*_cabana.hpp`, `aosoa_pack.hpp`) |
| Interactions | `nonbonded_interactions/`, `bonded_interactions/`, `bonds.cpp`, `bond_breakage/` | pair & bond potentials |
| Electrostatics | `electrostatics/` (`coulomb.hpp`, `p3m.hpp`, `p3m/`, `fft/`, `scafacos/`) | long-range Coulomb |
| Magnetostatics | `magnetostatics/` | see `magnetics.md` |
| Thermostats | `thermostat.cpp`, `thermostats/`, `dpd.cpp`, `npt.cpp` | Langevin/DPD/Brownian/NPT |
| Hydrodynamics | `lb/`, `ek/` + `src/walberla_bridge/` | Lattice-Boltzmann / electrokinetics (separate, heavier build path) |
| Rigid/virtual | `virtual_sites/`, `virtual_sites.cpp`, `rotation.cpp` | VIRTUAL_SITES_RELATIVE underpins magnetics |
| Measurement | `observables/`, `accumulators/`, `analysis/`, `cluster_analysis/`, `pair_criteria/` | |
| Parallel | `communication.cpp`, `MpiCallbacks.hpp`, `ghosts.cpp`, `cuda/` | MPI + GPU |

Most non-obvious: the MPI parameter-sync layer (below), long-range solvers (P3M/DP3M + FFT/HeFFTe/ScaFaCoS + GPU),
and the `features.def` `implies`/`requires` graph gating what compiles.

## The Python ↔ C++ binding seam (the pressomancy connection point)
Three cooperating layers (full pattern in `AGENT.md` §Script Interface):
1. **`src/python/espressomd/`** — user-facing package (`system.py`, `magnetostatics.py`, `particle_data.py`, …).
2. **Cython bridge** — `script_interface.pyx/.pxd` (`ScriptInterfaceHelper`, `@script_interface_register`).
3. **`src/script_interface/`** — C++ interface classes deriving `AutoParameters` (via `ObjectHandle`);
   `Context`/`GlobalContext`/`ContextManager` synchronize parameters across MPI ranks; `initialize.cpp` registers all classes.

Pattern: one Python class → one `AutoParameters`-derived C++ interface class → wraps the actual core object;
params registered with `add_parameters()` auto-sync across ranks. This registration/sync machinery is where
binding bugs hide. `pressomancy` builds simulations entirely through layer 1 (the `espressomd` Python API).
