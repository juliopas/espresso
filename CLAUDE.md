# espresso (ESPResSo) — C++ MD engine

Soft-matter molecular-dynamics framework (MD, MC, P3M electrostatics, Lattice-Boltzmann,
electrokinetics). Simulations are driven from Python via the `espressomd` module.

- **This checkout is a fork**: `fork` = `juliopas/espresso`, `origin` = upstream `espressomd/espresso`.
- **Working branch: `magnetodynamics`** (on top of upstream 5.0). Active work = magnetics/magnetodynamics.
  ⚠ Upstream `CONTRIBUTING.md` says PRs target the `python` branch — that's upstream convention; **we work on `magnetodynamics`**, so don't assume upstream branch/PR rules here.

## Reference & routing
| Need | Where |
|---|---|
| Build, test, formatting, key headers, script-interface pattern | **`AGENT.md`** (detailed reference — read on demand, it's not auto-loaded) |
| Magnetics / magnetodynamics (the active area) | `.claude/context/magnetics.md` |
| `src/core/` subsystem map + the Python↔C++ binding seam | `.claude/context/core-architecture.md` |

## Essentials (details in AGENT.md)
- Build: out-of-source in `build/`; `cd build && make -j$(nproc)`. **Don't re-run `cmake`** unless build files are missing or the user asks.
- Run/test: `cd build && ./pypresso ../testsuite/python/<test>.py` (serial) or `mpirun -n 4 ./pypresso ...` (parallel); aggregate `make check`. C++ unit tests in `src/core/unit_tests/`.
- Format before done: `maintainer/format/clang-format.sh -i <file.cpp>`, `maintainer/format/autopep8.sh -i <file.py>`. Every source file carries the GPL header.
- Features compiled in are selected by `myconfig.hpp` (presets in `maintainer/configs/`, defs in `src/config/features.def`).

## Cross-links
- **pressomancy** (../pressomancy) drives this engine through the Python API / `script_interface` layer —
  see core-architecture.md for that seam.
- **SCRIPTS/dev_espresso** (../SCRIPTS/dev_espresso) holds your own test scripts + benchmarks for changes to
  this engine: after editing C++, **rebuild** (`cd build && make -j`) then run them via `build/pypresso`.
  (From the espresso workspace, `/add-dir ../SCRIPTS` to reach them.)

**Plans:** tag each step `[quick]`/`[impl]`/`[impl-opus]` (tiers in `~/.claude/CLAUDE.md`).
Git remote/sync ops are the user's only (see `~/.claude/CLAUDE.md`).
