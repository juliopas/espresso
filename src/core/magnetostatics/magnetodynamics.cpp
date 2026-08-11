/*
 * Copyright (C) 2026 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <config/config.hpp>

#if defined(ESPRESSO_LANGEVIN_MAGNETIZATION) ||                                \
    defined(ESPRESSO_FROELICH_KENNELLY) ||                                     \
    defined(ESPRESSO_THERMAL_STONER_WOHLFARTH)

#include "Particle.hpp"
#include "cell_system/CellStructure.hpp"
#include "cell_system/for_each_particle.hpp"
#include "constraints/Constraints.hpp"
#include "constraints/HomogeneousMagneticField.hpp"
#include "magnetostatics/magnetodynamics.hpp"
#include "system/System.hpp"

#include <utils/Vector.hpp>

#include <memory>

#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
#include "random.hpp"
#include "thermostat.hpp"
#include "virtual_sites/relative.hpp"

#include <utils/uniform.hpp>

#include <cassert>
#endif

/**
 * @brief Collect the external homogeneous magnetic field from active
 * constraints.
 *
 * @return The total external homogeneous magnetic field.
 */
static auto get_external_field(Constraints::Constraints const &constraints) {
  using HomogeneousMagneticField = ::Constraints::HomogeneousMagneticField;
  Utils::Vector3d ext_fld = {0., 0., 0.};
  for (auto const &constraint : constraints) {
    if (auto const ptr =
            std::dynamic_pointer_cast<HomogeneousMagneticField>(constraint)) {
      ext_fld += ptr->H();
    }
  }
  return ext_fld;
}

/**
 * @brief Run the magnetization dynamics update for local virtual particles.
 *
 * Iterate over local particles carrying a dipole moment and update it
 * according to the enabled magnetization model. The effective field is the sum
 * of the external homogeneous field and the total dipolar field @ref
 * Particle::dip_fld.
 *
 * Deterministic field-response models (Langevin, Froelich-Kennelly) run on
 * every call. Stochastic/thermal models (thermal Stoner-Wohlfarth) advance a
 * kinetic process by one time step and therefore run only inside the
 * integration loop, i.e. when @p initial_step is @c false.
 *
 * @note @ref Particle::dip_fld is only populated by @ref DipolarDirectSum
 * (CPU and GPU). With any other dipolar solver it stays zero, so the models
 * respond to the external field alone. It also lags by one time step, since
 * it is recomputed in @ref System::calculate_forces after this call, and is
 * zero on the pre-loop call.
 *
 * @param initial_step Whether this is the pre-loop initial-force evaluation.
 */
void System::System::integrate_magnetodynamics(bool initial_step) {
  auto const ext_fld = get_external_field(*constraints);
#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
  auto const kT = thermostat->kT;
#endif
  cell_structure->for_each_local_particle([&](Particle &p) {
#ifdef ESPRESSO_LANGEVIN_MAGNETIZATION
    if (p.langevin_magnetization_is_enabled()) {
      if (p.is_virtual()) {
        langevin_magnetization(p, ext_fld + p.dip_fld());
      }
      return;
    }
#endif
#ifdef ESPRESSO_FROELICH_KENNELLY
    if (p.froelich_kennelly_is_enabled()) {
      if (p.is_virtual()) {
        froelich_kennelly(p, ext_fld + p.dip_fld());
      }
      return;
    }
#endif

    // --- stochastic/thermal models (advance in time; in-loop only) ---
    if (initial_step) {
      return;
    }
#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
    if (p.is_virtual() and p.stoner_wohlfarth_is_enabled()) {
      auto *p_ref = get_reference_particle(*cell_structure, p);
      if (not p_ref) {
        return;
      }
      assert(thermostat->thermo_switch & THERMO_LANGEVIN);
      auto const &langevin = *thermostat->langevin;
      auto const e_k = p_ref->calc_director();
      auto const ext_fld_dpl = ext_fld + p.dip_fld();
      auto const random_ints =
          Random::philox_4_uint64s<RNGSalt::THERMAL_STONER_WOHLFARTH>(
              langevin.rng_counter(), langevin.rng_seed(), p.id());
      auto const noise = Utils::uniform(random_ints[0]);
      if (ext_fld_dpl.norm2() == 0.) {
        stoner_wohlfarth_no_field(p, e_k, kT, noise);
      } else {
        // full Stoner-Wohlfarth update with external + dipolar field
        stoner_wohlfarth_main(p, e_k, ext_fld_dpl, kT, noise);
      }
    }
#endif
  });
}

#endif // magnetization dynamics models
