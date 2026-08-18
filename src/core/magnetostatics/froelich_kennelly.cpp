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

#ifdef ESPRESSO_FROELICH_KENNELLY

#include "Particle.hpp"
#include "magnetostatics/magnetodynamics.hpp"

#include <utils/Vector.hpp>
#include <utils/math/quaternion.hpp>

#include <cmath>

constexpr static double field_zero2 = 1e-10;

void froelich_kennelly(Particle &p, Utils::Vector3d const &total_field) {
  auto const field_norm2 = total_field.norm2();
  auto const dipm_sat = p.dipm_sat();

  if (field_norm2 < field_zero2 or dipm_sat <= 0.) {
    p.dipm() = 0.;
    return;
  }
  auto const field_norm = std::sqrt(field_norm2);
  auto const xi0 = p.mag_susc_0();

  auto const pre = xi0 * dipm_sat / (dipm_sat + xi0 * field_norm);

  p.dipm() = pre * field_norm;
  p.quat() = Utils::convert_director_to_quaternion(total_field);
}

#endif // ESPRESSO_FROELICH_KENNELLY
