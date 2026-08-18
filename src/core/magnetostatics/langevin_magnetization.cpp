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

#ifdef ESPRESSO_LANGEVIN_MAGNETIZATION

#include "Particle.hpp"
#include "magnetostatics/magnetodynamics.hpp"

#include <utils/Vector.hpp>
#include <utils/math/quaternion.hpp>

#include <cmath>

constexpr static double field_zero2 = 1e-10;
constexpr static double alpha_series_threshold = 1e-2;

void langevin_magnetization(Particle &p, Utils::Vector3d const &total_field) {
  auto const field_norm2 = total_field.norm2();
  auto const dipm_sat = p.dipm_sat();

  if (field_norm2 < field_zero2 or dipm_sat <= 0.) {
    p.dipm() = 0.;
    return;
  }
  auto const field_norm = std::sqrt(field_norm2);

  auto const alpha = 3. * p.mag_susc_0() * field_norm / dipm_sat;
  double langevin;
  if (alpha < alpha_series_threshold) {
    langevin = alpha / 3. - alpha * alpha * alpha / 45.;
  } else {
    langevin = 1. / std::tanh(alpha) - 1. / alpha;
  }

  p.dipm() = dipm_sat * langevin;
  p.quat() = Utils::convert_director_to_quaternion(total_field);
}

#endif // ESPRESSO_LANGEVIN_MAGNETIZATION
