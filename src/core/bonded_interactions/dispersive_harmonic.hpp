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

#pragma once

/** \file
 *  Routines to calculate the dispersive harmonic bond (Kot and Nagahashi)
 *  between particle pairs.
 *
 *  Each bond b = (i, j) of length l carries the harmonic spring plus a
 *  tension (a_i + a_j) / r0, where a_i is the mean of
 *  s_b = q k r0 (l - r0) over all dispersive bonds of particle i.
 *  The means a_i are computed by a separate pass over all bonds
 *  (see dispersive_bonds.hpp) before the force kernel runs.
 *  The energy is the harmonic spring only.
 */

#include "config/config.hpp"
#include "errorhandling.hpp"

#include <utils/Vector.hpp>
#include <utils/math/sqr.hpp>

#include <optional>
#include <stdexcept>

/** Parameters for the dispersive harmonic bond */
struct DispersiveHarmonicBond {
  /** spring constant */
  double k;
  /** equilibrium bond length */
  double r;
  /** cutoff length */
  double r_cut;
  /** dispersion coefficient */
  double q;

  double cutoff() const { return r_cut; }

  static constexpr int num = 1;

  DispersiveHarmonicBond(double k, double r, double r_cut, double q) {
    if (r <= 0.) {
      throw std::domain_error(
          "DispersiveHarmonicBond parameter 'r_0' has to be > 0");
    }
    if (r_cut < 0.) {
      throw std::domain_error(
          "DispersiveHarmonicBond parameter 'r_cut' has to be >= 0 (a "
          "negative r_cut would make espresso skip the bond loop)");
    }
    this->k = k;
    this->r = r;
    this->r_cut = r_cut;
    this->q = q;
  }

  std::optional<Utils::Vector3d> force(Utils::Vector3d const &dx,
                                       double a12) const;
  std::optional<double> energy(Utils::Vector3d const &dx) const;
};

/** Compute the dispersive harmonic bond force.
 *  @param[in]  dx        Distance between the particles.
 *  @param[in]  a12       Sum of the mean dispersion terms of both particles.
 */
inline std::optional<Utils::Vector3d>
DispersiveHarmonicBond::force(Utils::Vector3d const &dx,
                              double const a12) const {
  auto const dist = dx.norm();

  if ((r_cut > 0.0) && (dist > r_cut)) {
    return {};
  }

  auto fac = -(k * (dist - r) + a12 / r);
  if (dist > round_error_prec) { /* Regular case */
    fac /= dist;
  } else {
    runtimeErrorMsg() << "Dispersive harmonic bond: Particles have zero "
                         "distance. This is most likely an error in the "
                         "system setup.";
  }
  return fac * dx;
}

/** Compute the dispersive harmonic bond energy (spring only).
 *  @param[in]  dx        Distance between the particles.
 */
inline std::optional<double>
DispersiveHarmonicBond::energy(Utils::Vector3d const &dx) const {
  auto const dist = dx.norm();

  if ((r_cut > 0.0) && (dist > r_cut)) {
    return {};
  }

  return 0.5 * k * Utils::sqr(dist - r);
}
