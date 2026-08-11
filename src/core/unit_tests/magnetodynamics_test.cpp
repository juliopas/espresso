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

#define BOOST_TEST_MODULE "Magnetodynamics field-response models test"
#define BOOST_TEST_DYN_LINK

#include <config/config.hpp>

#if defined(ESPRESSO_LANGEVIN_MAGNETIZATION) ||                                \
    defined(ESPRESSO_FROELICH_KENNELLY)

#include <boost/test/unit_test.hpp>

#include "Particle.hpp"
#include "magnetostatics/magnetodynamics.hpp"

#include <utils/Vector.hpp>

#include <cmath>
#include <limits>

auto constexpr tol = 5. * 100. * std::numeric_limits<double>::epsilon();

/** @brief Langevin function @f$ L(x) = \coth(x) - 1/x @f$. */
static auto langevin(double x) { return 1. / std::tanh(x) - 1. / x; }

/** @brief Set up a magnetizable particle with the given model parameters. */
static auto make_particle(double dipm_sat, double mag_susc_0) {
  auto p = Particle();
  p.dipm_sat() = dipm_sat;
  p.mag_susc_0() = mag_susc_0;
  return p;
}

#ifdef ESPRESSO_LANGEVIN_MAGNETIZATION

BOOST_AUTO_TEST_CASE(langevin_magnetization_curve) {
  auto constexpr dipm_sat = 2.;
  auto constexpr chi0 = 0.5;
  // well away from the series threshold, the closed form must hold exactly
  for (auto const h : {0.5, 1., 3., 10.}) {
    auto p = make_particle(dipm_sat, chi0);
    langevin_magnetization(p, {0., 0., h});
    auto const alpha = 3. * chi0 * h / dipm_sat;
    BOOST_CHECK_CLOSE(p.dipm(), dipm_sat * langevin(alpha), tol);
  }
}

BOOST_AUTO_TEST_CASE(langevin_magnetization_series_branch) {
  auto constexpr dipm_sat = 2.;
  auto constexpr chi0 = 0.5;
  // alpha = 3 * chi0 * h / dipm_sat = 1e-2 is the branch point; the series and
  // the closed form must agree across it to well within the series truncation
  // error, which is O(alpha^5) ~ 1e-10 relative.
  auto const h_switch = 1e-2 * dipm_sat / (3. * chi0);
  auto p_below = make_particle(dipm_sat, chi0);
  auto p_above = make_particle(dipm_sat, chi0);
  langevin_magnetization(p_below, {0., 0., h_switch * (1. - 1e-9)});
  langevin_magnetization(p_above, {0., 0., h_switch * (1. + 1e-9)});
  BOOST_CHECK_CLOSE(p_below.dipm(), p_above.dipm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(langevin_magnetization_linear_response) {
  auto constexpr dipm_sat = 2.;
  auto constexpr chi0 = 0.5;
  // for alpha << 1, L(alpha) -> alpha/3, hence dipm -> chi0 * |H|
  auto constexpr h = 1e-4;
  auto p = make_particle(dipm_sat, chi0);
  langevin_magnetization(p, {0., 0., h});
  BOOST_CHECK_CLOSE(p.dipm(), chi0 * h, 1e-6);
}

BOOST_AUTO_TEST_CASE(langevin_magnetization_saturation) {
  auto constexpr dipm_sat = 2.;
  auto p = make_particle(dipm_sat, 0.5);
  langevin_magnetization(p, {0., 0., 1e6});
  BOOST_CHECK_CLOSE(p.dipm(), dipm_sat, 1e-3);
  BOOST_CHECK_LT(p.dipm(), dipm_sat);
}

BOOST_AUTO_TEST_CASE(langevin_magnetization_guards) {
  // vanishing field
  auto p_zero = make_particle(2., 0.5);
  p_zero.dipm() = 1.;
  langevin_magnetization(p_zero, {0., 0., 0.});
  BOOST_CHECK_EQUAL(p_zero.dipm(), 0.);
  // non-positive saturation moment
  auto p_sat = make_particle(0., 0.5);
  p_sat.dipm() = 1.;
  langevin_magnetization(p_sat, {0., 0., 1.});
  BOOST_CHECK_EQUAL(p_sat.dipm(), 0.);
  // fields below the cutoff of 1e-5 are treated as vanishing
  auto p_tiny = make_particle(2., 0.5);
  p_tiny.dipm() = 1.;
  langevin_magnetization(p_tiny, {0., 0., 9e-6});
  BOOST_CHECK_EQUAL(p_tiny.dipm(), 0.);
  auto p_above = make_particle(2., 0.5);
  langevin_magnetization(p_above, {0., 0., 2e-5});
  BOOST_CHECK_GT(p_above.dipm(), 0.);
}

BOOST_AUTO_TEST_CASE(langevin_magnetization_direction) {
  auto p = make_particle(2., 0.5);
  auto const field = Utils::Vector3d{1., -2., 3.};
  langevin_magnetization(p, field);
  // the moment must point along the total field
  auto const dip = p.calc_dip();
  BOOST_CHECK_CLOSE(dip.norm(), p.dipm(), tol);
  for (auto i = 0u; i < 3u; ++i) {
    BOOST_CHECK_CLOSE(dip[i], p.dipm() * field.normalized()[i], 1e-8);
  }
}

#endif // ESPRESSO_LANGEVIN_MAGNETIZATION

#ifdef ESPRESSO_FROELICH_KENNELLY

BOOST_AUTO_TEST_CASE(froelich_kennelly_curve) {
  auto constexpr dipm_sat = 2.;
  auto constexpr chi0 = 0.5;
  for (auto const h : {0.1, 1., 10.}) {
    auto p = make_particle(dipm_sat, chi0);
    froelich_kennelly(p, {0., 0., h});
    BOOST_CHECK_CLOSE(p.dipm(), chi0 * dipm_sat * h / (dipm_sat + chi0 * h),
                      tol);
  }
}

BOOST_AUTO_TEST_CASE(froelich_kennelly_limits) {
  auto constexpr dipm_sat = 2.;
  auto constexpr chi0 = 0.5;
  // linear response at small field; the leading correction is of relative
  // order chi0 * h / dipm_sat, hence the loosened tolerance
  auto p_lin = make_particle(dipm_sat, chi0);
  froelich_kennelly(p_lin, {0., 0., 1e-4});
  BOOST_CHECK_CLOSE(p_lin.dipm(), chi0 * 1e-4, 1e-2);
  // saturation at large field
  auto p_sat = make_particle(dipm_sat, chi0);
  froelich_kennelly(p_sat, {0., 0., 1e9});
  BOOST_CHECK_CLOSE(p_sat.dipm(), dipm_sat, 1e-3);
  BOOST_CHECK_LT(p_sat.dipm(), dipm_sat);
}

BOOST_AUTO_TEST_CASE(froelich_kennelly_guards) {
  auto p_zero = make_particle(2., 0.5);
  p_zero.dipm() = 1.;
  froelich_kennelly(p_zero, {0., 0., 0.});
  BOOST_CHECK_EQUAL(p_zero.dipm(), 0.);
  auto p_sat = make_particle(0., 0.5);
  p_sat.dipm() = 1.;
  froelich_kennelly(p_sat, {0., 0., 1.});
  BOOST_CHECK_EQUAL(p_sat.dipm(), 0.);
}

#endif // ESPRESSO_FROELICH_KENNELLY

#else // no field-response model compiled in

int main(int, char **) {}

#endif
