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

/** @file
 *  Per-particle magnetization dynamics models. Each model updates the dipole
 *  moment (magnitude and orientation) of a magnetizable virtual particle given
 *  the local magnetic field. The shared driver
 *  @ref System::System::integrate_magnetodynamics dispatches to these helpers.
 */

#pragma once

#include <config/config.hpp>

#include <utils/Vector.hpp>

struct Particle;

#ifdef ESPRESSO_LANGEVIN_MAGNETIZATION
/**
 * @brief Langevin (superparamagnetic) magnetization response.
 *
 * Deterministic, thermostat-free field response intended for particles large
 * enough that thermal fluctuations of the moment are negligible; the response
 * is set by the user-supplied initial susceptibility @c mag_susc_0 (chi_0).
 * The reduced field is @f$ \alpha = 3\,\chi_0\,|H| / m_\mathrm{sat} @f$ and the
 * induced moment magnitude is @f$ m_\mathrm{sat}\,L(\alpha) @f$ with the
 * Langevin function @f$ L(\alpha)=\coth\alpha-1/\alpha @f$.
 *
 * @param[in,out] p Magnetizable particle to update.
 * @param total_field External homogeneous field plus total dipolar field.
 */
void langevin_magnetization(Particle &p, Utils::Vector3d const &total_field);
#endif

#ifdef ESPRESSO_FROELICH_KENNELLY
/**
 * @brief Froelich-Kennelly magnetization response.
 *
 * Deterministic, thermostat-free field response with prefactor
 * @f$ \chi_0 m_\mathrm{sat} / (m_\mathrm{sat} + \chi_0 |H|) @f$.
 *
 * @param[in,out] p Magnetizable particle to update.
 * @param total_field External homogeneous field plus total dipolar field.
 */
void froelich_kennelly(Particle &p, Utils::Vector3d const &total_field);
#endif

#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
/**
 * @brief Simplified thermal Stoner-Wohlfarth update in the field-free case.
 * @see stoner_wohlfarth_thermal.cpp
 */
void stoner_wohlfarth_no_field(Particle &p, Utils::Vector3d const &e_k,
                               double kT, double noise);
/**
 * @brief Full in-field thermal Stoner-Wohlfarth update (with kinetic MC step).
 * @see stoner_wohlfarth_thermal.cpp
 */
void stoner_wohlfarth_main(Particle &p, Utils::Vector3d const &e_k,
                           Utils::Vector3d const &ext_fld_dpl, double kT,
                           double noise);
#endif
