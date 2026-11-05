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

/** @file
 *  Dispersion pass of the dispersive harmonic bonds
 *  (@ref DispersiveHarmonicBond): computes, for every particle i,
 *  the mean a_i of s_b = q k r0 (l - r0) over its dispersive bonds b.
 *  The force and pressure kernels read a_i + a_j for each bond.
 */

#include <config/config.hpp>

#ifdef ESPRESSO_DISPERSIVE_BONDS

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cstddef>
#include <optional>

namespace System {
class System;
}

/** @brief Buffers of the dispersion pass, indexed like the unique particles
 *  of @ref CellStructure.
 */
struct BondDispersionState {
  using memory_space = Kokkos::HostSpace;
  using SumType = Kokkos::View<double *[2], Kokkos::LayoutRight, memory_space>;
  using ScatterSum =
      Kokkos::Experimental::ScatterView<double *[2], Kokkos::LayoutRight,
                                        memory_space>;
  using MeanType = Kokkos::View<double *, Kokkos::LayoutRight, memory_space>;

  /** per particle: {sum of s_b, number of dispersive bonds} */
  SumType sum;
  std::optional<ScatterSum> scatter_sum;
  /** per particle: mean dispersion term a_i */
  MeanType a;

  /** @brief Reallocate the buffers if the number of particles changed. */
  void resize(std::size_t n_part);

  /** @brief Deallocate the buffers. */
  void clear();
};

/**
 * @brief Compute the mean dispersion term a_i of every local and ghost
 * particle into @ref BondDispersionState::a.
 *
 * Must run after the AoSoA positions and bond lists are up to date and
 * before the bond force or pressure kernels copy the view handle.
 * Collective: every rank enters the ghost exchanges whenever any dispersive
 * bond type exists, whether or not it holds such bonds locally.
 */
void calc_bond_dispersion(System::System const &system);

#endif // ESPRESSO_DISPERSIVE_BONDS
