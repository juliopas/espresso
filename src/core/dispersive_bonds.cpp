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

#ifdef ESPRESSO_DISPERSIVE_BONDS

#include "dispersive_bonds.hpp"

#include "BoxGeometry.hpp"
#include "Particle.hpp"
#include "aosoa_pack.hpp"
#include "bonded_interactions/bonded_interaction_data.hpp"
#include "cell_system/CellStructure.hpp"
#include "cell_system/LocalBondState.hpp"
#include "cell_system/for_each_particle.hpp"
#include "kokkos_helpers.hpp"
#include "system/System.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cstddef>
#include <variant>

void BondDispersionState::resize(std::size_t const n_part) {
  if (not scatter_sum or sum.extent(0) != n_part) {
    sum = SumType("bond_dispersion_sum", n_part);
    a = MeanType("bond_dispersion_mean", n_part);
    // underlying View changed -> scratch buffers must be rebuilt
    scatter_sum.emplace(Kokkos::Experimental::create_scatter_view(sum));
  }
}

void BondDispersionState::clear() {
  scatter_sum.reset();
  sum = SumType();
  a = MeanType();
}

void calc_bond_dispersion(System::System const &system) {
  if (system.bonded_ias->get_n_dispersive_bonds() == 0) {
    return;
  }
  using execution_space = Kokkos::DefaultHostExecutionSpace;
  auto &cell_structure = *system.cell_structure;
  auto &state = cell_structure.bond_dispersion_state();
  auto const &unique_particles = cell_structure.get_unique_particles();
  auto const n_part = unique_particles.size();
  state.resize(n_part);
  kokkos_deep_copy(execution_space{}, state.sum, 0.);
  state.scatter_sum->reset();

  // Sum s_b and count the dispersive bonds of each local and ghost particle.
  // Every bond is stored on exactly one rank, so it is visited once.
  {
    auto const &bonded_ias = *system.bonded_ias;
    auto const &box_geo = *system.box_geo;
    auto const &aosoa = cell_structure.get_aosoa();
    auto const &bs = cell_structure.bond_state();
    auto const scatter_sum = *state.scatter_sum;
    kokkos_parallel_range_for<execution_space>(
        "bond_dispersion_sum", std::size_t{0},
        static_cast<std::size_t>(bs.pair_count),
        [&bonded_ias, &box_geo, &aosoa, &bs, scatter_sum](std::size_t idx) {
          auto const *iap = std::get_if<DispersiveHarmonicBond>(
              bonded_ias.at(bs.pair_ids(idx)).get());
          if (iap == nullptr) {
            return;
          }
          auto const i = bs.pair_list(idx, 0);
          auto const j = bs.pair_list(idx, 1);
          auto const dist =
              box_geo
                  .get_mi_vector(aosoa.get_vector_at(aosoa.position, i),
                                 aosoa.get_vector_at(aosoa.position, j))
                  .norm();
          if (iap->r_cut > 0. and dist > iap->r_cut) {
            return;
          }
          auto const s = iap->q * iap->k * iap->r * (dist - iap->r);
          auto access = scatter_sum.access();
          access(i, 0) += s;
          access(i, 1) += 1.;
          access(j, 0) += s;
          access(j, 1) += 1.;
        });
    Kokkos::Experimental::contribute(state.sum, scatter_sum);
  }

  // Ghost copies still hold the previous call's means
  // only the copies indexed in the unique particles below contribute.
  cell_structure.for_each_ghost_particle(
      [](Particle &p) { p.bond_dispersion() = {}; });
  auto const &sum = state.sum;
  kokkos_parallel_range_for<execution_space>(
      "bond_dispersion_assign", std::size_t{0}, n_part,
      [&unique_particles, &sum](std::size_t i) {
        unique_particles[i]->bond_dispersion() = {sum(i, 0), sum(i, 1)};
      });
  Kokkos::fence();

  // Every ghost is a direct copy of its owner's real particle, so the
  // reduction adds each contribution exactly once.
  cell_structure.ghosts_reduce_bond_dispersion();
  cell_structure.for_each_local_particle([](Particle &p) {
    auto &bd = p.bond_dispersion();
    bd.P = (bd.n > 0.) ? bd.P / bd.n : 0.;
  });
  cell_structure.ghosts_update(Cells::DATA_PART_BOND_DISPERSION);

  auto const &a = state.a;
  kokkos_parallel_range_for<execution_space>(
      "bond_dispersion_mean", std::size_t{0}, n_part,
      [&unique_particles, &a](std::size_t i) {
        a(i) = unique_particles[i]->bond_dispersion().P;
      });
  Kokkos::fence();
}

#endif // ESPRESSO_DISPERSIVE_BONDS
