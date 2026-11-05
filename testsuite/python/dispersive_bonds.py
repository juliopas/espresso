#
# Copyright (C) 2026 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import itertools
import unittest as ut
import unittest_decorators as utx
import numpy as np

import espressomd
import espressomd.interactions

BOX_L = 12.
TOL = 1e-10

# Network around the origin, in x, y < 6: on the 4-rank node grid [2, 2, 1]
# the x, y > 6 domain holds no particle. Bonds cross the periodic boundaries
# in x, y and z (z is the axis where the node grid is 1); in x and y these
# are also rank boundaries. Bead 5 also carries the plain HarmonicBond.
# The ghost layer (2) is narrower than the far side of the box (12 - 6), so
# a drifting network changes the ghost count of every rank.
POS = np.array([
    (0.5, 0.6, -0.3), (0.7, 0.9, 0.6), (1.6, 0.4, 0.3), (-0.4, 0.8, 0.2),
    (0.4, -0.5, -0.6), (2.6, 1.0, 0.8), (3.5, 1.9, 1.0), (4.4, 1.6, 1.3),
    (3.4, 3.1, 1.1), (3.6, 4.2, 1.5), (5.3, 1.2, 1.6), (2.4, 1.8, -0.4)])
KINDS = {"A": dict(k=30., r_0=1.0, q=0.4),
         "B": dict(k=55., r_0=1.2, q=-0.25),
         "C": dict(k=20., r_0=0.9, q=1.3),
         "harmonic": dict(k=40., r_0=1.1)}
NETWORK = [(0, 1, "A"), (1, 2, "B"), (0, 2, "A"), (1, 3, "C"), (0, 3, "B"),
           (0, 4, "C"), (2, 5, "B"), (5, 6, "A"), (6, 7, "C"), (6, 8, "B"),
           (8, 9, "A"), (7, 10, "C"), (5, 11, "harmonic")]
# jittered position sets: non-uniform strains, different at each evaluation
POS_SETS = [POS + np.random.default_rng(seed).uniform(-0.15, 0.15, POS.shape)
            for seed in (1, 2)]


def reference(pos):
    """
    Forces and bonded pressure tensor of the Kot dispersive bonds (and the
    plain harmonic bond, which neither counts in n nor gets a Kot term).
    """
    P, n = np.zeros(len(pos)), np.zeros(len(pos))
    geometry = []
    for i, j, kind in NETWORK:
        d = pos[i] - pos[j]
        d -= BOX_L * np.round(d / BOX_L)
        length = np.linalg.norm(d)
        geometry.append((d, length))
        if kind != "harmonic":
            k, r0, q = KINDS[kind]["k"], KINDS[kind]["r_0"], KINDS[kind]["q"]
            P[[i, j]] += q * k * r0 * (length - r0)
            n[[i, j]] += 1.
    a = np.divide(P, n, out=np.zeros_like(P), where=n > 0.)
    forces, tensor = np.zeros_like(pos), np.zeros((3, 3))
    for (i, j, kind), (d, length) in zip(NETWORK, geometry):
        k, r0 = KINDS[kind]["k"], KINDS[kind]["r_0"]
        tension = 0. if kind == "harmonic" else (a[i] + a[j]) / r0
        f = -(k * (length - r0) + tension) / length * d
        forces[i] += f
        forces[j] -= f
        tensor += np.outer(d, f)
    return forces, tensor / BOX_L**3


@utx.skipIfMissingFeatures(["DISPERSIVE_BONDS"])
class DispersiveBonds(ut.TestCase):
    system = espressomd.System(box_l=3 * [BOX_L])
    system.time_step = 0.01
    system.cell_system.skin = 0.3
    system.min_global_cut = 1.7
    if system.cell_system.get_state()["n_nodes"] == 4:
        system.cell_system.node_grid = [2, 2, 1]  # z: the axis with 1 rank

    def tearDown(self):
        self.system.part.clear()
        self.system.bonded_inter.clear()
        self.system.box_l = 3 * [BOX_L]
        self.system.cell_system.set_regular_decomposition()

    def build_network(self):
        bonds = {}
        for kind, params in KINDS.items():
            bonds[kind] = (espressomd.interactions.HarmonicBond(**params)
                           if kind == "harmonic" else
                           espressomd.interactions.DispersiveHarmonicBond(
                               **params))
            self.system.bonded_inter.add(bonds[kind])
        partcls = self.system.part.add(pos=POS)
        p = [self.system.part.by_id(pid) for pid in partcls.id]
        for i, j, kind in NETWORK:
            p[i].add_bond((bonds[kind], p[j]))
        return partcls

    def decompositions(self):
        return (self.system.cell_system.set_regular_decomposition,
                self.system.cell_system.set_n_square)

    def test_forces(self):
        partcls = self.build_network()
        for decomposition in self.decompositions():
            with self.subTest(decomposition=decomposition.__name__):
                decomposition()
                for pos in POS_SETS:
                    partcls.pos = pos
                    # run() drops the pass-1 buffers when it starts: only
                    # the later steps of a run reuse them
                    for steps in (0, 3):
                        self.system.integrator.run(steps, recalc_forces=True)
                        ref = reference(partcls.pos)[0]
                        np.testing.assert_allclose(
                            partcls.f, ref, rtol=0.,
                            atol=TOL * np.max(np.abs(ref)))
                # a drifting network: resorts and Verlet rebuilds in mid-run
                # change the ghost count, so pass 1 must resize its buffers
                # (bites on 4 ranks only: 1 rank has no ghosts)
                partcls.v = (3., 2., 1.)
                self.system.integrator.run(100)
                ref = reference(partcls.pos)[0]
                np.testing.assert_allclose(
                    partcls.f, ref, rtol=0., atol=TOL * np.max(np.abs(ref)))

    def test_parameters(self):
        # values for which q k r0 / (k r0) != q in floating point
        params = dict(k=7., r_0=1.1, r_cut=0., q=0.3)
        bond = espressomd.interactions.DispersiveHarmonicBond(**params)
        self.assertEqual(bond.params, params)
        for invalid in (dict(r_0=0.), dict(r_cut=-1.)):
            with self.assertRaises(ValueError):
                espressomd.interactions.DispersiveHarmonicBond(
                    **{**params, **invalid})

    def test_pressure(self):
        """Pressure recomputes the tensions at the current positions."""
        partcls = self.build_network()
        for decomposition in self.decompositions():
            with self.subTest(decomposition=decomposition.__name__):
                decomposition()
                partcls.pos = POS_SETS[0]
                self.system.integrator.run(0, recalc_forces=True)
                partcls.pos = POS_SETS[1]
                ref = reference(POS_SETS[1])[1]
                np.testing.assert_allclose(
                    self.system.analysis.pressure_tensor()["bonded"], ref,
                    rtol=0., atol=TOL * np.max(np.abs(ref)))

    def test_affine_pressure(self):
        """
        Uniform expansion of two interpenetrating simple-cubic lattices with
        nearest-neighbour bonds, one with q and one with q = 0. All bonds
        have the same strain, so every bead has a = q k r0 (l - r0) and the
        bonded pressure grows by exactly 2q.
        """
        q, n_cells, scale = 0.4, 4, 1.07
        self.system.box_l = 3 * [n_cells * scale]
        grid = np.array(list(itertools.product(range(n_cells), repeat=3)))
        lattice_bonds = []
        for offset, bond_q in ((0., q), (0.5, 0.)):
            bond = espressomd.interactions.DispersiveHarmonicBond(
                k=10., r_0=1., q=bond_q)
            self.system.bonded_inter.add(bond)
            partcls = self.system.part.add(pos=scale * (grid + offset))
            p = [self.system.part.by_id(pid) for pid in partcls.id]
            for site, axis in itertools.product(grid, np.eye(3, dtype=int)):
                neighbour = (site + axis) % n_cells
                p[np.ravel_multi_index(site, 3 * [n_cells])].add_bond(
                    (bond, p[np.ravel_multi_index(neighbour, 3 * [n_cells])]))
            lattice_bonds.append(bond)
        tensor = self.system.analysis.pressure_tensor()
        p_q, p_0 = (tensor["bonded", bond._bond_id] for bond in lattice_bonds)
        np.testing.assert_allclose(p_q, (1. + 2. * q) * p_0, rtol=0.,
                                   atol=1e-12 * np.max(np.abs(p_0)))


if __name__ == "__main__":
    ut.main()
