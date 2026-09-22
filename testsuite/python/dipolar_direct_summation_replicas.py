#
# Copyright (C) 2010-2026 The ESPResSo project
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
import espressomd
import espressomd.magnetostatics
import numpy as np
import unittest as ut
import unittest_decorators as utx

N_PART = 6
BOX_L = [5., 5., 9.]


@utx.skipIfMissingFeatures(["DIPOLES"])
class Test(ut.TestCase):

    """
    Check the energy of the dipolar direct sum with periodic replicas
    against an explicit lattice sum. The energy per simulation cell is
    ``E = 1/2 sum_i sum_j sum_n' U_ij(n)``: unordered pairs ``i != j``
    are counted once, but the interaction of a particle with its own
    periodic images has to be counted with a factor ``1/2``.
    """

    system = espressomd.System(box_l=BOX_L)

    system.time_step = 0.01
    system.cell_system.skin = 0.1

    def setUp(self):
        self.system.box_l = BOX_L

    def tearDown(self):
        self.system.part.clear()
        self.system.magnetostatics.clear()
        self.system.periodicity = [True, True, True]

    def add_particles(self):
        rng = np.random.default_rng(seed=3)
        box_l = np.copy(self.system.box_l)
        pos = rng.uniform(0.5, box_l - 0.5, (N_PART, 3))
        dip = rng.normal(size=(N_PART, 3))
        dip /= np.linalg.norm(dip, axis=1)[:, np.newaxis]
        self.system.part.add(pos=pos, dip=dip,
                             rotation=N_PART * [(True, True, True)])
        return pos, dip

    @staticmethod
    def image_indices(periodicity, n_replicas):
        """
        Image set of the core: ``get_n_cut()`` returns
        ``ncut = n_replicas * periodicity`` and ``for_each_image()`` visits
        the index cube ``[-ncut, ncut]`` clipped by the ball
        ``nx^2 + ny^2 + nz^2 <= |ncut|^2``.
        """
        ncut = n_replicas * np.array(periodicity, dtype=int)
        ncut2 = int(ncut @ ncut)
        return [(nx, ny, nz)
                for nx in range(-ncut[0], ncut[0] + 1)
                for ny in range(-ncut[1], ncut[1] + 1)
                for nz in range(-ncut[2], ncut[2] + 1)
                if nx**2 + ny**2 + nz**2 <= ncut2]

    @staticmethod
    def pair_potential(d, m1, m2):
        r = np.linalg.norm(d)
        return np.dot(m1, m2) / r**3 - \
            3. * np.dot(m1, d) * np.dot(m2, d) / r**5

    def reference_energy(self, pos, dip, periodicity, n_replicas):
        """
        Lattice sum over the image set of the core, with at least one
        periodic direction and ``n_replicas >= 1``, i.e. the core uses
        Euclidean distances instead of the minimum image convention.
        Returns the pair contribution and the self-image contribution.
        """
        assert n_replicas >= 1 and any(periodicity)
        box_l = np.copy(self.system.box_l)
        indices = self.image_indices(periodicity, n_replicas)
        energy_pairs = 0.
        energy_self = 0.
        for i in range(len(pos)):
            for index in indices:
                if index == (0, 0, 0):
                    continue
                shift = np.array(index) * box_l
                energy_self += 0.5 * \
                    self.pair_potential(shift, dip[i], dip[i])
            for j in range(i + 1, len(pos)):
                for index in indices:
                    d = pos[i] - pos[j] + np.array(index) * box_l
                    energy_pairs += self.pair_potential(d, dip[i], dip[j])
        return energy_pairs, energy_self

    def check_energy(self, tol, **kwargs):
        system = self.system
        for periodicity in ([True, True, True], [True, True, False]):
            for n_replicas in [1, 2]:
                with self.subTest(periodicity=periodicity,
                                  n_replicas=n_replicas):
                    system.periodicity = periodicity
                    pos, dip = self.add_particles()
                    system.magnetostatics.solver = \
                        espressomd.magnetostatics.DipolarDirectSum(
                            prefactor=1., n_replicas=n_replicas, **kwargs)
                    system.integrator.run(0, recalc_forces=True)
                    energy = system.analysis.energy()["dipolar", 1]
                    energy_pairs, energy_self = self.reference_energy(
                        pos, dip, periodicity, n_replicas)
                    # the self-image term is sizeable in a non-cubic box
                    self.assertGreater(abs(energy_self), 1e-2)
                    np.testing.assert_allclose(
                        energy, energy_pairs + energy_self, **tol)
                system.part.clear()
                system.magnetostatics.clear()

    def test_energy_cpu(self):
        self.check_energy(tol={"rtol": 1e-9, "atol": 0.})

    @utx.skipIfMissingGPU()
    @utx.skipIfMissingFeatures("CUDA")
    def test_energy_gpu(self):
        self.check_energy(tol={"rtol": 1e-4, "atol": 1e-6}, gpu=True)

    def test_cubic_box_cpu(self):
        """
        In a cubic box the self-image term vanishes by symmetry, i.e. the
        energy is insensitive to the weight given to the self-images.
        """
        system = self.system
        system.box_l = [5., 5., 5.]
        system.periodicity = [True, True, True]
        pos, dip = self.add_particles()
        for n_replicas in [1, 2]:
            with self.subTest(n_replicas=n_replicas):
                system.magnetostatics.solver = \
                    espressomd.magnetostatics.DipolarDirectSum(
                        prefactor=1., n_replicas=n_replicas)
                system.integrator.run(0, recalc_forces=True)
                energy = system.analysis.energy()["dipolar", 1]
                energy_pairs, energy_self = self.reference_energy(
                    pos, dip, [True, True, True], n_replicas)
                self.assertAlmostEqual(energy_self, 0., delta=1e-12)
                np.testing.assert_allclose(energy, energy_pairs, rtol=1e-9)
                system.magnetostatics.clear()


if __name__ == "__main__":
    ut.main()
