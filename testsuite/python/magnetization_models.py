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
"""
Tests for the deterministic field-response magnetization models, i.e. the
``LANGEVIN_MAGNETIZATION`` and ``FROELICH_KENNELLY`` features. Both are
thermostat-free algebraic responses of the particle moment to the total field
``H = H_ext + dip_fld``, so they can be checked against closed-form curves.

Note that ``dip_fld`` is only populated by ``DipolarDirectSum``, and that it
lags the moment by one time step, so the mutual-magnetization case relaxes to
its self-consistent value over several steps.
"""

import numpy as np
import unittest as ut
import unittest_decorators as utx
import espressomd
import espressomd.magnetostatics
import espressomd.constraints
import espressomd.propagation

Propagation = espressomd.propagation.Propagation


def langevin_L(alpha):
    """Langevin function, matching the small-argument series used in C++."""
    if abs(alpha) < 1e-2:
        return alpha / 3. - alpha**3 / 45.
    return 1. / np.tanh(alpha) - 1. / alpha


def langevin_M(field, chi0, m_sat):
    return m_sat * langevin_L(3. * chi0 * field / m_sat)


def froelich_M(field, chi0, m_sat):
    return chi0 * m_sat * field / (m_sat + chi0 * field)


class MagnetizationCommon:
    """Setup helpers and model-agnostic checks, mixed into the test cases."""

    system = espressomd.System(box_l=[20., 20., 20.])
    m_sat = 1.
    chi0 = 1. / 3.
    prefactor = 1.
    separation = 1.3

    def setUp(self):
        system = self.system
        system.time_step = 0.01
        system.cell_system.skin = 0.4
        system.min_global_cut = 2.
        system.periodicity = [False, False, False]
        system.cell_system.set_n_square()

    def tearDown(self):
        self.system.part.clear()
        self.system.constraints.clear()
        self.system.magnetostatics.clear()

    def add_magnetizable(self, pos, model):
        """A magnetizable unit: a fixed real anchor plus a virtual site.

        The magnetization update is guarded by ``is_virtual()``, so the moment
        must live on a virtual site relative to a real particle.
        """
        anchor = self.system.part.add(pos=pos, fix=[True] * 3)
        p = self.system.part.add(pos=pos, rotation=[True] * 3,
                                 dip=[0., 0., self.m_sat])
        p.vs_auto_relate_to(anchor.id)
        p.propagation = (Propagation.TRANS_VS_RELATIVE |
                         Propagation.ROT_VS_INDEPENDENT)
        p.dipm_sat = self.m_sat
        p.mag_susc_0 = self.chi0
        setattr(p, f"{model}_is_enabled", True)
        return anchor, p

    def add_solver(self):
        # dip_fld is only populated by the direct sum
        self.system.magnetostatics.solver = \
            espressomd.magnetostatics.DipolarDirectSum(
                prefactor=self.prefactor)

    def apply_field(self, field):
        self.system.constraints.add(
            espressomd.constraints.HomogeneousMagneticField(
                H=[0., 0., field]))

    def check_curve(self, model, reference):
        """A lone magnetizable particle has ``dip_fld == 0``, so its moment
        must follow the closed-form response curve exactly."""
        center = 0.5 * self.system.box_l[0]
        # spans the small-argument series branch (alpha < 1e-2), the crossover
        # and the saturation regime
        for field in [1e-3, 5e-3, 0.05, 0.3, 1., 3., 30.]:
            self.tearDown()
            self.apply_field(field)
            _, p = self.add_magnetizable([center] * 3, model)
            self.add_solver()
            self.system.integrator.run(1)
            expected = reference(field, self.chi0, self.m_sat)
            self.assertAlmostEqual(p.dipm, expected, delta=1e-12,
                                   msg=f"{model} at H={field}")
            # the moment must be aligned with the total field, i.e. +z here
            np.testing.assert_allclose(
                np.copy(p.dip), [0., 0., expected], atol=1e-12)

    def check_zero_field(self, model):
        center = 0.5 * self.system.box_l[0]
        _, p = self.add_magnetizable([center] * 3, model)
        self.add_solver()
        self.system.integrator.run(1)
        self.assertEqual(p.dipm, 0.)

    def check_saturation(self, model):
        center = 0.5 * self.system.box_l[0]
        self.apply_field(1e6)
        _, p = self.add_magnetizable([center] * 3, model)
        self.add_solver()
        self.system.integrator.run(1)
        self.assertLess(p.dipm, self.m_sat)
        self.assertAlmostEqual(p.dipm, self.m_sat, delta=1e-4)

    def check_mutual_pair(self, model, reference):
        """Two magnetizable particles side by side in an axial field.

        Their moments are set self-consistently through the mutual dipolar
        field. Because ``dip_fld`` lags by one step, the engine performs the
        fixed-point iteration ``m -> M(|H - prefactor * m / r^3|)`` one step at
        a time; the equatorial field of the neighbour opposes the applied one.
        """
        field = 1.
        center = 0.5 * self.system.box_l[0]
        self.apply_field(field)
        _, p1 = self.add_magnetizable([center, center, center], model)
        _, p2 = self.add_magnetizable(
            [center, center + self.separation, center], model)
        self.add_solver()

        # the pre-loop update runs with dip_fld == 0, so the first iterate is
        # the free-particle response
        free_moment = reference(field, self.chi0, self.m_sat)
        moment = free_moment
        for step in range(10):
            self.system.integrator.run(1)
            local = abs(field - self.prefactor * moment / self.separation**3)
            moment = reference(local, self.chi0, self.m_sat)
            for p in (p1, p2):
                self.assertAlmostEqual(
                    p.dipm, moment, delta=1e-12,
                    msg=f"{model} pair mismatch at step {step + 1}")

        # the converged moment is reduced by the neighbour's opposing field
        self.assertLess(p1.dipm, free_moment)

    def check_validation(self, model):
        center = 0.5 * self.system.box_l[0]
        _, p = self.add_magnetizable([center] * 3, model)
        for invalid in (0., -1.):
            with self.assertRaisesRegex(ValueError, "dipm_sat"):
                p.dipm_sat = invalid
        with self.assertRaisesRegex(ValueError, "mag_susc_0"):
            p.mag_susc_0 = -1e-8
        # valid values are still accepted
        p.dipm_sat = 2.
        p.mag_susc_0 = 0.
        self.assertEqual(p.dipm_sat, 2.)
        self.assertEqual(p.mag_susc_0, 0.)


@utx.skipIfMissingFeatures(["LANGEVIN_MAGNETIZATION", "EXTERNAL_FORCES"])
class LangevinMagnetization(MagnetizationCommon, ut.TestCase):
    model = "langevin_magnetization"

    def test_response_curve(self):
        self.check_curve(self.model, langevin_M)

    def test_zero_field(self):
        self.check_zero_field(self.model)

    def test_saturation(self):
        self.check_saturation(self.model)

    def test_mutual_pair(self):
        self.check_mutual_pair(self.model, langevin_M)

    def test_validation(self):
        self.check_validation(self.model)


@utx.skipIfMissingFeatures(["FROELICH_KENNELLY", "EXTERNAL_FORCES"])
class FroelichKennelly(MagnetizationCommon, ut.TestCase):
    model = "froelich_kennelly"

    def test_response_curve(self):
        self.check_curve(self.model, froelich_M)

    def test_zero_field(self):
        self.check_zero_field(self.model)

    def test_saturation(self):
        self.check_saturation(self.model)

    def test_mutual_pair(self):
        self.check_mutual_pair(self.model, froelich_M)

    def test_validation(self):
        self.check_validation(self.model)


@utx.skipIfMissingFeatures(["LANGEVIN_MAGNETIZATION", "FROELICH_KENNELLY",
                            "EXTERNAL_FORCES"])
class ModelExclusivity(MagnetizationCommon, ut.TestCase):

    def test_one_model_per_particle(self):
        """Only one magnetization model may be active on a given particle."""
        center = 0.5 * self.system.box_l[0]
        self.apply_field(1.)
        _, p = self.add_magnetizable([center] * 3, "langevin_magnetization")
        p.froelich_kennelly_is_enabled = True
        self.add_solver()
        with self.assertRaisesRegex(Exception, "only enable one"):
            self.system.integrator.run(0, recalc_forces=True)


if __name__ == "__main__":
    ut.main()
