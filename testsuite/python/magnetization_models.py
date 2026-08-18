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


@utx.skipIfMissingFeatures(["LANGEVIN_MAGNETIZATION", "EXTERNAL_FORCES"])
class ZeroMomentReceiver(MagnetizationCommon, ut.TestCase):

    def test_zero_moment_still_receives_field(self):
        """A moment driven to zero must still be able to re-magnetize.

        ``DipolarDirectSum`` skips ``dipm == 0`` particles when gathering, so
        a particle whose moment collapsed used to drop out of the sum
        entirely: it could neither contribute to nor *receive* ``dip_fld``,
        and stayed demagnetized forever.
        """
        center = 0.5 * self.system.box_l[0]
        # zero susceptibility, so the model drives the moment to zero whatever
        # the field
        _, weak = self.add_magnetizable([center] * 3, "langevin_magnetization")
        weak.mag_susc_0 = 0.
        # a strongly magnetized neighbour to supply a field
        self.system.part.add(pos=[center, center + self.separation, center],
                             rotation=[True] * 3, dip=[0., 0., 5.])
        self.apply_field(1.)
        self.add_solver()
        self.system.integrator.run(1)

        self.assertEqual(weak.dipm, 0.)
        # carrying no moment, it must nonetheless still see its neighbour
        self.assertGreater(np.linalg.norm(np.copy(weak.dip_fld)), 0.,
                           msg="zero-moment particle received no dipole field")


class DP3MCommon(MagnetizationCommon):
    """Periodic, regularly decomposed setup for the DP3M-based checks."""

    def setUp(self):
        system = self.system
        system.time_step = 0.01
        system.cell_system.skin = 0.4
        system.min_global_cut = 1.5
        system.periodicity = [True] * 3
        system.cell_system.set_regular_decomposition()

    def tearDown(self):
        super().tearDown()

    def add_lattice(self, n_side, chi0=None, jitter=0.15):
        """Jittered lattice of magnetizable units spanning the whole box.

        Spanning the box means particles land on both sides of every domain
        boundary, which is what puts moments into ghost cells. The spacing
        must stay below the solver's real-space cutoff, or there are no
        short-range pairs at all and the ghost data is never read.
        """
        system = self.system
        rng = np.random.default_rng(42)
        spacing = system.box_l[0] / n_side
        sites = []
        for idx in np.ndindex(n_side, n_side, n_side):
            pos = (np.array(idx) + 0.5) * spacing
            pos += rng.uniform(-jitter, jitter, 3) * spacing
            _, p = self.add_magnetizable(pos, "langevin_magnetization")
            if chi0 is not None:
                p.mag_susc_0 = chi0
            sites.append(p)
        return sites

    def min_pair_distance(self, sites):
        """Smallest minimum-image separation, to assert against the cutoff."""
        pos = np.array([np.copy(p.pos) for p in sites])
        box = np.array(self.system.box_l)
        best = np.inf
        for i in range(len(pos)):
            d = pos[i] - pos[i + 1:]
            d -= box * np.round(d / box)
            if d.size:
                best = min(best, float(np.min(np.linalg.norm(d, axis=1))))
        return best


@utx.skipIfMissingFeatures(["LANGEVIN_MAGNETIZATION", "DP3M",
                            "EXTERNAL_FORCES"])
class SolverCapabilityGuard(DP3MCommon, ut.TestCase):

    def add_dp3m(self):
        # fixed parameters rather than tuned: these checks only need DP3M to
        # be *active*, and tuning a small sparse system is fragile
        self.system.magnetostatics.solver = \
            espressomd.magnetostatics.DipolarP3M(
                prefactor=1., accuracy=1e-3, r_cut=4.5, mesh=[8, 8, 8],
                cao=3, alpha=0.5, tune=False)

    def test_guard_rejects_solver_without_dipole_field(self):
        """Only the direct sum populates ``dip_fld``.

        With any other solver the models would respond to the external field
        alone and all mutual magnetization would be silently dropped, so the
        integrator must refuse to start.
        """
        self.apply_field(1.)
        self.add_lattice(2)
        with self.assertRaisesRegex(Exception, "does not provide the dipole"):
            self.add_dp3m()
            self.system.integrator.run(0, recalc_forces=True)

    def test_direct_sum_is_accepted(self):
        self.apply_field(1.)
        self.add_lattice(2)
        self.add_solver()
        self.system.integrator.run(0, recalc_forces=True)


@utx.skipIfMissingFeatures(["LANGEVIN_MAGNETIZATION", "DP3M",
                            "EXTERNAL_FORCES"])
class GhostMomentConsistency(DP3MCommon, ut.TestCase):
    """The dipole moment must reach ghosts every step, not only on resorts.

    ``dipm`` lives in ``ParticleProperties``, which the ghost protocol treats
    as resort-only. Once a magnetization model rewrites it every step, any
    solver reading it through a ghost -- the DP3M real-space kernel -- sees a
    stale magnitude and computes wrong forces.

    Only meaningful on more than one MPI rank: for a single rank
    ``RegularDecomposition::prepare_comm`` returns an empty communicator and
    wires periodic neighbours directly to the real cells, so no ghost exchange
    happens at all.
    """

    def forces(self):
        return np.array([np.copy(p.f)
                         for p in sorted(self.system.part.all(),
                                         key=lambda p: p.id)])

    def test_ghost_moments_match_frozen_reference(self):
        system = self.system
        sites = self.add_lattice(4, chi0=0.4)
        # DP3M never fills dip_fld; here it only has to *consume* dipm, and it
        # is the real-space kernel that reads ghosts. The parameters are fixed
        # rather than tuned because tuning picks whatever cutoff is fastest --
        # on a sparse system that can be smaller than the nearest-neighbour
        # distance, leaving no short-range pairs and no ghost reads at all,
        # which would make this test silently vacuous.
        r_cut = 4.5
        self.assertLess(self.min_pair_distance(sites), r_cut,
                        msg="no pair is within the real-space cutoff, so the "
                            "test would not exercise ghost moments")
        system.magnetostatics.solver = espressomd.magnetostatics.DipolarP3M(
            prefactor=1., accuracy=1e-3, r_cut=r_cut, mesh=[8, 8, 8], cao=3,
            alpha=0.5, tune=False)

        # A solver that cannot supply dip_fld is rejected outright, and DP3M
        # cannot supply it yet. There is no way around that here: DipolarDirect
        # Sum is the only solver allowed with a model, and it never reads a
        # ghost at all (it gathers local particles over MPI), so running this
        # check against it would pass even with the bug fully reintroduced.
        # This reactivates on its own once DP3M populates dip_fld.
        try:
            system.integrator.run(0, recalc_forces=True)
        except Exception as err:
            if "does not provide the dipole" not in str(err):
                raise
            self.skipTest(
                "needs a solver that both reads ghost moments and supplies "
                "dip_fld; DP3M does not populate dip_fld yet")

        # Phase A: ramp the external field so dipm changes on every step. The
        # anchors are fixed, so nothing ever moves and no resort is triggered
        # -- exactly the regime where a resort-only exchange leaves ghosts
        # holding a stale moment.
        for i in range(1, 11):
            system.constraints.clear()
            system.constraints.add(
                espressomd.constraints.HomogeneousMagneticField(
                    H=[0., 0., 0.4 * i]))
            system.integrator.run(1)
        driven_forces = self.forces()
        driven_dipm = [p.dipm for p in sites]
        self.assertGreater(max(driven_dipm), 0.)

        # Phase B: freeze those exact moments and force a resort, so ghosts
        # are correct by construction whatever route the data took. Same
        # solver, same positions, same field, so the forces must agree.
        for p in sites:
            p.langevin_magnetization_is_enabled = False
        for p, dipm in zip(sites, driven_dipm):
            p.dipm = dipm
        system.cell_system.resort()
        system.integrator.run(0, recalc_forces=True)
        frozen_forces = self.forces()

        np.testing.assert_allclose(
            driven_forces, frozen_forces, atol=1e-9, rtol=1e-6,
            err_msg="model-driven forces disagree with the same moments "
                    "applied statically: ghost dipole moments are stale")


if __name__ == "__main__":
    ut.main()
