.. _Magnetodynamics:

Magnetodynamics
===============

|es| contains methods to simulate the internal magnetisation dynamics
of magnetic particles and the interactions between point dipoles.
The default behaviour in |es| is that, once a dipole moment is assigned
to a particle, it will strictly follow the rotation of the particle’s quaternion.
This is called the **fixed point dipole** model.
This model is predominantly used to simulate magnetic soft matter.

However, this approximation does not always represent realistic behaviour
of a single-domain magnetic nanoparticle — it only holds in the limit of
infinitely high magnetic anisotropy energy.
In reality, several **internal relaxation mechanisms** exist,
so the dipole moment is not necessarily co-aligned with the particle quaternion,
nor does it perfectly follow its motion.

To incorporate this phenomenology in simulations, |es| offers several models, listed below.

.. _Magnetodynamics_common:

Common requirements
-------------------

All magnetodynamics models share the same setup constraints.

* The moment must live on a **virtual site** attached to a real particle via
  :meth:`~espressomd.particle_data.ParticleHandle.vs_auto_relate_to`, with
  propagation mode ``Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT``.
  The update is skipped silently for non-virtual particles.
* A particle may enable **at most one** model at a time. Enabling two raises a
  runtime error when the integrator starts.
* The effective field driving the response is
  :math:`\vec{H} = \vec{H}_\mathrm{ext} + \vec{H}_\mathrm{dip}`, where
  :math:`\vec{H}_\mathrm{ext}` is the sum of all
  :class:`~espressomd.constraints.HomogeneousMagneticField` constraints and
  :math:`\vec{H}_\mathrm{dip}` is the particle's
  :attr:`~espressomd.particle_data.ParticleHandle.dip_fld`.

.. note::

    ``dip_fld`` is only computed by :class:`~espressomd.magnetostatics.DipolarDirectSum`
    (CPU and GPU), the only solvers that support ``DIPOLE_FIELD_TRACKING``. With
    :class:`~espressomd.magnetostatics.DipolarP3M`, :class:`~espressomd.magnetostatics.DLC`
    or ScaFaCoS, ``dip_fld`` stays zero, so the models respond to the **external field
    alone** and all interparticle magnetic coupling is ignored. This is not reported at
    runtime, so pick the direct sum whenever the mutual magnetization matters.

.. note::

    ``dip_fld`` is recomputed during the force calculation, i.e. *after* the
    magnetization update, and is zero on the initial force evaluation. The mutual
    magnetization of interacting particles is therefore solved **explicitly**: it
    relaxes to its self-consistent value over several time steps rather than within
    one. Keep the time step small when particles significantly magnetize each other.

.. _Magnetodynamics_stability:

Stability of the explicit mutual-magnetization scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The explicit fixed-point iteration described above,
:math:`\vec{m} \leftarrow m(\vec{H}_\mathrm{ext} + \vec{H}_\mathrm{dip}(\vec{m}))`,
converges only if it is a contraction. Linearising one Picard step gives an
amplification factor

.. math::

   |f'| = \chi_\mathrm{diff} \, c \, \frac{\text{prefactor}}{r^3}

where :math:`\chi_\mathrm{diff} = \mathrm{d}m/\mathrm{d}H` is the local differential
susceptibility, :math:`r` is the center-to-center distance between two coupled
particles, and :math:`c` is a geometric factor that depends on the relative
arrangement of the two dipoles: :math:`c = 1` for particles side-by-side and
:math:`c = 2` for particles head-to-tail. For two touching spheres this works out to
roughly :math:`|f'| \approx 0.52 \, c \, \chi_\mathrm{app}`, and summed over a
head-to-tail chain of touching spheres it grows to roughly
:math:`|f'| \approx 4.8 \, \chi_\mathrm{app} \, \text{prefactor} / r^3`. The iteration
is only guaranteed to converge to the physical fixed point while :math:`|f'| < 1`;
dense or touching assemblies with an apparent susceptibility of order 1 or larger are
routinely well past this limit.

.. warning::

    Because :math:`m` saturates, :math:`\chi_\mathrm{diff}` decreases with field, so
    the map can have **two stable fixed points** at strong coupling: the correct,
    weakly-magnetized branch, and a spurious, saturated branch. An over-coupled
    chain does **not** visibly diverge — it walks onto the saturated branch over a
    few time steps and then sits there, looking perfectly converged. A residual (or
    force/torque) check cannot detect this, because both branches are genuine fixed
    points with zero residual. The diagnostics that *can* detect it are:

    * the **contraction ratio** :math:`\lVert \delta\vec{m}_{n+1} \rVert /
      \lVert \delta\vec{m}_n \rVert` between successive iterates, which is the
      empirical :math:`|f'|`: if it persists at or above 1, the one-iterate-per-step
      scheme is not resolving the physics, regardless of how converged the moments
      look;
    * a **field up/down hysteresis sweep**: ramp the external field up, then back
      down, through the same values. If the two branches disagree at equal field,
      the "solution" the simulation found is being picked by numerical accident
      (e.g. initial conditions, step ordering) rather than by physical history.

    See :ref:`Magnetodynamics_probe` for how to measure the contraction ratio.

.. _Magnetodynamics_probe:

Probing convergence with ``run(0, recalc_forces=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~espressomd.integrate.IntegratorHandle.run` with ``steps=0`` and
``recalc_forces=True`` performs **exactly one Picard iterate at frozen positions**:
it runs the pre-integration-loop block (virtual site update, magnetodynamics update,
force calculation, which refills ``dip_fld``) and then skips the main integration
loop entirely, since zero steps were requested. Calling it repeatedly therefore
drives the fixed-point iteration forward step by step without advancing the
simulation, which is exactly what is needed to measure the contraction ratio from
:ref:`Magnetodynamics_stability` or to check for the two-branch hazard::

    dipm_prev = None
    for i in range(50):
        system.integrator.run(0, recalc_forces=True)
        dipm = np.array([p.dipm for p in system.part])
        if dipm_prev is not None:
            delta = dipm - dipm_prev
            print(i, np.linalg.norm(delta))
        dipm_prev = dipm

A contraction ratio (the ratio of successive ``np.linalg.norm(delta)`` values) that
stays below 1 and shrinks towards zero indicates convergence to the physical fixed
point. A ratio that plateaus at or above 1, or moments that keep growing towards
saturation, indicate the over-coupled regime described above.

.. _Langevin_magnetization:

Langevin magnetization
----------------------

.. note::

    Requires feature ``LANGEVIN_MAGNETIZATION``.

A superparamagnetic particle whose internal moment relaxes much faster than the
MD time step carries no memory: its moment is an instantaneous, purely algebraic
function of the local field. The **Langevin magnetization** model assigns

.. math::

   \vec{m} = m_\mathrm{sat} \, L(\alpha) \, \hat{H},
   \qquad
   \alpha = \frac{3 \chi_0 |\vec{H}|}{m_\mathrm{sat}},
   \qquad
   L(x) = \coth(x) - \frac{1}{x}

where :math:`m_\mathrm{sat}` is the saturation moment
(:attr:`~espressomd.particle_data.ParticleHandle.dipm_sat`) and :math:`\chi_0` the
initial susceptibility (:attr:`~espressomd.particle_data.ParticleHandle.mag_susc_0`).
The prefactor :math:`3\chi_0/M_\mathrm{sat}` is chosen so that the response is linear
at weak fields, :math:`|\vec{m}| \to \chi_0 |\vec{H}|`, and saturates at
:math:`M_\mathrm{sat}` at strong fields. Below :math:`\alpha = 10^{-2}` the series
:math:`L(\alpha) \approx \alpha/3 - \alpha^3/45` is used instead of the closed form.

The model is deterministic and thermostat-free: unlike
:ref:`Thermal_Stoner_Wohlfarth` it does not sample thermal fluctuations of the moment,
and :math:`k_B T` does not enter. It suits large particles for which the thermal
energy is negligible against the field energy.

Fields with :math:`|\vec{H}| < 10^{-5}` are treated as vanishing, and set
:math:`|\vec{m}| = 0`.

.. warning::

    ``mag_susc_0`` is the **apparent** susceptibility of the isolated particle,
    :math:`\chi_0 = \chi_\mathrm{app} = m/H`, evaluated as if that single particle
    were placed in the field alone — it is *not* the intrinsic material
    susceptibility :math:`\chi_\mathrm{int}` of the magnetic material it is made of.
    For a sphere the two are related through the demagnetizing factor
    :math:`N = 1/3`:

    .. math::

       \chi_\mathrm{app} = \frac{\chi_\mathrm{int}}{1 + \chi_\mathrm{int}/3}

    The particle volume is also already folded into ``mag_susc_0``: the model
    predicts the moment :math:`m` directly, not a magnetization density, so
    ``mag_susc_0`` must already include the factor of volume that converts a
    material susceptibility into a per-particle one. |es| applies **no**
    demagnetizing correction and knows nothing of the particle's physical
    volume — both must be folded into ``mag_susc_0`` by the user before it is
    set. Passing the bare intrinsic/material susceptibility here is a common
    mistake that silently overestimates the coupling and the saturation field.

.. _Froelich_Kennelly:

Froelich–Kennelly magnetization
-------------------------------

.. note::

    Requires feature ``FROELICH_KENNELLY``.

The **Froelich–Kennelly** relation is an interpolation between the same two
limits, avoiding the hyperbolic functions:

.. math::

   \vec{m} = \frac{\chi_0 m_\mathrm{sat}}{m_\mathrm{sat} + \chi_0 |\vec{H}|} \vec{H}

It uses the same two particle properties, with the same meaning, and shares the
linear-response and saturation limits of the Langevin model while differing in
between.

.. warning::

    As with the Langevin model above, ``mag_susc_0`` is the **apparent**
    susceptibility of the isolated particle (demagnetizing factor and particle
    volume already folded in by the user), not the intrinsic material
    susceptibility. |es| applies no demagnetizing correction. See the warning in
    :ref:`Langevin_magnetization` for the full explanation and the sphere formula.

.. _Magnetization_models_example:

Example
-------

Both models are configured entirely through particle properties::

    import espressomd
    import espressomd.magnetostatics
    import espressomd.constraints
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation

    system = espressomd.System(box_l=[10.0, 10.0, 10.0])
    system.time_step = 0.01
    system.cell_system.skin = 0.4
    system.min_global_cut = 2.

    # the driving external field
    system.constraints.add(
        espressomd.constraints.HomogeneousMagneticField(H=[0., 0., 1.]))

    # a magnetizable unit: real anchor + virtual site carrying the moment
    anchor = system.part.add(pos=[5., 5., 5.])
    p = system.part.add(pos=anchor.pos, rotation=[True, True, True])
    p.vs_auto_relate_to(anchor)
    p.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

    p.dipm_sat = 1.0    # saturation moment M_sat, must be > 0
    p.mag_susc_0 = 0.33  # initial susceptibility chi_0, must be >= 0
    p.langevin_magnetization_is_enabled = True
    # or, for the other model:
    # p.froelich_kennelly_is_enabled = True

    # only the direct sum feeds dip_fld back into the model
    system.magnetostatics.solver = \
        espressomd.magnetostatics.DipolarDirectSum(prefactor=1.)

    system.integrator.run(100)

.. _Thermal_Stoner_Wohlfarth:

Thermal Stoner–Wohlfarth
------------------------

.. note::

    Requires feature ``THERMAL_STONER_WOHLFARTH`` and
    optionally feature ``DIPOLE_FIELD_TRACKING``, as well as
    external feature ``NLOPT``,
    enabled with ``-D ESPRESSO_BUILD_WITH_NLOPT=ON``.

The **thermal Stoner–Wohlfarth (SW)** model includes Néel relaxation in simulations
of single-domain magnetic nanoparticles.
This is an implementation of the algorithm originally presented in :cite:`mostarac25a`.
If you use this method in your work, please cite the original publication, in addition to |es|.

In interacting systems, the method relies on the ``DIPOLE_FIELD_TRACKING`` feature.
Make sure you use magnetostatics actors that support this feature.

---

The magnetic energy of a Stoner–Wohlfarth particle (at :math:`T=0`) is given by

.. math::

   U = - \mu_0 \mu (\vec{e} \cdot \vec{H}) - K V (\vec{e} \cdot \vec{n})^2

The algorithm can be logically separated into three steps:

**Step 1 — Critical Field and Energy Minima**

First, the **critical field** :math:`h_\mathrm{cr}` is calculated based on
the current angle :math:`\phi` between the magnetic field vector :math:`\vec{h}`
and the particle’s anisotropy axis.

For a given :math:`\phi`, the algorithm finds the angle :math:`\theta'_\mathrm{min}`
that minimizes the total magnetic energy and is closest to the previous dipole-moment state.
This is ensured by initializing the minimizer with the previous particle state.

If the field acting on the particle is less than :math:`h_\mathrm{cr}`,
the algorithm proceeds to find the angles :math:`\theta'_\mathrm{max}` and
:math:`\theta''_\mathrm{max}` that maximize the total magnetic energy on
both sides of :math:`\theta'_\mathrm{min}`.

**Step 2 — Energy Barrier and Néel Relaxation**

Using these extrema, the algorithm calculates the **energy barriers**
on both sides of :math:`\theta'_\mathrm{min}`:

.. math::

   \begin{aligned}
   \Delta E'  &= \frac{1}{K V} \left| U(\phi, \theta'_\mathrm{max})  - U(\phi, \theta'_\mathrm{min}) \right|, \\
   \Delta E'' &= \frac{1}{K V} \left| U(\phi, \theta''_\mathrm{max}) - U(\phi, \theta'_\mathrm{min}) \right|.
   \end{aligned}

The smaller of the two barriers is chosen:

.. math::

   \Delta E = \min(\Delta E', \Delta E'').

This barrier is used to estimate a **characteristic timescale** for the Néel relaxation process:

.. math::

   \tau_N = \frac{\tau_D}{2\sigma} \sqrt{\frac{\pi}{\sigma}} e^{\Delta E \sigma},

and the **transition probability** (neglecting back-switching) is

.. math::

   p = 1 - e^{-\delta t / \tau_N},

where :math:`\delta t` is the integration time step.

**Step 3 — Trial Move and Dipole Update**

A **trial move** is made by drawing a random number and comparing it with
the transition probability, similar to a Metropolis Monte Carlo step.

- If the trial move is **successful**, the algorithm finds a new minimum
  :math:`\theta''_\mathrm{min}` and aligns the dipole moment accordingly.
- Otherwise, the dipole moment remains aligned with :math:`\theta'_\mathrm{min}`.

For more details, particularly if you are unsure about the physical quantities involved,
please refer to the original publication.

The example below shows how to set up and parametrise a particle to be used by
the **thermal Stoner-Wohlfarth** solver. Note that ``is_enabled`` needs to be set
to ``True`` explicitly on the virtual site.
Moreover, the virtual sites that are tagged for magnetodynamics must be set
to use the ``Propagation.ROT_VS_INDEPENDENT`` propagation mode.

.. code-block:: python

    import espressomd
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation

    system = espressomd.System(box_l=[10.0, 10.0, 10.0])
    system.time_step = 0.001  # MD time step in simulation units
    system.thermostat.set_langevin(kT=1., gamma=75., gamma_rotation=25., seed=42)

    # One particle with thermal Stoner-Wohlfarth enabled
    p1 = system.part.add(pos=[1, 1, 1])
    p1.director = (1, 0, 0) # easy axis direction
    p1.rotation = (True, True, True)  # allow particle rotation

    p2 = system.part.add(pos=p1.pos)
    p2.dip = (1.75,0,0) # set dipole moment for the virtual particle in reduced units
    p2.rotation = (False, False, False) # disable rotations of the virtual site
    p2.magnetodynamics = {
      'is_enabled': True,
      'anisotropy_field_inv': 0.175, # inverse anisotropy field (1/H_k) in reduced units
      'sat_mag': 1.75, # saturation magnetisation in reduced units
      'anisotropy_energy': 5., # anisotropy energy K * V in reduced units
      'sw_dt_incr': 1.0e-10, # kinetic Monte Carlo time increment [s]
      'sw_tau0_inv': 1.0e9  # inverse attempt time (1/tau_0) [1/s]
      }
    # make virtual and set the proper propagation mode for magnetodynamics
    p2.vs_auto_relate_to(p1)
    p2.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

