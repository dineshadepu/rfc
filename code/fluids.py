"""
Run it by

python lid_driven_cavity.py --openmp --scheme etvf --integrator pec --internal-flow --pst sun2019 --re 100 --tf 25 --nx 50 --no-edac -d lid_driven_cavity_scheme_etvf_integrator_pec_pst_sun2019_re_100_nx_50_no_edac_output --detailed-output --pfreq 100

"""
import numpy
import numpy as np

from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import (CubicSpline, WendlandQuintic, QuinticSpline,
                                WendlandQuinticC4, Gaussian, SuperGaussian)
from pysph.sph.integrator import Integrator


from pysph.sph.wc.transport_velocity import (MomentumEquationArtificialViscosity)

from solid_mech import (ComputeAuHatETVF, ComputeAuHatETVFSun2019,
                        SavePositionsIPSTBeforeMoving, AdjustPositionIPST,
                        CheckUniformityIPST, ResetParticlePositionsIPST,
                        EDACEquation, setup_ipst, ComputeAuhatETVFIPSTFluids,
                        SetHIJForInsideParticles)

from pysph.sph.wc.edac import (SolidWallPressureBC)

from pysph.sph.integrator import PECIntegrator
from boundary_particles import (add_boundary_identification_properties)

from boundary_particles import (ComputeNormals, SmoothNormals,
                                IdentifyBoundaryParticleCosAngleEDAC)

from pysph.examples.solid_mech.impact import add_properties
from pysph.sph.wc.linalg import mat_vec_mult
from solid_mech import (ContinuityEquationETVFCorrection)
from pysph.sph.wc.edac import ClampWallPressure

from rigid_body_3d import (add_properties_stride,
                           set_total_mass, set_center_of_mass,
                           set_body_frame_position_vectors,
                           set_body_frame_normal_vectors,
                           set_moment_of_inertia_and_its_inverse,
                           ResetForce, SumUpExternalForces,
                           normalize_R_orientation)

from rigid_rigid_interaction_equations import (
    ComputeContactForceNormalsMohseni,
    ComputeContactForceDistanceAndClosestPointMohseni,
    ComputeContactForceMohseni,
    TransferContactForceMohseni)
# compute the boundary particles
from boundary_particles import (get_boundary_identification_etvf_equations,
                                add_boundary_identification_properties)


class SetWallVelocityFreeSlipAndNoSlip(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf, s_u, s_v, s_w, d_wij, WIJ):
        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ugfs, d_vgfs, d_wgfs,
                  d_ugns, d_vgns, d_wgns, d_u, d_v, d_w, d_normal):
        idx3 = declare('int', 1)
        idx3 = 3 * d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        tmp1 = d_u[d_idx] - d_uf[d_idx]
        tmp2 = d_v[d_idx] - d_vf[d_idx]
        tmp3 = d_w[d_idx] - d_wf[d_idx]

        projection = (tmp1 * d_normal[idx3] + tmp2 * d_normal[idx3 + 1] +
                      tmp3 * d_normal[idx3 + 2])

        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ugfs[d_idx] = 2 * projection * d_normal[idx3] + d_uf[d_idx]
        d_vgfs[d_idx] = 2 * projection * d_normal[idx3 + 1] + d_vf[d_idx]
        d_wgfs[d_idx] = 2 * projection * d_normal[idx3 + 2] + d_wf[d_idx]

        # For No slip boundary conditions
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ugns[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vgns[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wgns[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]


class SetWallVelocityNoSlip(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf, s_u, s_v, s_w, d_wij, WIJ):
        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ugns, d_vgns, d_wgns,
                  d_u, d_v, d_w):
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ugns[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vgns[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wgns[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]


class SetWallVelocityUhatNoSlip(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf, s_uhat, s_vhat, s_what,
             d_wij, WIJ):

        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_uhat[s_idx] * WIJ
        d_vf[d_idx] += s_vhat[s_idx] * WIJ
        d_wf[d_idx] += s_what[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ughatns, d_vghatns,
                  d_wghatns, d_u, d_v, d_w):

        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ughatns[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vghatns[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wghatns[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]


class SetWallVelocityUhatFreeSlipAndNoSlip(Equation):
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf, s_uhat, s_vhat, s_what,
             d_wij, WIJ):

        # normalisation factor is different from 'V' as the particles
        # near the boundary do not have full kernel support
        d_wij[d_idx] += WIJ

        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_uhat[s_idx] * WIJ
        d_vf[d_idx] += s_vhat[s_idx] * WIJ
        d_wf[d_idx] += s_what[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx, d_ughatfs, d_vghatfs,
                  d_wghatfs, d_ughatns, d_vghatns, d_wghatns, d_u, d_v, d_w,
                  d_normal):
        idx3 = declare('int', 1)
        idx3 = 3 * d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        tmp1 = d_u[d_idx] - d_uf[d_idx]
        tmp2 = d_v[d_idx] - d_vf[d_idx]
        tmp3 = d_w[d_idx] - d_wf[d_idx]

        projection = (tmp1 * d_normal[idx3] + tmp2 * d_normal[idx3 + 1] +
                      tmp3 * d_normal[idx3 + 2])

        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ughatfs[d_idx] = 2 * projection * d_normal[idx3] + d_uf[d_idx]
        d_vghatfs[d_idx] = 2 * projection * d_normal[idx3 + 1] + d_vf[d_idx]
        d_wghatfs[d_idx] = 2 * projection * d_normal[idx3 + 2] + d_wf[d_idx]

        # For No slip boundary conditions
        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ughatns[d_idx] = 2 * d_u[d_idx] - d_uf[d_idx]
        d_vghatns[d_idx] = 2 * d_v[d_idx] - d_vf[d_idx]
        d_wghatns[d_idx] = 2 * d_w[d_idx] - d_wf[d_idx]


class ContinuitySolidEquationGTVF(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_uhat, d_vhat, d_what,
             s_ughatfs, s_vghatfs, s_wghatfs, s_ughatns, s_vghatns, s_wghatns,
             d_arho, DWIJ):
        uhatij = d_uhat[d_idx] - s_ughatfs[s_idx]
        vhatij = d_vhat[d_idx] - s_vghatfs[s_idx]
        whatij = d_what[d_idx] - s_wghatfs[s_idx]

        # uhatij = d_uhat[d_idx] - s_ughatns[s_idx]
        # vhatij = d_vhat[d_idx] - s_vghatns[s_idx]
        # whatij = d_what[d_idx] - s_wghatns[s_idx]

        udotdij = DWIJ[0] * uhatij + DWIJ[1] * vhatij + DWIJ[2] * whatij
        fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_arho[d_idx] += fac * udotdij


class ContinuitySolidEquation(Equation):
    def initialize(self, d_arho, d_idx):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_u, d_v, d_w, s_ug, s_vg,
             s_wg, d_arho, DWIJ):
        uhatij = d_u[d_idx] - s_ug[s_idx]
        vhatij = d_v[d_idx] - s_vg[s_idx]
        whatij = d_w[d_idx] - s_wg[s_idx]

        udotdij = DWIJ[0] * uhatij + DWIJ[1] * vhatij + DWIJ[2] * whatij
        fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
        d_arho[d_idx] += fac * udotdij


# class ContinuitySolidEquation(Equation):
#     def initialize(self, d_arho, d_idx):
#         d_arho[d_idx] = 0.0

#     def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_u, d_v, d_w,
#              s_ug, s_vg, s_wg, d_arho, DWIJ):
#         uhatij = d_u[d_idx] - s_ug[s_idx]
#         vhatij = d_v[d_idx] - s_vg[s_idx]
#         whatij = d_w[d_idx] - s_wg[s_idx]

#         udotdij = DWIJ[0]*uhatij + DWIJ[1]*vhatij + DWIJ[2]*whatij
#         fac = d_rho[d_idx] * s_m[s_idx] / s_rho[s_idx]
#         d_arho[d_idx] += fac * udotdij


class ContinuitySolidEquationETVFCorrection(Equation):
    """
    This is the additional term arriving in the new ETVF continuity equation
    """
    def loop(self, d_idx, d_arho, d_rho, d_u, d_v, d_w, d_uhat, d_vhat, d_what,
             s_idx, s_rho, s_m, s_ugfs, s_vgfs, s_wgfs, s_ughatfs, s_vghatfs,
             s_wghatfs, s_ugns, s_vgns, s_wgns, s_ughatns, s_vghatns,
             s_wghatns, DWIJ):
        # by using free ship boundary conditions
        tmp0 = s_rho[s_idx] * (s_ughatfs[s_idx] - s_ugfs[s_idx]
                               ) - d_rho[d_idx] * (d_uhat[d_idx] - d_u[d_idx])

        tmp1 = s_rho[s_idx] * (s_vghatfs[s_idx] - s_vgfs[s_idx]
                               ) - d_rho[d_idx] * (d_vhat[d_idx] - d_v[d_idx])

        tmp2 = s_rho[s_idx] * (s_wghatfs[s_idx] - s_wgfs[s_idx]
                               ) - d_rho[d_idx] * (d_what[d_idx] - d_w[d_idx])

        # # by using no ship boundary conditions
        # tmp0 = s_rho[s_idx] * (s_ughatns[s_idx] - s_ugns[s_idx]
        #                        ) - d_rho[d_idx] * (d_uhat[d_idx] - d_u[d_idx])

        # tmp1 = s_rho[s_idx] * (s_vghatns[s_idx] - s_vgns[s_idx]
        #                        ) - d_rho[d_idx] * (d_vhat[d_idx] - d_v[d_idx])

        # tmp2 = s_rho[s_idx] * (s_wghatns[s_idx] - s_wgns[s_idx]
        #                        ) - d_rho[d_idx] * (d_what[d_idx] - d_w[d_idx])

        vijdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_arho[d_idx] += s_m[s_idx] / s_rho[s_idx] * vijdotdwij


class FluidEDACEquationNoCorrections(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACEquationNoCorrections, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_c0_ref, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p, s_m, s_rho, d_ap, DWIJ, XIJ, s_uhat, s_vhat,
             s_what, s_u, s_v, s_w, R2IJ, VIJ, EPS):
        vhatij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_uhat[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vhat[s_idx]
        vhatij[2] = d_what[d_idx] - s_what[s_idx]

        cs2 = d_c0_ref[0] * d_c0_ref[0]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        # pi = d_p[d_idx]
        # rhoj = s_rho[s_idx]
        # pj = s_p[s_idx]

        # vij_dot_dwij = -(VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] +
        #                  VIJ[2] * DWIJ[2])

        vhatij_dot_dwij = -(vhatij[0] * DWIJ[0] + vhatij[1] * DWIJ[1] +
                            vhatij[2] * DWIJ[2])

        # vhatij_dot_dwij = (VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] +
        #                    VIJ[2]*DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 22 of the current paper #
        #######################################################
        d_ap[d_idx] += -rhoi * cs2 * Vj * vhatij_dot_dwij

        #######################################################
        # fourth term on the rhs of Eq 22 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class EDACSolidEquation(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(EDACSolidEquation, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_c0_ref, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p, s_m, s_rho, d_ap, DWIJ, XIJ, s_ughatfs, s_vghatfs,
             s_wghatfs, s_ugfs, s_vgfs, s_wgfs, R2IJ, EPS):
        vhatij = declare('matrix(3)')
        vij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_ughatfs[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vghatfs[s_idx]
        vhatij[2] = d_what[d_idx] - s_wghatfs[s_idx]

        vij[0] = d_u[d_idx] - s_ugfs[s_idx]
        vij[1] = d_v[d_idx] - s_vgfs[s_idx]
        vij[2] = d_w[d_idx] - s_wgfs[s_idx]

        cs2 = d_c0_ref[0] * d_c0_ref[0]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        pi = d_p[d_idx]
        # rhoj = s_rho[s_idx]
        pj = s_p[s_idx]

        vij_dot_dwij = -(vij[0] * DWIJ[0] + vij[1] * DWIJ[1] +
                         vij[2] * DWIJ[2])

        vhatij_dot_dwij = -(vhatij[0] * DWIJ[0] + vhatij[1] * DWIJ[1] +
                            vhatij[2] * DWIJ[2])

        # vhatij_dot_dwij = (VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] +
        #                    VIJ[2]*DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += (pi - rhoi * cs2) * Vj * vij_dot_dwij

        #######################################################
        # second term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += -pi * Vj * vhatij_dot_dwij

        ########################################################
        # third term on the rhs of Eq 19 of the current paper #
        ########################################################
        tmp0 = pj * (s_ughatfs[s_idx] - s_ugfs[s_idx]) - pi * (d_uhat[d_idx] -
                                                               d_u[d_idx])

        tmp1 = pj * (s_vghatfs[s_idx] - s_vgfs[s_idx]) - pi * (d_vhat[d_idx] -
                                                               d_v[d_idx])

        tmp2 = pj * (s_wghatfs[s_idx] - s_wgfs[s_idx]) - pi * (d_what[d_idx] -
                                                               d_w[d_idx])

        tmpdotdwij = (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)
        d_ap[d_idx] += Vj * tmpdotdwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class FluidEDACSolidEquationNoCorrections(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(FluidEDACSolidEquationNoCorrections, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_c0_ref, d_u, d_v, d_w, d_uhat,
             d_vhat, d_what, s_p, s_m, s_rho, d_ap, DWIJ, XIJ, s_ughatfs,
             s_vghatfs, s_wghatfs, s_ugfs, s_vgfs, s_wgfs, R2IJ, EPS):
        vhatij = declare('matrix(3)')
        vij = declare('matrix(3)')
        vhatij[0] = d_uhat[d_idx] - s_ughatfs[s_idx]
        vhatij[1] = d_vhat[d_idx] - s_vghatfs[s_idx]
        vhatij[2] = d_what[d_idx] - s_wghatfs[s_idx]

        vij[0] = d_u[d_idx] - s_ugfs[s_idx]
        vij[1] = d_v[d_idx] - s_vgfs[s_idx]
        vij[2] = d_w[d_idx] - s_wgfs[s_idx]

        cs2 = d_c0_ref[0] * d_c0_ref[0]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        # pi = d_p[d_idx]
        # rhoj = s_rho[s_idx]
        # pj = s_p[s_idx]

        vhatij_dot_dwij = -(vhatij[0] * DWIJ[0] + vhatij[1] * DWIJ[1] +
                            vhatij[2] * DWIJ[2])

        # vhatij_dot_dwij = (VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] +
        #                    VIJ[2]*DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += - rhoi * cs2 * Vj * vhatij_dot_dwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class MomentumEquationViscosityNoSlip(Equation):
    def __init__(self, dest, sources, nu):
        r"""
        Parameters
        ----------
        nu : float
            viscosity of the fluid.
        """

        self.nu = nu
        super(MomentumEquationViscosityNoSlip, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au, d_av, d_aw, d_u, d_v,
             d_w, s_ugns, s_vgns, s_wgns, R2IJ, EPS, DWIJ, XIJ, HIJ):
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ugns[s_idx]
        vij[1] = d_v[d_idx] - s_vgns[s_idx]
        vij[2] = d_w[d_idx] - s_wgns[s_idx]

        xdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        mb = s_m[s_idx]

        rhoa = d_rho[d_idx]
        rhob = s_rho[s_idx]

        fac = mb * 4 * self.nu * xdotdwij / ((rhoa + rhob) * (R2IJ + EPS))

        d_au[d_idx] += fac * vij[0]
        d_av[d_idx] += fac * vij[1]
        d_aw[d_idx] += fac * vij[2]


class StateEquation(Equation):
    def __init__(self, dest, sources, p0, rho0, b=1.0):
        self.b = b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 * (d_rho[d_idx] / self.rho0 - self.b)


class SummationDensityDummy(Equation):
    def initialize(self, d_idx, d_rho_dummy):
        d_rho_dummy[d_idx] = 0.0

    def loop(self, d_idx, d_rho_dummy, s_idx, s_m, WIJ):
        d_rho_dummy[d_idx] += s_m[s_idx]*WIJ


class SummationDensitySplash(Equation):
    def initialize(self, d_idx, d_rho_splash):
        d_rho_splash[d_idx] = 0.0

    def loop(self, d_idx, d_rho_splash, s_idx, s_m, WIJ):
        d_rho_splash[d_idx] += s_m[s_idx]*WIJ


class SummationDensitySplashRigidBody(Equation):
    def initialize(self, d_idx, d_rho_splash):
        d_rho_splash[d_idx] = 0.0

    def loop(self, d_idx, d_rho_splash, s_idx, s_m_fsi, WIJ):
        d_rho_splash[d_idx] += s_m_fsi[s_idx]*WIJ


def setup_ipst_fluids(pa, dim, source_pa):
    props = 'ipst_x ipst_y ipst_z ipst_dx ipst_dy ipst_dz'.split()

    for prop in props:
        pa.add_property(prop)

    if 'ipst_chi0' not in pa.constants:
        pa.add_constant('ipst_chi0', 0.)

    pa.add_property('ipst_chi')

    sources = []
    for p in source_pa:
        sources.append(p.name)

    equations = [
        Group(equations=[
            CheckUniformityIPST(dest=pa.name, sources=sources),
        ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=source_pa, equations=equations, dim=dim,
                            kernel=QuinticSpline(dim=dim))

    sph_eval.evaluate(dt=0.1)

    pa.ipst_chi0[0] = np.average(pa.ipst_chi)


class CheckUniformityIPSTFluidInternalFlow(Equation):
    """
    For this specific equation one has to update the NNPS

    """
    def __init__(self, dest, sources, tolerance=0.02, debug=False):
        self.inhomogenity = 0.0
        self.debug = debug
        self.tolerance = tolerance
        super(CheckUniformityIPSTFluidInternalFlow,
              self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_chi):
        d_ipst_chi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_ipst_chi, d_h, WIJ, XIJ, RIJ,
             dt):
        d_ipst_chi[d_idx] += d_h[d_idx] * d_h[d_idx] * WIJ

    def reduce(self, dst, t, dt):
        chi_max = serial_reduce_array(dst.ipst_chi, 'max')
        self.inhomogenity = fabs(chi_max - dst.ipst_chi0[0])

        # chi_min = serial_reduce_array(dst.ipst_chi, 'min')
        # self.inhomogenity = fabs(chi_min - dst.ipst_chi0[0])

    def converged(self):
        debug = self.debug
        inhomogenity = self.inhomogenity

        if inhomogenity > self.tolerance:
            if debug:
                print("Not converged:", inhomogenity)
            return -1.0
        else:
            if debug:
                print("Converged:", inhomogenity)
            return 1.0


class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au, d_av, d_aw,
             DWIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx] / rhoi2 + s_p[s_idx] / rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class MomentumEquationTVFDivergence(Equation):
    def loop(self, d_rho, d_idx, s_idx, d_p, s_p, d_au, d_av, d_aw, d_u, d_v,
             d_w, d_uhat, d_vhat, d_what, s_rho, s_m, s_u, s_v, s_w, s_uhat,
             s_vhat, s_what, DWIJ):
        tmp0 = (d_uhat[d_idx] - s_uhat[s_idx]) - (d_u[d_idx] - s_u[s_idx])

        tmp1 = (d_vhat[d_idx] - s_vhat[s_idx]) - (d_v[d_idx] - s_v[s_idx])

        tmp2 = (d_what[d_idx] - s_what[s_idx]) - (d_w[d_idx] - s_w[s_idx])

        tmp = s_m[s_idx] / s_rho[s_idx]

        tmp3 = tmp * (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_au[d_idx] += d_u[d_idx] * tmp3
        d_av[d_idx] += d_v[d_idx] * tmp3
        d_aw[d_idx] += d_w[d_idx] * tmp3


class MomentumEquationTVFDivergenceSolids(Equation):
    def loop(self, d_rho, d_idx, s_idx, d_p, s_p, d_au, d_av, d_aw, d_u, d_v,
             d_w, d_uhat, d_vhat, d_what, s_rho, s_m, s_ug, s_vg, s_wg,
             s_ughat, s_vghat, s_wghat, DWIJ):
        tmp0 = (d_uhat[d_idx] - s_ughat[s_idx]) - (d_u[d_idx] - s_ug[s_idx])

        tmp1 = (d_vhat[d_idx] - s_vghat[s_idx]) - (d_v[d_idx] - s_vg[s_idx])

        tmp2 = (d_what[d_idx] - s_wghat[s_idx]) - (d_w[d_idx] - s_wg[s_idx])

        tmp = s_m[s_idx] / s_rho[s_idx]

        tmp3 = tmp * (DWIJ[0] * tmp0 + DWIJ[1] * tmp1 + DWIJ[2] * tmp2)

        d_au[d_idx] += d_u[d_idx] * tmp3
        d_av[d_idx] += d_v[d_idx] * tmp3
        d_aw[d_idx] += d_w[d_idx] * tmp3


class SetDensityFromPressure(Equation):
    def __init__(self, dest, sources, rho0, p0, b=1.):
        self.rho0 = rho0
        self.p0 = p0
        self.b = b
        super(SetDensityFromPressure, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_rho, d_p, d_m):
        # update the density from the pressure Eq. (28)
        d_rho[d_idx] = self.rho0 * (d_p[d_idx] / self.p0 + self.b)
        # d_vol[d_idx] = d_m[d_idx] / d_rho[d_idx]


class EDACIntegrator(Integrator):
    def initial_acceleration(self, t, dt):
        pass

    def one_timestep(self, t, dt):
        self.compute_accelerations(0, update_nnps=False)
        self.stage1()
        self.do_post_stage(dt, 1)
        self.update_domain()

        self.compute_accelerations(1)

        self.stage2()
        self.do_post_stage(dt, 2)


class EDACGTVFStep(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat, d_vhat,
               d_what, d_auhat, d_avhat, d_awhat, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

    def stage2(self, d_idx, d_uhat, d_vhat, d_what, d_x, d_y, d_z, d_rho,
               d_arho, d_m, d_p, d_ap, dt):
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_p[d_idx] += dt * d_ap[d_idx]

        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        # d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        # d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        # d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]


class PECStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
                   d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_p, d_p0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]
        # d_p0[d_idx] = d_p[d_idx]

    def stage1(self, d_idx, d_m, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
               d_w0, d_u, d_v, d_w, d_rho, d_rho0, d_arho, d_au, d_av, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat, d_aw, d_p, d_p0,
               d_ap, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_what[d_idx]

        # Update volume
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]
        # d_p[d_idx] = d_p0[d_idx] + dtb2 * d_ap[d_idx]

    def stage2(self, d_idx, d_m, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
               d_w0, d_u, d_v, d_w, d_rho, d_arho, d_rho0, d_au, d_av, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat, d_aw, d_p, d_p0,
               d_ap, dt):
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt * d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_what[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
        # d_p[d_idx] = d_p0[d_idx] + dt * d_ap[d_idx]


class EDACPECStep(IntegratorStep):
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
                   d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_p, d_p0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]
        d_p0[d_idx] = d_p[d_idx]

    def stage1(self, d_idx, d_m, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
               d_w0, d_u, d_v, d_w, d_rho, d_rho0, d_arho, d_au, d_av, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat, d_aw, d_p, d_p0,
               d_ap, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_what[d_idx]

        # Update volume
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]
        d_p[d_idx] = d_p0[d_idx] + dtb2 * d_ap[d_idx]

    def stage2(self, d_idx, d_m, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_u0, d_v0,
               d_w0, d_u, d_v, d_w, d_rho, d_arho, d_rho0, d_au, d_av, d_uhat,
               d_vhat, d_what, d_auhat, d_avhat, d_awhat, d_aw, d_p, d_p0,
               d_ap, dt):
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dt * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt * d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_what[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
        d_p[d_idx] = d_p0[d_idx] + dt * d_ap[d_idx]


class SetUhatVelocitySolidsToU(Equation):
    def initialize(self, d_idx, d_uhat, d_vhat, d_what, d_u, d_v, d_w):
        d_uhat[d_idx] = d_u[d_idx]
        d_vhat[d_idx] = d_v[d_idx]
        d_what[d_idx] = d_w[d_idx]


class SetUhatVelocitySolidsToUg(Equation):
    def initialize(self, d_idx, d_uhat, d_vhat, d_what, d_ug, d_vg, d_wg):
        d_uhat[d_idx] = d_ug[d_idx]
        d_vhat[d_idx] = d_vg[d_idx]
        d_what[d_idx] = d_wg[d_idx]


class MakeAuhatZero(Equation):
    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.
        d_avhat[d_idx] = 0.
        d_awhat[d_idx] = 0.


class MomentumEquationArtificialStressSolids(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super(MomentumEquationArtificialStressSolids,
              self).__init__(dest, sources)

    def _get_helpers_(self):
        return [mat_vec_mult]

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_u, d_v, d_w, d_uhat, d_vhat,
             d_what, s_ug, s_vg, s_wg, s_ughat, s_vghat, s_wghat, d_au, d_av,
             d_aw, s_m, DWIJ):
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]

        i, j = declare('int', 2)
        ui, uj, uidif, ujdif, res = declare('matrix(3)', 5)
        Aij = declare('matrix(9)')

        for i in range(3):
            res[i] = 0.0
            for j in range(3):
                Aij[3 * i + j] = 0.0

        ui[0] = d_u[d_idx]
        ui[1] = d_v[d_idx]
        ui[2] = d_w[d_idx]

        uj[0] = s_ug[s_idx]
        uj[1] = s_vg[s_idx]
        uj[2] = s_wg[s_idx]

        uidif[0] = d_uhat[d_idx] - d_u[d_idx]
        uidif[1] = d_vhat[d_idx] - d_v[d_idx]
        uidif[2] = d_what[d_idx] - d_w[d_idx]

        ujdif[0] = s_ughat[s_idx] - s_ug[s_idx]
        ujdif[1] = s_vghat[s_idx] - s_vg[s_idx]
        ujdif[2] = s_wghat[s_idx] - s_wg[s_idx]

        for i in range(3):
            for j in range(3):
                Aij[3 * i + j] = (ui[i] * uidif[j] / rhoi +
                                  uj[i] * ujdif[j] / rhoj)

        mat_vec_mult(Aij, DWIJ, 3, res)

        d_au[d_idx] += s_m[s_idx] * res[0]
        d_av[d_idx] += s_m[s_idx] * res[1]
        d_aw[d_idx] += s_m[s_idx] * res[2]


class MakeSurfaceParticlesPressureApZeroEDACFluids(Equation):
    def initialize(self, d_idx, d_is_boundary, d_p, d_ap, d_p0):
        # if the particle is boundary set it's h_b to be zero
        if d_is_boundary[d_idx] == 1:
            d_ap[d_idx] = 0.
            d_p[d_idx] = 0.

            # for GTVF
            d_p0[d_idx] = 0.


class MomentumEquationViscosityETVF(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(MomentumEquationViscosityETVF, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_uhat, d_vhat, d_what,
             s_uhat, s_vhat, s_what, d_au, d_av, d_aw, R2IJ, EPS, DWIJ, XIJ):
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 4 * (etai * etaj) / (etai + etaj)

        xdotdij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]

        tmp = s_m[s_idx] / (d_rho[d_idx] * s_rho[s_idx])
        fac = tmp * etaij * xdotdij / (R2IJ + EPS)

        d_au[d_idx] += fac * (d_uhat[d_idx] - s_uhat[s_idx])
        d_av[d_idx] += fac * (d_vhat[d_idx] - s_vhat[s_idx])
        d_aw[d_idx] += fac * (d_what[d_idx] - s_what[s_idx])


class SolidWallNoSlipBCDensityBased(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(SolidWallNoSlipBCDensityBased, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_m, d_rho, s_m, s_rho, d_u, d_v, d_w, d_au,
             d_av, d_aw, s_ug, s_vg, s_wg, s_ugns, s_vgns, s_wgns, DWIJ, R2IJ,
             EPS, XIJ):

        # averaged shear viscosity Eq. (6).
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 2 * (etai * etaj) / (etai + etaj)

        # particle volumes; d_V inverse volume.
        Vi = d_m[d_idx] / d_rho[d_idx]
        Vj = s_m[s_idx] / s_rho[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        # scalar part of the kernel gradient
        Fij = XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2]

        # viscous contribution (third term) from Eq. (8), with VIJ
        # defined appropriately using the ghost values
        tmp = 1. / d_m[d_idx] * (Vi2 + Vj2) * (etaij * Fij / (R2IJ + EPS))

        # d_au[d_idx] += tmp * (d_u[d_idx] - s_ug[s_idx])
        # d_av[d_idx] += tmp * (d_v[d_idx] - s_vg[s_idx])
        # d_aw[d_idx] += tmp * (d_w[d_idx] - s_wg[s_idx])

        d_au[d_idx] += tmp * (d_u[d_idx] - s_ugns[s_idx])
        d_av[d_idx] += tmp * (d_v[d_idx] - s_vgns[s_idx])
        d_aw[d_idx] += tmp * (d_w[d_idx] - s_wgns[s_idx])


# class SolidWallNoSlipBCDensityBased(Equation):
#     def __init__(self, dest, sources, nu):
#         self.nu = nu
#         super(SolidWallNoSlipBCDensityBased, self).__init__(dest, sources)

#     def loop(self, d_idx, s_idx, d_m, d_rho, s_m, s_rho,
#              d_uhat, d_vhat, d_what,
#              d_au, d_av, d_aw,
#              s_ug, s_vg, s_wg,
#              DWIJ, R2IJ, EPS, XIJ):

#         # averaged shear viscosity Eq. (6).
#         etai = self.nu * d_rho[d_idx]
#         etaj = self.nu * s_rho[s_idx]

#         etaij = 2 * (etai * etaj)/(etai + etaj)

#         # particle volumes; d_V inverse volume.
#         Vi = d_m[d_idx] / d_rho[d_idx]
#         Vj = s_m[s_idx] / s_rho[s_idx]
#         Vi2 = Vi * Vi
#         Vj2 = Vj * Vj

#         # scalar part of the kernel gradient
#         Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

#         # viscous contribution (third term) from Eq. (8), with VIJ
#         # defined appropriately using the ghost values
#         tmp = 1./d_m[d_idx] * (Vi2 + Vj2) * (etaij * Fij/(R2IJ + EPS))

#         d_au[d_idx] += tmp * (d_uhat[d_idx] - s_ug[s_idx])
#         d_av[d_idx] += tmp * (d_vhat[d_idx] - s_vg[s_idx])
#         d_aw[d_idx] += tmp * (d_what[d_idx] - s_wg[s_idx])


class GTVFRigidBody3DStep(IntegratorStep):
    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.force[i3 + j] /
                                                     dst.total_mass[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_is_boundary,
               d_uhat, d_vhat, d_what):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly

        bid, i9, i3, = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw

        d_uhat[d_idx] = d_u[d_idx]
        d_vhat[d_idx] = d_v[d_idx]
        d_what[d_idx] = d_w[d_idx]

    def py_stage2(self, dst, t, dt):
        # move positions to t + dt time step
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3 + j] = dst.xcm[i3 + j] + dt * dst.vcm[i3 + j]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3 + 2], dst.omega[i3 + 1]],
                                  [dst.omega[i3 + 2], 0, -dst.omega[i3 + 0]],
                                  [-dst.omega[i3 + 1], dst.omega[i3 + 0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9 + 9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9 + 9] = dst.R[i9:i9 + 9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9 + 9])

            # update the moment of inertia
            R = dst.R[i9:i9 + 9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(
                R,
                dst.inertia_tensor_inverse_body_frame[i9:i9 + 9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9 + 9] = (np.matmul(
                tmp, R_t)).ravel()[:]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_normal0, d_normal,
               d_is_boundary):
        # some variables to update the positions seamlessly
        bid, i9, i3, idx3 = declare('int', 4)
        bid = d_body_id[d_idx]
        idx3 = 3 * d_idx
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

        # update normal vectors of the boundary
        if d_is_boundary[d_idx] == 1:
            d_normal[idx3 + 0] = (d_R[i9 + 0] * d_normal0[idx3] +
                                  d_R[i9 + 1] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 2] * d_normal0[idx3 + 2])
            d_normal[idx3 + 1] = (d_R[i9 + 3] * d_normal0[idx3] +
                                  d_R[i9 + 4] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 5] * d_normal0[idx3 + 2])
            d_normal[idx3 + 2] = (d_R[i9 + 6] * d_normal0[idx3] +
                                  d_R[i9 + 7] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 8] * d_normal0[idx3 + 2])

    def py_stage3(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.force[i3 + j] /
                                                     dst.total_mass[i])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_is_boundary):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly

        bid, i9, i3, = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw


class ETVFScheme(Scheme):
    def __init__(self, fluids, solids, rigid_bodies, dim, c0, nu, rho0, u_max,
                 mach_no, pb=0.0, gx=0.0, gy=0.0, gz=0.0, tdamp=0.0, eps=0.0,
                 h=0.0, kernel_factor=3, edac_alpha=0.5, alpha=0.0,
                 pst="sun2019", debug=False, edac=False, clamp_p=False,
                 summation=False, corrections=True, ipst_max_iterations=10,
                 ipst_tolerance=0.2, ipst_interval=5, internal_flow=False,
                 kernel_choice="1", integrator='gtvf', kr=1e5, kf=1e5, en=0.5,
                 fric_coeff=0.5, ):
        self.c0 = c0
        self.nu = nu
        self.rho0 = rho0
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        self.dim = dim
        self.eps = eps
        self.fluids = fluids
        self.solids = solids
        self.pb = pb
        self.solver = None
        self.h = h
        self.kernel_factor = kernel_factor
        self.edac_alpha = edac_alpha
        self.alpha = alpha
        self.pst = pst

        # attributes for P Sun 2019 PST technique
        self.u_max = u_max
        self.mach_no = mach_no

        # attributes for IPST technique
        self.ipst_max_iterations = ipst_max_iterations
        self.ipst_tolerance = ipst_tolerance
        self.ipst_interval = ipst_interval
        self.debug = debug
        self.internal_flow = internal_flow
        self.edac = edac
        self.clamp_p = clamp_p
        self.summation = summation
        self.corrections = corrections

        # TODO: kernel_fac will change with kernel. This should change
        self.kernel_choice = kernel_choice
        self.kernel = QuinticSpline
        self.kernel_factor = 2

        self.integrator = integrator

        # edac
        self.surf_p_zero = True

        self.attributes_changed()

        # rigid bodies specific variables
        self.rigid_bodies = rigid_bodies
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff
        self.en = en

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=None,
                           help="Alpha for the artificial viscosity.")

        group.add_argument("--edac-alpha", action="store", type=float,
                           dest="edac_alpha", default=None,
                           help="Alpha for the EDAC scheme viscosity.")

        group.add_argument("--ipst-max-iterations", action="store",
                           dest="ipst_max_iterations", default=10, type=int,
                           help="Max iterations of IPST")

        group.add_argument("--ipst-interval", action="store",
                           dest="ipst_interval", default=5, type=int,
                           help="Frequency at which IPST is to be done")

        add_bool_argument(group, 'debug', dest='debug', default=False,
                          help='Check if the IPST converged')

        add_bool_argument(group, 'edac', dest='edac', default=False,
                          help='Use pressure evolution by EDAC')

        add_bool_argument(group, 'summation', dest='summation', default=False,
                          help='Use summation density to compute the density')

        add_bool_argument(
            group, 'clamp-pressure', dest='clamp_p',
            help="Clamp pressure on boundaries to be non-negative.",
            default=None
        )

        choices = ['sun2019', 'ipst', 'tvf', 'None']
        group.add_argument(
            "--pst", action="store", dest='pst', default="sun2019",
            choices=choices,
            help="Specify what PST to use (one of %s)." % choices)

        choices = ['pec', 'gtvf']
        group.add_argument(
            "--integrator", action="store", dest="integrator", default='gtvf',
            choices=choices,
            help="Specify integrator to use (one of %s)." % choices)

        add_bool_argument(group, 'internal-flow', dest='internal_flow',
                          default=False,
                          help='Check if it is an internal flow')

        group.add_argument("--ipst-tolerance", action="store", type=float,
                           dest="ipst_tolerance", default=None,
                           help="Tolerance limit of IPST")

        add_bool_argument(
            group, 'surf-p-zero', dest='surf_p_zero', default=False,
            help='Make the surface pressure and acceleration to be zero')

        choices = ["1", "2", "3", "4", "5", "6", "7", "8"]
        group.add_argument(
            "--kernel-choice", action="store", dest='kernel_choice',
            default="1", choices=choices,
            help="""Specify what kernel to use (one of %s).
                           1. QuinticSpline
                           2. WendlandQuintic
                           3. CubicSpline
                           4. WendlandQuinticC4
                           5. Gaussian
                           6. SuperGaussian
                           7. Gaussian
                           8. Gaussian""" % choices)

        add_bool_argument(group, 'corrections', dest='corrections',
                          default=True,
                          help='Add corrections of CTVF')

        group.add_argument("--kr-stiffness", action="store",
                           dest="kr", default=1e5,
                           type=float,
                           help="Repulsive spring stiffness")

        group.add_argument("--kf-stiffness", action="store",
                           dest="kf", default=1e3,
                           type=float,
                           help="Tangential spring stiffness")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.5,
                           type=float,
                           help="Friction coefficient")

    def consume_user_options(self, options):
        vars = [
            'alpha', 'edac_alpha', 'pst', 'debug', 'ipst_max_iterations',
            'integrator', 'internal_flow', 'ipst_tolerance', 'ipst_interval',
            'surf_p_zero', 'edac', 'summation', 'kernel_choice', 'clamp_p',
            'corrections',
            'kr', 'kf', 'fric_coeff'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def attributes_changed(self):
        if self.pb is not None:
            self.use_tvf = abs(self.pb) > 1e-14
        if self.h is not None and self.c0 is not None:
            self.art_nu = self.edac_alpha * self.h * self.c0 / 8

        if self.kernel_choice == "1":
            self.kernel = QuinticSpline
            self.kernel_factor = 3
        elif self.kernel_choice == "2":
            self.kernel = WendlandQuintic
            self.kernel_factor = 2
        elif self.kernel_choice == "3":
            self.kernel = CubicSpline
            self.kernel_factor = 2
        elif self.kernel_choice == "4":
            self.kernel = WendlandQuinticC4
            self.kernel_factor = 2
            self.h = self.h / self.hdx * 2.0
        elif self.kernel_choice == "5":
            self.kernel = Gaussian
            self.kernel_factor = 3
        elif self.kernel_choice == "6":
            self.kernel = SuperGaussian
            self.kernel_factor = 3

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        kernel = self.kernel(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        if self.integrator == 'gtvf':
            step_cls = EDACGTVFStep
            cls = (integrator_cls
                   if integrator_cls is not None else GTVFIntegrator)
        elif self.integrator == 'pec':
            if self.edac is True:
                step_cls = EDACPECStep
            else:
                step_cls = PECStep
            cls = (integrator_cls
                   if integrator_cls is not None else PECIntegrator)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        bodystep = GTVFRigidBody3DStep()
        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = bodystep


        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def check_ipst_time(self, t, dt):
        if int(t / dt) % self.ipst_interval == 0:
            return True
        else:
            return False

    def get_equations(self):
        if self.integrator == 'gtvf':
            return self._get_gtvf_equations()

        elif self.integrator == 'pec':
            return self._get_pec_equations()

    def _get_pec_equations(self):
        from pysph.sph.wc.gtvf import (MomentumEquationArtificialStress,
                                       MomentumEquationViscosity)
        from pysph.sph.wc.transport_velocity import (SetWallVelocity)
        from pysph.sph.iisph import (SummationDensity)
        # from pysph.sph.basic_equations import (ContinuityEquation)
        from pysph.sph.wc.gtvf import (ContinuityEquationGTVF)
        nu_edac = self._get_edac_nu()
        all = self.fluids + self.solids
        stage1 = []

        if self.edac is False:
            tmp = []
            for fluid in self.fluids:
                tmp.append(
                    StateEquation(dest=fluid, sources=None, p0=self.pb,
                                  rho0=self.rho0))

            stage1.append(Group(equations=tmp, real=False))

        if self.summation is True:
            tmp = []
            for fluid in self.fluids:
                tmp.append(SummationDensity(dest=fluid, sources=all))

            stage1.append(Group(equations=tmp, real=False))

        eqs = []
        if len(self.solids) > 0:
            for solid in self.solids:
                eqs.append(
                    SetWallVelocityFreeSlipAndNoSlip(dest=solid,
                                                     sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=solid, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz), )

            stage1.append(Group(equations=eqs, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    # This equation will only project uhat of fluids onto
                    # solids (ughat)
                    # SetWallVelocityUhatNoSlip(dest=solid,
                    #                           sources=self.fluids)
                    SetWallVelocityUhatFreeSlipAndNoSlip(
                        dest=solid, sources=self.fluids))

            stage1.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            eqs.append(
                # This equation will only project uhat of fluids onto
                # solids (ughat)
                # SetWallVelocityUhatNoSlip(dest=solid,
                #                           sources=self.fluids)
                SetHIJForInsideParticles(dest=fluid, sources=self.fluids,
                                         h=self.h, kernel_factor=3))

            if self.internal_flow is False:
                eqs.append(SummationDensitySplash(dest=fluid,
                                                  sources=self.fluids))

        stage1.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            if self.summation is False:
                eqs.append(
                    ContinuityEquationGTVF(dest=fluid, sources=self.fluids))
                eqs.append(
                    ContinuityEquationETVFCorrection(dest=fluid,
                                                     sources=self.fluids))
            if self.edac is True:
                eqs.append(
                    EDACEquation(dest=fluid, sources=self.fluids,
                                 nu=nu_edac), )

            eqs.append(
                MomentumEquationPressureGradient(dest=fluid, sources=all,
                                                 gx=self.gx, gy=self.gy,
                                                 gz=self.gz))

            eqs.append(
                MomentumEquationArtificialStress(dest=fluid,
                                                 sources=self.fluids,
                                                 dim=self.dim))
            eqs.append(
                MomentumEquationTVFDivergence(dest=fluid, sources=self.fluids))

        if len(self.solids) > 0:
            if self.summation is False:
                for fluid in self.fluids:
                    eqs.append(
                        ContinuitySolidEquationGTVF(dest=fluid,
                                                    sources=self.solids), )
                    eqs.append(
                        ContinuitySolidEquationETVFCorrection(
                            dest=fluid, sources=self.solids), )
            if self.edac is True:
                eqs.append(
                    EDACSolidEquation(dest=fluid, sources=self.solids,
                                      nu=nu_edac), )

        if self.pst == 'tvf':
            eqs.append(ComputeAuHatETVF(dest=fluid, sources=all, pb=self.pb))

        elif self.pst == 'sun2019':
            eqs.append(
                ComputeAuHatETVFSun2019(dest=fluid, sources=all,
                                        mach_no=self.mach_no,
                                        u_max=self.u_max))

        if self.nu > 0:
            eqs.append(
                MomentumEquationViscosity(dest=fluid, sources=self.fluids,
                                          nu=self.nu),
                # MomentumEquationViscosity(dest=fluid,
                #                           sources=all,
                #                           nu=self.nu)
            )
            if len(self.solids) > 0:
                eqs.append(
                    MomentumEquationViscosityNoSlip(dest=fluid,
                                                    sources=self.solids,
                                                    nu=self.nu)

                    # MomentumEquationViscosity(dest=fluid,
                    #                           sources=self.solids,
                    #                           nu=self.nu)
                )

        stage1.append(Group(equations=eqs, real=True))

        if self.pst == 'ipst':
            g5 = []
            g6 = []
            g7 = []
            g8 = []

            # make auhat zero before computation of ipst force
            eqns = []
            for fluid in self.fluids:
                eqns.append(MakeAuhatZero(dest=fluid, sources=None))

            stage1.append(Group(eqns))

            for fluid in self.fluids:
                g5.append(
                    SavePositionsIPSTBeforeMoving(dest=fluid, sources=None))

                # these two has to be in the iterative group and the nnps has to
                # be updated
                # ---------------------------------------
                g6.append(
                    AdjustPositionIPST(dest=fluid,
                                       sources=[fluid] + self.solids,
                                       u_max=self.u_max))

                g7.append(
                    CheckUniformityIPST(dest=fluid,
                                        sources=[fluid] + self.solids,
                                        debug=self.debug))
                # ---------------------------------------

                g8.append(ComputeAuhatETVFIPST(dest=fluid, sources=None))
                g8.append(ResetParticlePositionsIPST(dest=fluid, sources=None))

            stage1.append(Group(g5, condition=self.check_ipst_time))

            # this is the iterative group
            stage1.append(
                Group(equations=[Group(equations=g6),
                                 Group(equations=g7)], iterate=True,
                      max_iterations=self.ipst_max_iterations,
                      condition=self.check_ipst_time))

            stage1.append(Group(g8, condition=self.check_ipst_time))

        return stage1

    def _get_gtvf_equations(self):
        from pysph.sph.wc.gtvf import (ContinuityEquationGTVF,
                                       MomentumEquationArtificialStress,
                                       MomentumEquationViscosity)
        from pysph.sph.iisph import (SummationDensity)
        from rigid_fluid_coupling import (
            ContinuityRigidBodyEquationGTVF,
            ContinuityRigidBodyEquationETVFCorrection,
            EDACRigidBodyEquation,
            FluidEDACRigidBodyEquationNoCorrections,
            RigidBodyWallPressureBC,
            ClampWallPressureRigidBody,
            ComputeAuHatETVFRigidBodySun2019,
            ForceOnFluidDueToRigidBody,
            ForceOnRigidBodyDueToFluid)

        nu_edac = self._get_edac_nu()
        print("nu_edac is ", nu_edac)
        all = self.fluids + self.solids
        stage1 = []

        eqs = []
        if len(self.solids) > 0:
            for solid in self.solids:
                eqs.append(
                    SetWallVelocityFreeSlipAndNoSlip(dest=solid,
                                                     sources=self.fluids), )

            stage1.append(Group(equations=eqs, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    SetWallVelocityUhatFreeSlipAndNoSlip(dest=solid,
                                                         sources=self.fluids))

            stage1.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            if self.summation is False:
                eqs.append(ContinuityEquationGTVF(dest=fluid,
                                                  sources=self.fluids), )
                if self.corrections == True:
                    eqs.append(
                        ContinuityEquationETVFCorrection(dest=fluid,
                                                         sources=self.fluids),
                    )

            if self.edac is True:
                if self.corrections == True:
                    eqs.append(
                        EDACEquation(dest=fluid, sources=self.fluids,
                                     nu=nu_edac), )
                else:
                    eqs.append(
                        FluidEDACEquationNoCorrections(dest=fluid,
                                                       sources=self.fluids,
                                                       nu=nu_edac), )

        if len(self.solids) > 0:
            if self.summation is False:
                for fluid in self.fluids:
                    eqs.append(
                        ContinuitySolidEquationGTVF(dest=fluid,
                                                    sources=self.solids), )
                    if self.corrections == True:
                        eqs.append(
                            ContinuitySolidEquationETVFCorrection(
                                dest=fluid, sources=self.solids), )

            if self.edac is True:
                if self.corrections == True:
                    eqs.append(
                        EDACSolidEquation(dest=fluid, sources=self.solids,
                                          nu=nu_edac), )
                else:
                    eqs.append(
                        FluidEDACSolidEquationNoCorrections(dest=fluid,
                                                            sources=self.solids,
                                                            nu=nu_edac), )

        if len(self.rigid_bodies) > 0:
            if self.summation is False:
                for fluid in self.fluids:
                    eqs.append(
                        ContinuityRigidBodyEquationGTVF(dest=fluid,
                                                        sources=self.rigid_bodies), )
                    if self.corrections == True:
                        eqs.append(
                            ContinuityRigidBodyEquationETVFCorrection(
                                dest=fluid,
                                sources=self.rigid_bodies), )

            if self.edac is True:
                if self.corrections == True:
                    eqs.append(
                        EDACRigidBodyEquation(dest=fluid, sources=self.rigid_bodies,
                                              nu=nu_edac), )
                else:
                    eqs.append(
                        FluidEDACRigidBodyEquationNoCorrections(dest=fluid,
                                                                sources=self.rigid_bodies,
                                                                nu=nu_edac), )

            eqs.append(
                SummationDensityDummy(dest=fluid,
                                      sources=self.fluids+self.solids))

        stage1.append(Group(equations=eqs, real=False))

        if self.surf_p_zero is True:
            eqs = []
            for pa in self.fluids:
                eqs.append(
                    MakeSurfaceParticlesPressureApZeroEDACFluids(
                        dest=pa, sources=None))

            stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # stage 2 equations start
        # =========================#

        stage2 = []

        if self.edac is False:
            tmp = []
            for fluid in self.fluids:
                tmp.append(
                    StateEquation(dest=fluid, sources=None, p0=self.pb,
                                  rho0=self.rho0))

            stage2.append(Group(equations=tmp, real=False))

        if len(self.solids) > 0:
            eqs = []
            for solid in self.solids:
                eqs.append(
                    SetWallVelocityFreeSlipAndNoSlip(dest=solid, sources=self.fluids))

                eqs.append(
                    SolidWallPressureBC(dest=solid, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                if self.clamp_p:
                    eqs.append(
                        ClampWallPressure(dest=solid, sources=None))

            stage2.append(Group(equations=eqs, real=False))

        if len(self.rigid_bodies) > 0:
            eqs = []
            for body in self.rigid_bodies:
                eqs.append(
                    RigidBodyWallPressureBC(dest=body, sources=self.fluids,
                                            gx=self.gx, gy=self.gy, gz=self.gz,
                                            c0=self.c0, rho0=self.rho0))
                # if self.clamp_p:
                #     eqs.append(
                #         ClampWallPressureRigidBody(dest=body, sources=None))

            stage2.append(Group(equations=eqs, real=False))

        # if len(self.solids) > 0:
        #     eqs = []
        #     for solid in self.solids:
        #         eqs.append(
        #             SetWallVelocityUhatFreeSlipAndNoSlip(dest=solid,
        #                                                  sources=self.fluids))

        #     stage2.append(Group(equations=eqs, real=False))

        eqs = []
        if self.internal_flow is not True:
            for fluid in self.fluids:
                eqs.append(
                    SetHIJForInsideParticles(dest=fluid, sources=[fluid],
                                             h=self.h,
                                             kernel_factor=self.kernel_factor))
                eqs.append(SummationDensitySplash(dest=fluid,
                                                  sources=self.fluids+self.solids))

                if len(self.rigid_bodies) > 0:
                    eqs.append(SummationDensitySplashRigidBody(
                        dest=fluid,
                        sources=self.rigid_bodies))
            stage2.append(Group(eqs))

        # summation density
        if self.summation is True:
            tmp = []
            for fluid in self.fluids:
                tmp.append(SummationDensity(dest=fluid, sources=all))

            stage2.append(Group(equations=tmp, real=False))

        eqs = []
        for fluid in self.fluids:
            # FIXME: Change alpha to variable
            if self.alpha > 0.:
                eqs.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=all+self.rigid_bodies, c0=self.c0,
                        alpha=self.alpha
                    )
                )
            eqs.append(
                MomentumEquationPressureGradient(dest=fluid, sources=all,
                                                 gx=self.gx, gy=self.gy,
                                                 gz=self.gz), )

            eqs.append(
                ForceOnFluidDueToRigidBody(dest=fluid,
                                           sources=self.rigid_bodies))
            if self.corrections == True:
                eqs.append(
                    MomentumEquationArtificialStress(dest=fluid,
                                                     sources=self.fluids,
                                                     dim=self.dim))
                eqs.append(
                    MomentumEquationTVFDivergence(dest=fluid, sources=self.fluids))

            if self.pst == 'tvf':
                eqs.append(
                    ComputeAuHatETVF(dest=fluid, sources=all, pb=self.pb))

            elif self.pst == 'sun2019':
                eqs.append(
                    ComputeAuHatETVFSun2019(dest=fluid, sources=all,
                                            mach_no=self.mach_no,
                                            u_max=self.u_max,
                                            rho0=self.rho0))

                if len(self.rigid_bodies) > 0:
                    eqs.append(
                        ComputeAuHatETVFRigidBodySun2019(
                            dest=fluid,
                            sources=self.rigid_bodies,
                            mach_no=self.mach_no,
                            u_max=self.u_max,
                            rho0=self.rho0))

            if self.nu > 0:
                eqs.append(
                    MomentumEquationViscosity(dest=fluid, sources=self.fluids,
                                              nu=self.nu))

                if len(self.solids) > 0:
                    eqs.append(
                        MomentumEquationViscosityNoSlip(
                            dest=fluid, sources=self.solids, nu=self.nu))

        stage2.append(Group(equations=eqs, real=True))

        # this PST is handled separately
        if self.pst == "ipst":
            g5 = []
            g6 = []
            g7 = []
            g8 = []

            # make auhat zero before computation of ipst force
            eqns = []
            for fluid in self.fluids:
                eqns.append(MakeAuhatZero(dest=fluid, sources=None))

            stage2.append(Group(eqns))

            for fluid in self.fluids:
                g5.append(
                    SavePositionsIPSTBeforeMoving(dest=fluid, sources=None))

                # these two has to be in the iterative group and the nnps has to
                # be updated
                # ---------------------------------------
                g6.append(
                    AdjustPositionIPST(dest=fluid, sources=all,
                                       u_max=self.u_max))

                if self.internal_flow == True:
                    g7.append(
                        CheckUniformityIPSTFluidInternalFlow(
                            dest=fluid, sources=all, debug=self.debug,
                            tolerance=self.ipst_tolerance))
                else:
                    g7.append(
                        CheckUniformityIPST(dest=fluid, sources=all,
                                            debug=self.debug,
                                            tolerance=self.ipst_tolerance))

                # ---------------------------------------
                g8.append(ComputeAuhatETVFIPSTFluids(dest=fluid, sources=None,
                                                     rho0=self.rho0))
                g8.append(ResetParticlePositionsIPST(dest=fluid, sources=None))

            stage2.append(Group(g5, condition=self.check_ipst_time))

            # this is the iterative group
            stage2.append(
                Group(equations=[Group(equations=g6),
                                 Group(equations=g7)], iterate=True,
                      max_iterations=self.ipst_max_iterations,
                      condition=self.check_ipst_time))

            stage2.append(Group(g8, condition=self.check_ipst_time))

        #######################
        # Handle rigid bodies #
        #######################
        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ResetForce(dest=name, sources=None))
            stage2.append(Group(equations=g5, real=False))

        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForceNormalsMohseni(
                        dest=name, sources=self.rigid_bodies))

            stage2.append(Group(equations=g5, real=False))

            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForceDistanceAndClosestPointMohseni(
                        dest=name, sources=self.rigid_bodies))
            stage2.append(Group(equations=g5, real=False))

        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    ComputeContactForceMohseni(dest=name,
                                               sources=None,
                                               kr=self.kr,
                                               kf=self.kf,
                                               fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g5, real=False))

        if len(self.rigid_bodies) > 0:
            g5 = []
            for name in self.rigid_bodies:
                g5.append(
                    TransferContactForceMohseni(
                        dest=name, sources=self.rigid_bodies))

            stage2.append(Group(equations=g5, real=False))

        # add the force due to fluid
        tmp = []
        if len(self.fluids) > 0:
            for name in self.rigid_bodies:
                g5.append(ForceOnRigidBodyDueToFluid(
                    dest=name, sources=self.fluids))
            stage2.append(Group(equations=tmp, real=False))

        # computation of total force and torque at center of mass
        if len(self.rigid_bodies) > 0:
            g6 = []
            for name in self.rigid_bodies:
                g6.append(SumUpExternalForces(dest=name,
                                              sources=None,
                                              gx=self.gx,
                                              gy=self.gy,
                                              gz=self.gz))

            stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def setup_properties(self, particles, clean=True):
        pas = dict([(p.name, p) for p in particles])
        for fluid in self.fluids:
            pa = pas[fluid]

            add_properties(pa, 'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'rho0',
                           'arho', 'ap', 'arho', 'p0', 'uhat', 'vhat', 'what',
                           'auhat', 'avhat', 'awhat', 'h_b', 'V', 'div_r')

            pa.h_b[:] = pa.h[:]

            if 'c0_ref' not in pa.constants:
                pa.add_constant('c0_ref', self.c0)
            pa.add_output_arrays(['p'])

            if self.pst == "ipst":
                if self.internal_flow == True:
                    setup_ipst_fluids(pa, source_pa=particles, dim=self.dim)
                else:
                    setup_ipst_fluids(pa, source_pa=particles, dim=self.dim)

                pa.add_output_arrays(['ipst_chi'])

            elif self.pst == "sun2019":
                if 'wdeltap' not in pa.constants:
                    pa.add_constant('wdeltap', -1.)

                if 'n' not in pa.constants:
                    pa.add_constant('n', 0.)

            add_boundary_identification_properties(pa)
            add_properties(pa, 'rho_splash')
            pa.add_output_arrays(['rho_splash'])
            pa.rho_splash[:] = self.rho0

            add_properties(pa, 'rho_dummy')
            pa.add_output_arrays(['rho_dummy'])

            pa.h_b[:] = pa.h

        for solid in self.solids:
            pa = pas[solid]

            add_properties(pa, 'rho', 'V', 'wij2', 'wij', 'uhat', 'vhat',
                           'what')

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf')
            add_properties(pa, 'ugfs', 'vgfs', 'wgfs')

            add_properties(pa, 'ughatns', 'vghatns', 'wghatns')
            add_properties(pa, 'ughatfs', 'vghatfs', 'wghatfs')

            # No slip boundary conditions for viscosity force
            add_properties(pa, 'ugns', 'vgns', 'wgns')

            # pa.h_b[:] = pa.h[:]

            # for normals
            pa.add_property('normal', stride=3)
            pa.add_output_arrays(['normal'])
            pa.add_property('normal_tmp', stride=3)

            name = pa.name

            props = ['m', 'rho', 'h']
            for p in props:
                x = pa.get(p)
                if numpy.all(x < 1e-12):
                    msg = f'WARNING: cannot compute normals "{p}" is zero'
                    print(msg)

            seval = SPHEvaluator(
                arrays=[pa], equations=[
                    Group(
                        equations=[ComputeNormals(dest=name, sources=[name])]),
                    Group(
                        equations=[SmoothNormals(dest=name, sources=[name])]),
                ], dim=self.dim)
            seval.evaluate()

        for rigid_body in self.rigid_bodies:
            pa = pas[rigid_body]

            # properties to find the find on the rigid body by
            # Mofidi, Drescher, Emden, Teschner
            add_properties_stride(pa, pa.total_no_bodies[0],
                                  'contact_force_normal_x',
                                  'contact_force_normal_y',
                                  'contact_force_normal_z',
                                  'contact_force_normal_wij',

                                  'contact_force_normal_x_source',
                                  'contact_force_normal_y_source',
                                  'contact_force_normal_z_source',
                                  'contact_force_dist_source',

                                  'contact_force_normal_tmp_x',
                                  'contact_force_normal_tmp_y',
                                  'contact_force_normal_tmp_z',

                                  'contact_force_dist_tmp',
                                  'contact_force_dist',

                                  'overlap',
                                  'ft_x',
                                  'ft_y',
                                  'ft_z',
                                  'fn_x',
                                  'fn_y',
                                  'fn_z',
                                  'delta_lt_x',
                                  'delta_lt_y',
                                  'delta_lt_z',
                                  'vx_source',
                                  'vy_source',
                                  'vz_source',
                                  'x_source',
                                  'y_source',
                                  'z_source',
                                  'ti_x',
                                  'ti_y',
                                  'ti_z',
                                  'closest_point_dist_to_source',
                                  'contact_force_weight_denominator',

                                  'total_mass_source',
                                  'radius_vec_dist_source')

            pa.add_property(name='idx_source',
                            stride=pa.total_no_bodies[0],
                            type='int')

            add_properties(pa, 'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0')

            nb = int(np.max(pa.body_id) + 1)

            # dem_id = props.pop('dem_id', None)

            consts = {
                'total_mass':
                np.zeros(nb, dtype=float),
                'xcm':
                np.zeros(3 * nb, dtype=float),
                'xcm0':
                np.zeros(3 * nb, dtype=float),
                'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                'R0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
                # moment of inertia izz (this is only for 2d)
                'izz':
                np.zeros(nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_body_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_body_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_global_frame':
                np.zeros(9 * nb, dtype=float),
                # moment of inertia inverse in body frame
                'inertia_tensor_inverse_global_frame':
                np.zeros(9 * nb, dtype=float),
                # total force at the center of mass
                'force':
                np.zeros(3 * nb, dtype=float),
                # torque about the center of mass
                'torque':
                np.zeros(3 * nb, dtype=float),
                # velocity, acceleration of CM.
                'vcm':
                np.zeros(3 * nb, dtype=float),
                'vcm0':
                np.zeros(3 * nb, dtype=float),
                # angular momentum
                'ang_mom':
                np.zeros(3 * nb, dtype=float),
                'ang_mom0':
                np.zeros(3 * nb, dtype=float),
                # angular velocity in global frame
                'omega':
                np.zeros(3 * nb, dtype=float),
                'omega0':
                np.zeros(3 * nb, dtype=float),
                'nb':
                nb
            }

            for key, elem in consts.items():
                pa.add_constant(key, elem)

            pa.add_constant('min_dem_id', min(pa.dem_id))
            pa.add_constant('max_dem_id', max(pa.dem_id))

            eta = np.zeros(pa.nb[0]*pa.total_no_bodies[0] * 1,
                           dtype=float)
            pa.add_constant('eta', eta)

            pa.add_property(name='dem_id_source', stride=pa.total_no_bodies[0],
                            type='int')

            # compute the properties of the body
            set_total_mass(pa)
            set_center_of_mass(pa)

            # this function will compute
            # inertia_tensor_body_frame
            # inertia_tensor_inverse_body_frame
            # inertia_tensor_global_frame
            # inertia_tensor_inverse_global_frame
            # of the rigid body
            set_moment_of_inertia_and_its_inverse(pa)

            set_body_frame_position_vectors(pa)

            ####################################################
            # compute the boundary particles of the rigid body #
            ####################################################
            add_boundary_identification_properties(pa)
            # make sure your rho is not zero
            equations = get_boundary_identification_etvf_equations([pa.name],
                                                                   [pa.name])
            # print(equations)

            sph_eval = SPHEvaluator(arrays=[pa],
                                    equations=equations,
                                    dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim))

            sph_eval.evaluate(dt=0.1)

            # make normals of particle other than boundary particle as zero
            # for i in range(len(pa.x)):
            #     if pa.is_boundary[i] == 0:
            #         pa.normal[3 * i] = 0.
            #         pa.normal[3 * i + 1] = 0.
            #         pa.normal[3 * i + 2] = 0.

            # normal vectors in terms of body frame
            set_body_frame_normal_vectors(pa)

            # Adami boundary conditions. SetWallVelocity
            add_properties(pa, 'rho_fsi', 'm_fsi', 'p_fsi')
            add_properties(pa, 'ug', 'vf', 'vg', 'wg', 'uf', 'wf', 'wij')
            add_properties(pa, 'uhat', 'vhat', 'what')
            pa.add_property('wij')

            pa.set_output_arrays([
                'x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'normal',
                'is_boundary', 'fz', 'm', 'body_id', 'h'
            ])

    def _set_particle_velocities(self, pa):
        for i in range(max(pa.body_id) + 1):
            fltr = np.where(pa.body_id == i)
            bid = i
            i9 = 9 * bid
            i3 = 3 * bid

            for j in fltr:
                dx = (pa.R[i9 + 0] * pa.dx0[j] + pa.R[i9 + 1] * pa.dy0[j] +
                      pa.R[i9 + 2] * pa.dz0[j])
                dy = (pa.R[i9 + 3] * pa.dx0[j] + pa.R[i9 + 4] * pa.dy0[j] +
                      pa.R[i9 + 5] * pa.dz0[j])
                dz = (pa.R[i9 + 6] * pa.dx0[j] + pa.R[i9 + 7] * pa.dy0[j] +
                      pa.R[i9 + 8] * pa.dz0[j])

                du = pa.omega[i3 + 1] * dz - pa.omega[i3 + 2] * dy
                dv = pa.omega[i3 + 2] * dx - pa.omega[i3 + 0] * dz
                dw = pa.omega[i3 + 0] * dy - pa.omega[i3 + 1] * dx

                pa.u[j] = pa.vcm[i3 + 0] + du
                pa.v[j] = pa.vcm[i3 + 1] + dv
                pa.w[j] = pa.vcm[i3 + 2] + dw

    def set_linear_velocity(self, pa, linear_vel):
        pa.vcm[:] = linear_vel

        self._set_particle_velocities(pa)

    def set_angular_velocity(self, pa, angular_vel):
        pa.omega[:] = angular_vel[:]

        # set the angular momentum
        for i in range(max(pa.body_id) + 1):
            i9 = 9 * i
            i3 = 3 * i
            pa.ang_mom[i3:i3 + 3] = np.matmul(
                pa.inertia_tensor_global_frame[i9:i9 + 9].reshape(3, 3),
                pa.omega[i3:i3 + 3])[:]

        self._set_particle_velocities(pa)

    def get_solver(self):
        return self.solver

    def _get_edac_nu(self):
        if self.art_nu > 0:
            nu = self.art_nu
            print(self.art_nu)
            print("Using artificial viscosity for EDAC with nu = %s" % nu)
        else:
            nu = self.nu
            print("Using real viscosity for EDAC with nu = %s" % self.nu)
        return nu
