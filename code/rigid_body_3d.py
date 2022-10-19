import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import (QuinticSpline)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

# from pysph.sph.rigid_body import (BodyForce)

from pysph.base.kernels import (QuinticSpline)

from pysph.sph.wc.gtvf import GTVFIntegrator
from rigid_rigid_interaction_equations import (
    ComputeContactForceNormalsMohseni,
    ComputeContactForceDistanceAndClosestPointMohseni,
    ComputeContactForceMohseni,
    TransferContactForceMohseni)

# compute the boundary particles
from boundary_particles import (get_boundary_identification_etvf_equations,
                                add_boundary_identification_properties)


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


def add_properties_stride(pa, stride=1, *props):
    for prop in props:
        pa.add_property(name=prop, stride=stride)


def set_total_mass(pa):
    # left limit of body i
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.total_mass[i] = np.sum(pa.m[fltr])
        assert pa.total_mass[i] > 0., "Total mass has to be greater than zero"


def set_center_of_mass(pa):
    # loop over all the bodies
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.xcm[3 * i] = np.sum(pa.m[fltr] * pa.x[fltr]) / pa.total_mass[i]
        pa.xcm[3 * i + 1] = np.sum(pa.m[fltr] * pa.y[fltr]) / pa.total_mass[i]
        pa.xcm[3 * i + 2] = np.sum(pa.m[fltr] * pa.z[fltr]) / pa.total_mass[i]


def set_moment_of_inertia_izz(pa):
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        izz = np.sum(pa.m[fltr] * ((pa.x[fltr] - pa.xcm[3 * i])**2. +
                                   (pa.y[fltr] - pa.xcm[3 * i + 1])**2.))
        pa.izz[i] = izz


def set_moment_of_inertia_and_its_inverse(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.xcm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. +
                               (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        pa.inertia_tensor_body_frame[9 * i:9 * i + 9] = I[:]

        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.inertia_tensor_inverse_body_frame[9 * i:9 * i + 9] = I_inv[:]

        # set the moment of inertia inverse in global frame
        # NOTE: This will be only computed once to compute the angular
        # momentum in the beginning.
        pa.inertia_tensor_global_frame[9 * i:9 * i + 9] = I[:]
        # set the moment of inertia inverse in global frame
        pa.inertia_tensor_inverse_global_frame[9 * i:9 * i + 9] = I_inv[:]


def set_body_frame_position_vectors(pa):
    """Save the position vectors w.r.t body frame"""
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.xcm[3 * i:3 * i + 3]
        for j in fltr:
            pa.dx0[j] = pa.x[j] - cm_i[0]
            pa.dy0[j] = pa.y[j] - cm_i[1]
            pa.dz0[j] = pa.z[j] - cm_i[2]


def set_body_frame_normal_vectors(pa):
    """Save the normal vectors w.r.t body frame"""
    pa.normal0[:] = pa.normal[:]


class ResetForce(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.


class SumUpExternalForces(Equation):
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(SumUpExternalForces, self).__init__(dest, sources)

    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        xcm = declare('object')
        total_mass = declare('object')
        body_id = declare('object')
        j = declare('int')
        i = declare('int')
        i3 = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        xcm = dst.xcm
        total_mass = dst.total_mass
        body_id = dst.body_id

        frc[:] = 0
        trq[:] = 0

        for j in range(len(x)):
            i = body_id[j]
            i3 = 3 * i
            frc[i3] += fx[j]
            frc[i3 + 1] += fy[j]
            frc[i3 + 2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i
            dx = x[j] - xcm[i3]
            dy = y[j] - xcm[i3 + 1]
            dz = z[j] - xcm[i3 + 2]

            # torque due to force on particle i
            # dri \cross fi
            trq[i3] += (dy * fz[j] - dz * fy[j])
            trq[i3 + 1] += (dz * fx[j] - dx * fz[j])
            trq[i3 + 2] += (dx * fy[j] - dy * fx[j])

        # add body force
        for i in range(max(body_id) + 1):
            i3 = 3 * i
            frc[i3] += total_mass[i] * self.gx
            frc[i3 + 1] += total_mass[i] * self.gy
            frc[i3 + 2] += total_mass[i] * self.gz


def normalize_R_orientation(orien):
    a1 = np.array([orien[0], orien[3], orien[6]])
    a2 = np.array([orien[1], orien[4], orien[7]])
    a3 = np.array([orien[2], orien[5], orien[8]])
    # norm of col0
    na1 = np.linalg.norm(a1)

    b1 = a1 / na1

    b2 = a2 - np.dot(b1, a2) * b1
    nb2 = np.linalg.norm(b2)
    b2 = b2 / nb2

    b3 = a3 - np.dot(b1, a3) * b1 - np.dot(b2, a3) * b2
    nb3 = np.linalg.norm(b3)
    b3 = b3 / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]


class GTVFRigidBody3DStep(IntegratorStep):
    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.acm[i3 + j] = dst.force[i3 + j] / dst.total_mass[i]
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.acm[i3 + j])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

            # compute the angular acceleration
            # https://physics.stackexchange.com/questions/688426/compute-angular-acceleration-from-torque-in-3d
            omega_cross_L = np.cross(dst.omega[i3:i3 + 3],
                                     dst.ang_mom[i3:i3 + 3])
            tmp = dst.torque[i3:i3 + 3] - omega_cross_L
            dst.ang_acc[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), tmp)

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_au, d_av, d_aw, d_xcm, d_vcm, d_acm, d_ang_acc, d_R, d_omega,
               d_body_id, d_is_boundary):
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

        # for particle acceleration we follow this
        # https://www.brown.edu/Departments/Engineering/Courses/En4/notes_old/RigidKinematics/rigkin.htm
        omega_omega_cross_x = d_omega[i3 + 1] * dw - d_omega[i3 + 2] * dv
        omega_omega_cross_y = d_omega[i3 + 2] * du - d_omega[i3 + 0] * dw
        omega_omega_cross_z = d_omega[i3 + 0] * dv - d_omega[i3 + 1] * du
        ang_acc_cross_x = d_ang_acc[i3 + 1] * dz - d_ang_acc[i3 + 2] * dy
        ang_acc_cross_y = d_ang_acc[i3 + 2] * dx - d_ang_acc[i3 + 0] * dz
        ang_acc_cross_z = d_ang_acc[i3 + 0] * dy - d_ang_acc[i3 + 1] * dx
        d_au[d_idx] = d_acm[i3 + 0] + omega_omega_cross_x + ang_acc_cross_x
        d_av[d_idx] = d_acm[i3 + 1] + omega_omega_cross_y + ang_acc_cross_y
        d_aw[d_idx] = d_acm[i3 + 2] + omega_omega_cross_z + ang_acc_cross_z

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
                dst.acm[i3 + j] = dst.force[i3 + j] / dst.total_mass[i]
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.acm[i3 + j])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

            # compute the angular acceleration
            # https://physics.stackexchange.com/questions/688426/compute-angular-acceleration-from-torque-in-3d
            omega_cross_L = np.cross(dst.omega[i3:i3 + 3],
                                     dst.ang_mom[i3:i3 + 3])
            tmp = dst.torque[i3:i3 + 3] - omega_cross_L
            dst.ang_acc[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), tmp)

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_au, d_av, d_aw, d_xcm, d_vcm, d_acm, d_ang_acc, d_R, d_omega,
               d_body_id, d_is_boundary):
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

        # for particle acceleration we follow this
        # https://www.brown.edu/Departments/Engineering/Courses/En4/notes_old/RigidKinematics/rigkin.htm
        omega_omega_cross_x = d_omega[i3 + 1] * dw - d_omega[i3 + 2] * dv
        omega_omega_cross_y = d_omega[i3 + 2] * du - d_omega[i3 + 0] * dw
        omega_omega_cross_z = d_omega[i3 + 0] * dv - d_omega[i3 + 1] * du
        ang_acc_cross_x = d_ang_acc[i3 + 1] * dz - d_ang_acc[i3 + 2] * dy
        ang_acc_cross_y = d_ang_acc[i3 + 2] * dx - d_ang_acc[i3 + 0] * dz
        ang_acc_cross_z = d_ang_acc[i3 + 0] * dy - d_ang_acc[i3 + 1] * dx
        d_au[d_idx] = d_acm[i3 + 0] + omega_omega_cross_x + ang_acc_cross_x
        d_av[d_idx] = d_acm[i3 + 1] + omega_omega_cross_y + ang_acc_cross_y
        d_aw[d_idx] = d_acm[i3 + 2] + omega_omega_cross_z + ang_acc_cross_z


class RigidBody3DScheme(Scheme):
    def __init__(self, rigid_bodies, boundaries, dim, kr=1e5, kf=1e5, en=0.5,
                 Cn=1.4*1e-5, fric_coeff=0.5, gx=0.0, gy=0.0, gz=0.0):
        self.rigid_bodies = rigid_bodies

        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        if rigid_bodies is None:
            self.rigid_bodies = []
        else:
            self.rigid_bodies = rigid_bodies

        # rigid body parameters
        self.dim = dim

        self.kernel = QuinticSpline

        self.integrator = "gtvf"

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kr = kr
        self.kf = kf
        self.fric_coeff = fric_coeff
        self.Cn = Cn
        self.contact_force_model = "Mohseni_Vyas"
        self.linear_contact_force = True
        self.non_linear_contact_force = False

        self.solver = None

    def add_user_options(self, group):
        # choices = ['bui', 'canelas']
        # group.add_argument("--dem",
        #                    action="store",
        #                    dest='dem',
        #                    default="bui",
        #                    choices=choices,
        #                    help="DEM interaction " % choices)

        group.add_argument("--kr-stiffness", action="store",
                           dest="kr", default=1e5,
                           type=float,
                           help="Repulsive spring stiffness")

        group.add_argument("--kf-stiffness", action="store",
                           dest="kf", default=1e3,
                           type=float,
                           help="Tangential spring stiffness")

        group.add_argument("--Cn", action="store",
                           dest="Cn", default=1.4*1e-5,
                           type=float,
                           help="Damping coefficient")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.5,
                           type=float,
                           help="Friction coefficient")

    def consume_user_options(self, options):
        _vars = ['kr', 'kf', 'fric_coeff', 'Cn']

        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        # elastic solid equations
        stage1 = []
        # ==============================
        # Stage 2 equations
        # ==============================
        stage2 = []
        #######################
        # Handle rigid bodies #
        #######################
        g5 = []
        for name in self.rigid_bodies:
            g5.append(
                ResetForce(dest=name, sources=None))
        stage2.append(Group(equations=g5, real=False))

        g5 = []
        for name in self.rigid_bodies:
            g5.append(
                ComputeContactForceNormalsMohseni(
                    dest=name, sources=self.rigid_bodies+self.boundaries))

        stage2.append(Group(equations=g5, real=False))

        g5 = []
        for name in self.rigid_bodies:
            g5.append(
                ComputeContactForceDistanceAndClosestPointMohseni(
                    dest=name, sources=self.rigid_bodies+self.boundaries))
        stage2.append(Group(equations=g5, real=False))

        g5 = []
        for name in self.rigid_bodies:
            g5.append(
                ComputeContactForceMohseni(dest=name,
                                           sources=None,
                                           kr=self.kr,
                                           kf=self.kf,
                                           fric_coeff=self.fric_coeff))

        stage2.append(Group(equations=g5, real=False))

        g5 = []
        for name in self.rigid_bodies:
            g5.append(
                TransferContactForceMohseni(dest=name,
                                            sources=self.rigid_bodies))

        stage2.append(Group(equations=g5, real=False))

        # computation of total force and torque at center of mass
        g6 = []
        for name in self.rigid_bodies:
            g6.append(SumUpExternalForces(dest=name,
                                          sources=None,
                                          gx=self.gx,
                                          gy=self.gy,
                                          gz=self.gz))

        stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        bodystep = GTVFRigidBody3DStep()
        integrator_cls = GTVFIntegrator

        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = bodystep

        cls = integrator_cls
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def setup_properties(self, particles, clean=True):
        from pysph.examples.solid_mech.impact import add_properties

        pas = dict([(p.name, p) for p in particles])

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

            add_properties(pa, 'rho_fsi', 'm_fsi', 'p_fsi')

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
                # velocity, of CM.
                'vcm':
                np.zeros(3 * nb, dtype=float),
                'vcm0':
                np.zeros(3 * nb, dtype=float),
                # acceleration of CM.
                'acm':
                np.zeros(3 * nb, dtype=float),
                # angular momentum
                'ang_mom':
                np.zeros(3 * nb, dtype=float),
                'ang_mom0':
                np.zeros(3 * nb, dtype=float),
                # angular velocity in global frame
                'omega':
                np.zeros(3 * nb, dtype=float),
                # angular acceleration in global frame
                'ang_acc':
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

            pa.set_output_arrays([
                'x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'normal',
                'is_boundary', 'fz', 'm', 'body_id', 'h'
            ])

        for boundary in self.boundaries:
            pa = pas[boundary]

            nb = 1
            consts = {
                'total_mass':
                np.zeros(nb, dtype=float),
                'xcm':
                np.zeros(3 * nb, dtype=float),
            }

            for key, elem in consts.items():
                pa.add_constant(key, elem)

            body_id = np.zeros(len(pa.x), dtype=int)
            pa.add_property('body_id', type='int', data=body_id)

            ####################################################
            # compute the boundary particles of the rigid body #
            ####################################################
            add_boundary_identification_properties(pa)
            # make sure your rho is not zero
            equations = get_boundary_identification_etvf_equations([pa.name],
                                                                   [pa.name])

            sph_eval = SPHEvaluator(arrays=[pa],
                                    equations=equations,
                                    dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim))

            sph_eval.evaluate(dt=0.1)

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
