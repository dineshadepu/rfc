"""[1] Particle-Based Numerical Simulation Study of Solid Particle
Erosion of Ductile Materials Leading to an Erosion Model,
Including the Particle Shape Effect

https://doi.org/10.3390/ma15010286


3.3.2 Controlled sliding on a flat surface

"""
from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body_3d import RigidBody3DScheme
from pysph.sph.equation import Equation, Group

from pysph.tools.geometry import get_2d_block


def setup_properties_for_gradual_force(pa):
    pa.add_constant('normal_force_time', 0.)
    pa.add_constant('tmp_normal_force', 0.)
    pa.add_constant('delta_fn', 0.)

    pa.add_constant('tangential_force_time', 0.)
    pa.add_constant('tmp_tangential_force', 0.)
    pa.add_constant('delta_ft', 0.)

    force_idx_fn = np.zeros_like(pa.x)
    force_idx_ft = np.zeros_like(pa.x)
    pa.add_property('force_idx_fn', type='int', data=force_idx_fn)
    pa.add_property('force_idx_ft', type='int', data=force_idx_ft)


def force_index_single(body):
    max_y = np.max(body.y)

    indices_fn = np.where(max_y == body.y)[0]
    body.force_idx_fn[indices_fn] = 1

    min_x = np.min(body.x)
    indices_ft = np.where(min_x == body.x)[0]
    body.force_idx_ft[indices_ft] = 1


class ApplyForceOnRigidBody(Equation):
    def initialize(self, d_idx,
                   d_m,
                   d_fx,
                   d_fy,
                   d_fz,
                   d_normal_force_time,
                   d_tmp_normal_force,
                   d_delta_fn,
                   d_force_idx_fn,
                   d_tangential_force_time,
                   d_tmp_tangential_force,
                   d_delta_ft,
                   d_force_idx_ft,
                   dt, t):
        t_1 = d_normal_force_time[0]

        if t <= t_1:
            if d_idx == 1:
                d_tmp_normal_force[0] += d_delta_fn[0]

        if d_force_idx_fn[d_idx] == 1:
            d_fy[d_idx] += d_tmp_normal_force[0]

        t_2 = d_normal_force_time[0] + d_tangential_force_time[0]
        if t > t_1 and t <= t_2:
            if d_idx == 1:
                d_tmp_tangential_force[0] += d_delta_ft[0]

        if d_force_idx_ft[d_idx] == 1:
            d_fx[d_idx] += d_tmp_tangential_force[0]


class Mohseni2021ControlledSlidingOnAFlatSurface(Application):
    def initialize(self):
        # Parameters specific to the world
        self.wall_inclination_angle = 30.
        self.dim = 2
        spacing = 1 * 1e-2
        self.hdx = 1.5
        self.alpha = 0.1
        self.gx = 0.
        self.gy = 0.
        self.h = self.hdx * spacing

        # Physical parameters of the rigid body
        self.rigid_body_length = 0.1
        self.rigid_body_height = 0.1
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        # Physical parameters of the wall (or tank) body
        self.wall_length = 20
        self.wall_height = 0.
        self.wall_spacing = spacing
        self.wall_layers = 0
        self.wall_rho = 2700.

        # solver data
        self.tf = 1.
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

        # =============================
        # force application parameters
        # =============================
        self.fn = -2. * 1e3
        self.ft = 4. * 1e3

        self.normal_force_time = 0.5
        self.tangential_force_time = 0.5

        timesteps_fn = self.normal_force_time / self.dt
        self.fn_increment = self.fn / timesteps_fn

        timesteps_ft = self.tangential_force_time / self.dt
        self.ft_increment = self.ft / timesteps_ft

    def get_boundary_particles(self, x, y):
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        # these are overridden for this example only.
        # This will change for another case
        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)
        pa = get_particle_array(name='pa',
                                x=x,
                                y=y,
                                h=h,
                                m=m,
                                rho=self.rigid_body_rho,
                                rad_s=rad_s,
                                E=69 * 1e9,
                                nu=0.3,
                                constants={
                                    'spacing0': self.rigid_body_spacing,
                                })

        add_boundary_identification_properties(pa)
        # make sure your rho is not zero
        equations = get_boundary_identification_etvf_equations([pa.name],
                                                               [pa.name])

        sph_eval = SPHEvaluator(arrays=[pa],
                                equations=equations,
                                dim=self.dim,
                                kernel=QuinticSpline(dim=self.dim))

        sph_eval.evaluate(dt=0.1)

        # set the boundary particles manually
        dx = self.rigid_body_spacing
        cond = (x < min(pa.x) + dx)
        pa.is_boundary[cond] = 1

        cond = (x > max(pa.x) - dx)
        pa.is_boundary[cond] = 1

        cond = (y < min(pa.y) + dx)
        pa.is_boundary[cond] = 1

        cond = (y > max(pa.y) - dx)
        pa.is_boundary[cond] = 1

        return pa.is_boundary

    def create_rigid_body(self):
        x = np.array([])
        y = np.array([])

        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        rigid_body = get_particle_array(name='rigid_body',
                                        x=x,
                                        y=y,
                                        h=h,
                                        m=m,
                                        rho=self.rigid_body_rho,
                                        rad_s=rad_s,
                                        E=69 * 1e9,
                                        nu=0.3,
                                        constants={
                                            'spacing0': self.rigid_body_spacing,
                                        })
        rigid_body.add_property('dem_id', type='int', data=dem_id)
        rigid_body.add_property('body_id', type='int', data=body_id)
        # The two here has to be hard coded per example
        rigid_body.add_constant('total_no_bodies', [2])
        return rigid_body

    def set_rigid_body_boundary(self, rigid_body):
        # Add the boundary particle information to the rigid body
        rigid_body.add_property('contact_force_is_boundary')
        is_boundary_one_body = self.get_boundary_particles(rigid_body.x,
                                                           rigid_body.y)
        is_boundary = np.tile(is_boundary_one_body, max(rigid_body.body_id)+1)
        is_boundary = is_boundary.ravel()
        rigid_body.contact_force_is_boundary[:] = is_boundary[:]
        rigid_body.add_property('is_boundary')
        rigid_body.is_boundary[:] = is_boundary[:]

    def create_wall(self):
        # Create wall particles
        length_fac = 1.
        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.wall_length * length_fac,
                            height=0.8 * self.wall_spacing)

        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.

        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=y,
                                  h=h,
                                  m=m,
                                  rho=self.rigid_body_rho,
                                  rad_s=rad_s,
                                  contact_force_is_boundary=1.,
                                  E=69 * 1e9,
                                  nu=0.3)
        max_dem_id = 0
        wall.add_property('dem_id', type='int', data=max_dem_id + 1)
        return wall

    def adjust_geometry(self, rigid_body, wall):
        rigid_body.y[:] += max(wall.y) - min(rigid_body.y)
        rigid_body.y[:] += self.rigid_body_spacing

        wall.x[:] += self.wall_length / 2. - self.rigid_body_length

    def create_particles(self):
        rigid_body = self.create_rigid_body()

        setup_properties_for_gradual_force(rigid_body)
        force_index_single(rigid_body)

        rigid_body.normal_force_time[0] = self.normal_force_time
        rigid_body.tmp_normal_force[0] = 0.
        rigid_body.delta_fn[0] = 0.

        rigid_body.tangential_force_time[0] = self.tangential_force_time
        rigid_body.tmp_tangential_force[0] = 0.

        # no of particles the tangential force to be applied
        no_par = len(np.where(rigid_body.force_idx_fn == 1.)[0])
        rigid_body.delta_fn[0] = self.fn_increment / no_par

        # no of particles the tangential force to be applied
        no_par = len(np.where(rigid_body.force_idx_ft == 1.)[0])
        rigid_body.delta_ft[0] = self.ft_increment / no_par

        wall = self.create_wall()

        self.adjust_geometry(rigid_body, wall)
        self.scheme.setup_properties([rigid_body, wall])

        # Set the boundary of the rigid body
        self.set_rigid_body_boundary(rigid_body)

        return [rigid_body, wall]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                 boundaries=['wall'],
                                 gx=0.,
                                 gy=self.gy,
                                 gz=0.,
                                 dim=2,
                                 fric_coeff=0.5)

        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def create_equations(self):
        eqns = self.scheme.get_equations()

        # Apply external force
        force_eqs = []
        force_eqs.append(
            ApplyForceOnRigidBody(dest="rigid_body", sources=None))

        eqns.groups[-1].insert(-2, Group(force_eqs))

        return eqns

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    # def post_step(self, solver):
    #     t = solver.t
    #     # dt = solver.dt
    #     # T = self.wall_time
    #     for pa in self.particles:
    #         if pa.name == 'rigid_body':
    #             t_1 = pa.normal_force_time[0]

    #             if t <= t_1:
    #                 pa.tmp_normal_force[0] += pa.delta_fn[0]
    #                 pa.fy[np.where(pa.force_idx_fn == 1)] += pa.tmp_normal_force[0]

    #             t_2 = pa.normal_force_time[0] + pa.tangential_force_time[0]
    #             if t > t_1 and t <= t_2:
    #                 pa.tmp_tangential_force[0] += pa.delta_ft[0]
    #                 pa.fx[np.where(pa.force_idx_ft == 1)] += pa.tmp_tangential_force[0]

    def post_process(self, fname):
        from pysph.solver.utils import iter_output, load
        import os
        from matplotlib import pyplot as plt

        info = self.read_info(fname)
        files = self.output_files

        data = load(files[0])
        arrays = data['arrays']
        t, v_x, v_y = [], [], []
        f_n, f_t = [], []

        for sd, rb in iter_output(files[::1], 'rigid_body'):
            no_frc_idx_fn = len(np.where(rb.force_idx_fn == 1)[0])
            no_frc_idx_ft = len(np.where(rb.force_idx_ft == 1)[0])

            _t = sd['t']
            t.append(_t)
            v_x.append(rb.vcm[0])
            v_y.append(rb.vcm[1])
            f_n.append(-rb.tmp_normal_force[0] * no_frc_idx_fn)
            f_t.append(rb.tmp_tangential_force[0] * no_frc_idx_ft)

        fig, ax1 = plt.subplots(figsize=(12, 10))

        ax2 = ax1.twinx()

        ax1.plot(t, f_n, label=r'$F_n$')
        ax1.plot(t, f_t, "-^", label=r'$F_t$')

        ax2.plot(t, v_x, color="black", label='V t')
        # ax2.plot(t, v_y, label='V n')

        ax1.set_xlabel('time')
        ax1.set_ylabel('Force (N)', color='g')
        ax1.legend()

        ax2.set_ylabel('Velocity', color='b')
        ax2.legend(loc="lower right")

        fig = os.path.join(os.path.dirname(fname), "force_velocity_vs_t.pdf")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = Mohseni2021ControlledSlidingOnAFlatSurface()
    app.run()
    app.post_process(app.info_filename)

# ft_x, ft_y, z
# contact_force_normal_x, contact_force_normal_y, contact_force_normal_z,
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
# ti_x, ti_y, z
