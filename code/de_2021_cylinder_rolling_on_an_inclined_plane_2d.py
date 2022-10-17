from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body_3d import RigidBody3DScheme
from pysph.sph.equation import Equation, Group

from pysph.tools.geometry import rotate, get_2d_block
from geometry import create_circle_1


class De2021CylinderRollingOnAnInclinedPlane2d(Application):
    def initialize(self):
        # Parameters specific to the world
        self.wall_inclination_angle = 60.
        self.dim = 2
        spacing = 25 * 1e-3
        self.hdx = 1.5
        self.alpha = 0.1
        self.gx = 9.81 * np.sin(self.wall_inclination_angle * np.pi / 180)
        self.gy = - 9.81 * np.cos(self.wall_inclination_angle * np.pi / 180)
        self.h = self.hdx * spacing

        # Physical parameters of the rigid body
        self.rigid_body_radius = 0.5
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        # Physical parameters of the wall (or tank) body
        self.wall_length = 40.
        self.wall_height = 0.
        self.wall_spacing = spacing
        self.wall_layers = 0
        self.wall_rho = 2700.

        # solver data
        self.tf = 0.6
        self.dt = 1e-4

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def get_boundary_particles(self, x, y):
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
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

        return pa.is_boundary

    def create_rigid_body(self):
        x = np.array([])
        y = np.array([])

        x, y = create_circle_1(diameter=2. * self.rigid_body_radius,
                               spacing=self.rigid_body_spacing,
                               center=None)

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
        rigid_body.y[:] += max(wall.y) - min(rigid_body.y) + self.rigid_body_spacing
        # wall.x[:] = x[:]
        # wall.y[:] = y[:]

    def create_particles(self):
        rigid_body = self.create_rigid_body()
        wall = self.create_wall()

        self.adjust_geometry(rigid_body, wall)
        self.scheme.setup_properties([rigid_body, wall])

        # Set the boundary of the rigid body
        self.set_rigid_body_boundary(rigid_body)

        return [rigid_body, wall]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                 boundaries=['wall'],
                                 gx=self.gx,
                                 gy=self.gy,
                                 gz=0.,
                                 dim=2,
                                 fric_coeff=0.5)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output
        import os

        info = self.read_info(fname)
        output_files = self.output_files

        data = load(output_files[0])
        arrays = data['arrays']
        rb = arrays['rigid_body']
        x0 = rb.xcm[0]

        t, x_com = [], []

        for sd, rb in iter_output(output_files[::1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            x_com.append(rb.xcm[0] - x0)

        # analytical data
        theta = np.pi / 3.
        t_analytical = np.linspace(0., max(t), 100)

        if self.options.fric_coeff == 0.3:
            x_analytical = 0.5 * 9.81 * t_analytical**2 * (np.sin(theta) - 0.3 * np.cos(theta))
        elif self.options.fric_coeff == 0.6:
            x_analytical = 1. / 3. * 9.81 * t_analytical**2. * np.sin(theta)

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 t=t,
                 x_com=x_com,

                 t_analytical=t_analytical,
                 x_analytical=x_analytical)

        plt.clf()
        plt.plot(t, x_com, "^-", label='Simulated')
        plt.plot(t_analytical, x_analytical, "--", label='Analytical')

        plt.title('x-center of mass')
        plt.xlabel('t')
        plt.ylabel('x-center of mass (m)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "xcom_vs_time.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = De2021CylinderRollingOnAnInclinedPlane2d()
    app.run()
    app.post_process(app.info_filename)
