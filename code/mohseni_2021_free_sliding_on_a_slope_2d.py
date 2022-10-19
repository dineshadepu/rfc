"""[1] Particle-Based Numerical Simulation Study of Solid Particle
Erosion of Ductile Materials Leading to an Erosion Model,
Including the Particle Shape Effect

https://doi.org/10.3390/ma15010286


3.3.2 Free sliding on a slope

"""
from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body_3d import RigidBody3DScheme, get_files_at_given_times_from_log
from pysph.sph.equation import Equation, Group
import os

from pysph.tools.geometry import get_2d_block, get_2d_tank, rotate


class Mohseni2021FreeSlidingOnASlope(Application):
    def initialize(self):
        # Parameters specific to the world
        self.dim = 2
        spacing = 1 * 1e-2
        self.hdx = 1.5
        self.alpha = 0.1
        self.gx = 0.
        self.gy = - 9.81
        self.h = self.hdx * spacing

        # Physical parameters of the rigid body
        self.rigid_body_length = 0.1
        self.rigid_body_height = 0.1
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        # Physical parameters of the wall (or tank) body
        self.wall_length = 30
        self.wall_height = 0.
        self.wall_spacing = spacing
        self.wall_layers = 0
        self.wall_rho = 2000.

        # solver data
        self.tf = 3.
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
        angle = self.wall_inclination_angle
        rigid_body.y[:] += max(wall.y) - min(rigid_body.y) - self.rigid_body_spacing / 2.
        xc, yc, _zs = rotate(rigid_body.x, rigid_body.y, rigid_body.z, axis=np.array([0., 0., 1.]),
                             angle=-angle)
        x, y, _z = rotate(wall.x, wall.y, wall.z, axis=np.array([0., 0., 1.]), angle=-angle)

        wall.x[:] = x[:]
        wall.y[:] = y[:]

        rigid_body.x[:] = xc[:]
        rigid_body.y[:] = yc[:]
        radians = (90. - angle) * np.pi / 180.
        rigid_body.x[:] += (self.rigid_body_length / 2. + self.rigid_body_spacing) * np.cos(radians)

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
                                 gx=0.,
                                 gy=self.gy,
                                 gz=0.,
                                 dim=2,
                                 fric_coeff=0.45)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        tf = self.tf

        output_at_times = np.array([0., 0.5, 1.0])
        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=500,
                                     output_at_times=output_at_times)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import load, get_files

        info = self.read_info(fname)
        output_files = self.output_files

        from pysph.solver.utils import iter_output

        t, velocity = [], []

        for sd, rb in iter_output(output_files[::1], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            vel = (rb.vcm[0]**2. + rb.vcm[1]**2. + rb.vcm[2]**2.)**0.5
            velocity.append(vel)

        # analytical data
        theta = np.pi / 6.
        t_analytical = np.linspace(0., max(t), 100)
        v_analytical = (np.sin(theta) - self.options.fric_coeff * np.cos(theta)) * 9.81 * np.asarray(t_analytical)

        if self.options.fric_coeff > np.tan(theta):
            v_analytical = 0. * np.asarray(t_analytical)

        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 t=t,
                 velocity_rbd=velocity,

                 t_analytical=t_analytical,
                 v_analytical=v_analytical)

        plt.clf()
        plt.plot(t, velocity, "-", label='Mohsen')
        plt.plot(t_analytical, v_analytical, "--", label='Analytical')

        plt.title('Velocity')
        plt.xlabel('t')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "velocity_vs_time.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================
        # generate plots
        info = self.read_info(fname)
        output_files = self.output_files
        output_times = np.array([0., 5 * 1e-1, 1. * 1e-0])
        logfile = os.path.join(os.path.dirname(fname), 'mohseni_2021_free_sliding_on_a_slope_2d.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)

        for i, f in enumerate(to_plot):
            data = load(f)
            t = data['solver_data']['t']
            body = data['arrays']['rigid_body']
            wall = data['arrays']['wall']

            s = 20.
            fig, axs = plt.subplots(1, 1)
            axs.scatter(body.x, body.y, s=s, color="orangered")
            # axs.grid()
            axs.set_aspect('equal', 'box')
            # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

            # get the maximum and minimum of the geometry
            x_min = min(body.x) - self.rigid_body_height
            x_max = max(body.x) + 3. * self.rigid_body_height
            y_min = min(body.y) - 4. * self.rigid_body_height
            y_max = max(body.y) + 1. * self.rigid_body_height

            filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
                (wall.y >= y_min) & (wall.y <= y_max))
            wall_x = wall.x[filtr_1]
            wall_y = wall.y[filtr_1]
            wall_m = wall.m[filtr_1]

            tmp = axs.scatter(wall_x, wall_y, s=s, color="black")

            # save the figure
            figname = os.path.join(os.path.dirname(fname), "time" + str(i) + ".png")
            fig.savefig(figname, dpi=300)
            # plt.show()

        # =======================================
        # =======================================
        # Schematic
        # =======================================
        files = self.output_files
        for sd, body, wall in iter_output(files[0:2], 'rigid_body', 'wall'):
            _t = sd['t']
            if _t == 0.:
                s = 20.
                fig, axs = plt.subplots(1, 1)
                axs.scatter(body.x, body.y, s=s, color="orangered")
                # axs.grid()
                axs.set_aspect('equal', 'box')
                # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

                # im_ratio = tmp.shape[0]/tmp.shape[1]
                x_min = min(body.x) - self.rigid_body_height
                x_max = max(body.x) + 3. * self.rigid_body_height
                y_min = min(body.y) - 4. * self.rigid_body_height
                y_max = max(body.y) + 1. * self.rigid_body_height

                filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
                    (wall.y >= y_min) & (wall.y <= y_max))
                wall_x = wall.x[filtr_1]
                wall_y = wall.y[filtr_1]
                wall_m = wall.m[filtr_1]
                tmp = axs.scatter(wall_x, wall_y, s=s, color="black")
                axs.axis('off')
                axs.set_xticks([])
                axs.set_yticks([])

                # save the figure
                figname = os.path.join(os.path.dirname(fname), "pre_schematic.png")
                fig.savefig(figname, dpi=300)


if __name__ == '__main__':
    app = Mohseni2021FreeSlidingOnASlope()
    app.run()
    app.post_process(app.info_filename)

# ft_x, ft_y, z
# fn_x, fn_y, z
# u, v, w
# delta_lt_x, delta_lt_y, delta_lt_z
