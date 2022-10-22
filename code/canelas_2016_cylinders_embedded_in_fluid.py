"""Simulation of solid-fluid mixture flow using moving particle methods
Shuai Zhang

TODO: 1. Fix the dam such that the bottom layer is y - spacing/2.
TODO: 2. Implement a simple 2d variant of rigid body collision.
"""

from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

# from rigid_fluid_coupling import RigidFluidCouplingScheme
from rigid_body_3d import RigidBody3DScheme
# from rigid_body_common import setup_damping_coefficient

from pysph.tools.geometry import get_2d_block, get_2d_tank

from geometry import (create_circle_1, create_circle, hydrostatic_tank_2d)
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from fluids import ETVFScheme


class Zhang2009CylindersEmbeddedInFluid(Application):
    def initialize(self):
        # Parameters specific to the world
        self.dim = 2
        spacing = 1 * 1e-3
        self.hdx = 1.5
        self.alpha = 0.1
        self.gx = 0.
        self.gy = - 9.81
        self.h = self.hdx * spacing

        # Physical parameters of the rigid body
        self.rigid_body_radius = 1. / 2. * 1e-2
        self.rigid_body_diameter = 1. * 1e-2
        self.cylinder_radius = 1. / 2. * 1e-2
        self.cylinder_diameter = 1. * 1e-2
        self.rigid_body_spacing = spacing
        self.cylinder_spacing = spacing
        self.rigid_body_rho = 2700
        self.cylinder_rho = 2700

        # Physical parameters of the wall (or tank) body
        self.dam_length = 26 * 1e-2
        self.dam_height = 26 * 1e-2
        self.dam_spacing = spacing
        self.dam_layers = 5
        self.dam_rho = 2000.

        self.wall_height = 14 * 1e-2
        self.wall_spacing = spacing
        self.wall_layers = 2
        self.wall_time = 0.2
        self.wall_rho = 2700

        # Fluid parameters
        self.fluid_column_height = 12 * 1e-2
        self.fluid_column_width = 7. * 1e-2
        self.fluid_spacing = spacing
        self.fluid_rho = 1000.0
        self.vref = np.sqrt(2*abs(self.gy)*self.fluid_column_height)
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.p0 = self.fluid_rho * self.c0**2
        self.boundary_equations = get_boundary_identification_etvf_equations(
            destinations=["fluid"],
            sources=["fluid", "dam", "cylinders"],
            boundaries=["dam", "cylinders"])

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
        # these are overridden for this example only.
        # This will change for another case
        x, y = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing / 2.
            ])

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

        x, y, body_id = self.create_cylinders_stack_1()

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        rigid_body = get_particle_array(name='cylinders',
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
        rigid_body.add_constant('total_no_bodies', [35])
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

    def create_dam(self):
        # create dam with normals
        _xf, _yf, xd, yd = hydrostatic_tank_2d(
            self.dam_length, self.dam_height, self.dam_height, self.dam_layers,
            self.cylinder_spacing, self.cylinder_spacing)
        m = self.fluid_rho * self.rigid_body_spacing**2
        h = self.h
        dam = get_particle_array(x=xd,
                                 y=yd,
                                 rho=self.fluid_rho,
                                 h=h,
                                 m=m,
                                 rad_s=self.dam_spacing / 2.,
                                 name="dam",
                                 E=30*1e8,
                                 nu=0.3)
        dam.add_property('dem_id', type='int', data=33)
        return dam

    def set_dam_boundary(self, dam):
        dam.add_property('contact_force_is_boundary')
        dam.contact_force_is_boundary[:] = dam.is_boundary[:]

        # remove particles which are not used in computation
        indices = []
        for i in range(len(dam.x)):
            if dam.is_boundary[i] == 0:
                indices.append(i)

        # dam.remove_particles(indices)

        # remove particles which are not used in computation
        min_x = min(dam.x)
        max_x = max(dam.x)
        min_y = min(dam.y)
        indices = []
        for i in range(len(dam.x)):
            if dam.x[i] < min_x + self.cylinder_spacing/2.:
                indices.append(i)

            if dam.y[i] < min_y + self.cylinder_spacing/2.:
                indices.append(i)

            if dam.x[i] > max_x - self.cylinder_spacing/2.:
                indices.append(i)

        # dam.remove_particles(indices)

    def create_wall(self):
        # create wall with normals
        xw, yw = get_2d_block(
            self.wall_spacing,
            self.dam_layers * self.cylinder_spacing,
            self.wall_height
        )
        m = self.fluid_rho * self.rigid_body_spacing**2
        h = self.h
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  rho=self.fluid_rho,
                                  h=h,
                                  m=m,
                                  rad_s=self.cylinder_spacing / 2.,
                                  name="wall",
                                  E=30*1e8,
                                  nu=0.3)

        wall.add_property('dem_id', type='int', data=34)

        wall.add_property('contact_force_is_boundary')
        wall.contact_force_is_boundary[:] = 1
        return wall

    def create_fluid(self):
        xf, yf = get_2d_block(dx=self.fluid_spacing,
                              length=self.fluid_column_width,
                              height=self.fluid_column_height,
                              center=[-1.5, 1])

        h = self.hdx * self.fluid_spacing
        self.h = h
        m = self.fluid_spacing**2 * self.fluid_rho
        h = self.hdx * self.fluid_spacing
        fluid = get_particle_array(name='fluid', x=xf, y=yf, h=h, m=m,
                                   rho=self.fluid_rho)

        if self.options.pst == "sun2019":
            fluid.add_constant('wdeltap', 0.)
            fluid.add_constant('n', 4.)
            kernel = self.scheme.scheme.kernel(dim=2)
            wdeltap = kernel.kernel(rij=self.fluid_spacing, h=self.h)
            fluid.wdeltap[0] = wdeltap
            fluid.n[0] = 4

        return fluid

    def set_fluid_coupling_properties_for_rigid_body(self, rigid_body):
        rigid_body.m_fsi[:] = self.fluid_rho * self.rigid_body_spacing**self.dim
        rigid_body.rho_fsi[:] = 1000.

    def adjust_geometry(self, cylinders, dam, wall, fluid):
        dam.x += min(cylinders.x) - min(dam.x) - self.dam_spacing * self.dam_layers

        fluid.x += min(dam.x) - min(fluid.x)
        fluid.y -= min(fluid.y) - min(dam.y)
        fluid.x += self.dam_layers * self.fluid_spacing
        fluid.y += self.dam_layers * self.fluid_spacing

        wall.x += max(fluid.x) - min(wall.x) + self.cylinder_spacing * 1.
        wall.y += min(dam.y) - min(wall.y) + self.dam_layers * self.dam_spacing

        cylinders.x += self.fluid_spacing

        # Remove the fluid particles
        from pysph.tools import geometry as G
        G.remove_overlap_particles(
            fluid, cylinders, self.fluid_spacing, dim=self.dim
        )

    def create_particles(self):
        fluid = self.create_fluid()
        cylinders = self.create_rigid_body()
        dam = self.create_dam()
        wall = self.create_wall()

        self.adjust_geometry(cylinders, dam, wall, fluid)
        self.scheme.setup_properties([cylinders, dam, wall, fluid])

        # Set the boundary of the rigid body
        self.set_rigid_body_boundary(cylinders)
        self.set_dam_boundary(dam)
        self.set_fluid_coupling_properties_for_rigid_body(cylinders)

        return [cylinders, dam, wall, fluid]

    def create_scheme(self):
        etvf = ETVFScheme(
            fluids=['fluid'], solids=['dam', 'wall'], rigid_bodies=['cylinders'],
            dim=2, rho0=self.fluid_rho, c0=self.c0, nu=None, pb=self.p0,
            h=None, u_max=3. * self.vref, mach_no=self.mach_no,
            internal_flow=False, gy=self.gy, alpha=0.05)

        s = SchemeChooser(default='etvf', etvf=etvf)
        return s

    def create_cylinders_stack_1(self):
        # create a row of six cylinders
        x_six_1 = np.array([])
        y_six_1 = np.array([])
        x_tmp1, y_tmp1 = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                self.cylinder_radius,
                self.cylinder_radius + self.cylinder_spacing / 2.
            ])
        x_tmp = x_tmp1
        for i in range(6):
            x_tmp = x_tmp + self.cylinder_diameter + self.cylinder_spacing * 1.5
            x_six_1 = np.concatenate((x_six_1, x_tmp))
            y_six_1 = np.concatenate((y_six_1, y_tmp1))

        # create a row of five cylinders
        x_five_1 = np.array([])
        y_five_1 = np.array([])
        x_tmp1, y_tmp1 = create_circle_1(
            self.cylinder_diameter, self.cylinder_spacing, [
                2. * self.cylinder_radius, self.cylinder_radius +
                self.cylinder_spacing + 2. * self.cylinder_spacing
            ])

        x_tmp = x_tmp1
        for i in range(5):
            x_tmp = x_tmp + self.cylinder_diameter + self.cylinder_spacing * 1.5
            x_five_1 = np.concatenate((x_five_1, x_tmp))
            y_five_1 = np.concatenate((y_five_1, y_tmp1))
        # ============================================

        y_five_1 = y_five_1 + .9 * self.cylinder_diameter
        x_five_1 = x_five_1 - self.cylinder_spacing/2.

        # Create the third row from bottom, six cylinders
        x_six_2 = np.array(x_six_1, copy=True)
        y_six_2 = np.array(y_six_1, copy=True)
        y_six_2 += np.max(y_five_1) - np.min(y_six_1) + self.cylinder_spacing * 2.5

        # Create the fourth row from bottom, five cylinders
        x_five_2 = np.array(x_five_1, copy=True)
        y_five_2 = np.array(y_five_1, copy=True)
        y_five_2 += np.max(y_six_2) - np.min(y_five_2) + self.cylinder_spacing * 2.5

        # Create the third row from bottom, six cylinders
        x_six_3 = np.array(x_six_2, copy=True)
        y_six_3 = np.array(y_six_2, copy=True)
        y_six_3 += np.max(y_five_2) - np.min(y_six_3) + self.cylinder_spacing * 2.5

        # Create the fourth row from bottom, five cylinders
        x_five_3 = np.array(x_five_2, copy=True)
        y_five_3 = np.array(y_five_2, copy=True)
        y_five_3 += np.max(y_six_3) - np.min(y_five_2) + self.cylinder_spacing * 2.5

        x = np.concatenate((x_six_1, x_five_1, x_six_2, x_five_2,
                            x_six_3, x_five_3))
        y = np.concatenate((y_six_1, y_five_1, y_six_2, y_five_2,
                            y_six_3, y_five_3))

        # create body_id
        no_particles_one_cylinder = len(x_tmp)
        total_bodies = 3 * 5 + 3 * 6

        body_id = np.array([], dtype=int)
        for i in range(total_bodies):
            b_id = np.ones(no_particles_one_cylinder, dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        return x, y, body_id

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        if t > 0.1 and t < 0.2:
            for pa in self.particles:
                if pa.name == 'wall':
                    pa.y += 2. * dt

    def configure_scheme(self):
        super().configure_scheme()

        # dt = 0.125*self.h/(co + vref)
        dt = self.dt
        h0 = self.hdx * self.fluid_spacing
        scheme = self.scheme
        if self.options.scheme == 'etvf':
            scheme.configure(pb=self.p0, nu=self.nu, h=h0)

            times = [0.4, 0.6, 0.8]
            self.scheme.configure_solver(dt=dt, tf=self.tf, output_at_times=times)

    def _make_accel_eval(self, equations, pa_arrays):
        from pysph.base.kernels import (QuinticSpline)
        from pysph.tools.sph_evaluator import SPHEvaluator
        if self.seval is None:
            kernel = QuinticSpline(dim=self.dim)
            seval = SPHEvaluator(arrays=pa_arrays, equations=equations,
                                 dim=self.dim, kernel=kernel)
            self.seval = seval
            return self.seval
        else:
            self.seval.update()
            return self.seval
        return seval

    def pre_step(self, solver):
        if solver.count % 1 == 0:
            t = solver.t
            dt = solver.dt

            arrays = self.particles
            a_eval = self._make_accel_eval(self.boundary_equations, arrays)

            # When
            a_eval.evaluate(t, dt)

    def post_process(self, fname):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.
        By running the example it becomes much clear.
        """
        from pysph.solver.utils import iter_output, get_files
        import os
        info = self.read_info(fname)
        files = self.output_files
        t = []
        system_x = []
        system_y = []
        for sd, array in iter_output(files[::10], 'cylinders'):
            _t = sd['t']
            t.append(_t)
            # get the system center
            cm_x = 0
            cm_y = 0
            for i in range(array.nb[0]):
                cm_x += array.xcm[3 * i]
                cm_y += array.xcm[3 * i + 1]
            cm_x = cm_x / 33
            cm_y = cm_y / 33

            system_x.append(cm_x / self.dam_length)
            system_y.append(cm_y / self.dam_length)

        import matplotlib.pyplot as plt
        t = np.asarray(t)
        t = t - self.wall_time

        # gtvf data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        data = np.loadtxt(os.path.join(directory, 'x_com_zhang.csv'),
                          delimiter=',')
        tx, xcom_zhang = data[:, 0], data[:, 1]

        plt.plot(tx, xcom_zhang, "s--", label='Experimental')
        plt.plot(t, system_x, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("x/L")
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "xcom.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        data = np.loadtxt(os.path.join(directory, 'y_com_zhang.csv'),
                          delimiter=',')
        ty, ycom_zhang = data[:, 0], data[:, 1]

        plt.plot(ty, ycom_zhang, "s--", label='Experimental')
        plt.plot(t, system_y, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("y/L")
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "ycom.png")
        plt.savefig(fig, dpi=300)

        # generate plots
        info = self.read_info(fname)
        output_files = self.output_files
        output_times = np.array([0.2, 0.3, 0.5, 0.7])

        count = 0
        for sd, body, wall in iter_output(output_files, 'cylinders', 'dam'):
            _t = sd['t']
            if count >= len(output_times):
                break
            if abs(_t - output_times[count]) < _t * 1e-8:
                print(_t)
                s = 0.2
                # print(_t)
                fig, axs = plt.subplots(1, 1)
                axs.scatter(body.x, body.y, s=s, c=body.m)
                # axs.grid()
                axs.set_aspect('equal', 'box')
                # axs.set_title('still a circle, auto-adjusted data limits', fontsize=10)

                tmp = axs.scatter(wall.x, wall.y, s=s, c=wall.m)
                # save the figure
                figname = os.path.join(os.path.dirname(fname), "time" + str(count) + ".png")
                fig.savefig(figname, dpi=300)
                # plt.show()
                count += 1

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['cylinders']
        b.plot.actor.property.point_size = 2.
        ''')


if __name__ == '__main__':
    app = Zhang2009CylindersEmbeddedInFluid()
    # app.create_particles()
    # app.geometry()
    app.run()
    app.post_process(app.info_filename)
