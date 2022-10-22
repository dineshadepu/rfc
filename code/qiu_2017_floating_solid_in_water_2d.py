from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

# from rigid_fluid_coupling import RigidFluidCouplingScheme
from rigid_body_3d import RigidBody3DScheme
# from rigid_body_common import setup_damping_coefficient

from geometry import (get_fluid_tank_3d, get_2d_block, hydrostatic_tank_2d)
from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from fluids import ETVFScheme


class Qiu2017FallingSolidInWater2D(Application):
    def initialize(self):
        # Parameters specific to the world
        self.dim = 2
        spacing = 2. * 1e-3
        self.hdx = 1.0
        self.alpha = 0.1
        self.gx = 0.
        self.gy = - 9.81
        self.gz = 0.
        self.h = self.hdx * spacing

        # Fluid parameters
        self.fluid_column_length = 140. * 1e-3
        self.fluid_column_height = 52. * 1e-3
        # We do not use this in 2 dimensions
        self.fluid_column_depth = 140. * 1e-3
        self.fluid_length = 140. * 1e-3
        self.fluid_height = 52. * 1e-3
        # We do not use this in 2 dimensions
        self.fluid_depth = 140. * 1e-3
        self.fluid_spacing = spacing
        self.fluid_rho = 1000.0
        self.vref = np.sqrt(2*abs(self.gy)*self.fluid_column_height)
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.p0 = self.fluid_rho * self.c0**2
        self.boundary_equations = get_boundary_identification_etvf_equations(
            destinations=["fluid"],
            sources=["fluid", "dam", "rigid_body"],
            boundaries=["dam", "rigid_body"])

        # Physical parameters of the rigid body
        # x dimension
        self.rigid_body_length = 49. * 1e-3
        # y dimension
        self.rigid_body_height = 24. * 1e-3
        # z dimension
        self.rigid_body_depth = 48. * 1e-3
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 800.52

        # Physical parameters of the wall (or tank) body
        self.dam_length = 140. * 1e-3
        self.dam_height = 140 * 1e-3
        self.dam_depth = 140 * 1e-3  # ignored in 3d
        self.dam_spacing = spacing
        self.dam_layers = 5
        self.dam_rho = 1000.
        self.tank_length = 140. * 1e-3
        self.tank_height = 140 * 1e-3
        self.tank_depth = 140 * 1e-3
        self.tank_spacing = spacing
        self.tank_layers = 5
        self.tank_rho = 1000.

        # solver data
        self.tf = 1.5
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
        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
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

        return pa.is_boundary

    def create_rigid_body(self):
        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**self.dim
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
        rigid_body.add_constant('max_tng_contacts_limit', 3)
        rigid_body.add_constant('total_no_bodies', 2)
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
            fluid_length=self.fluid_length,
            fluid_height=self.fluid_height,
            tank_height=self.tank_height,
            tank_layers=self.tank_layers,
            fluid_spacing=self.fluid_spacing,
            tank_spacing=self.tank_spacing)

        m = self.fluid_rho * self.rigid_body_spacing**self.dim
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
        dam.add_property('dem_id', type='int', data=1)
        return dam

    def set_dam_boundary(self, dam):
        dam.add_property('contact_force_is_boundary')
        dam.contact_force_is_boundary[:] = dam.is_boundary[:]

    def create_fluid(self):
        # create dam with normals
        xf, yf, _xd, _yd = hydrostatic_tank_2d(
            fluid_length=self.fluid_length,
            fluid_height=self.fluid_height,
            tank_height=self.tank_height,
            tank_layers=self.tank_layers,
            fluid_spacing=self.fluid_spacing,
            tank_spacing=self.tank_spacing)

        h = self.hdx * self.fluid_spacing
        self.h = h
        m = self.fluid_spacing**self.dim * self.fluid_rho
        h = self.hdx * self.fluid_spacing
        fluid = get_particle_array(name='fluid', x=xf, y=yf,
                                   h=h, m=m,
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

    def adjust_geometry(self, rigid_body, dam, fluid):
        # Place the rigid body at right position
        rigid_body.y[:] -= min(rigid_body.y) - min(fluid.y)
        rigid_body.y[:] += self.fluid_spacing * 3.
        # rigid_body.y[:] -= self.rigid_body_height / 2.

        # Remove the fluid particles
        from pysph.tools import geometry as G
        G.remove_overlap_particles(
            fluid, rigid_body, self.fluid_spacing, dim=self.dim
        )
        # move the whole system so that fluids min y is at zero
        dx = min(fluid.y)
        fluid.y -= dx
        dam.y -= dx
        rigid_body.y -= dx

    def create_particles(self):
        fluid = self.create_fluid()
        rigid_body = self.create_rigid_body()
        dam = self.create_dam()

        self.adjust_geometry(rigid_body, dam, fluid)
        self.scheme.setup_properties([rigid_body, dam, fluid])

        # Set the boundary of the rigid body
        self.set_rigid_body_boundary(rigid_body)
        self.set_fluid_coupling_properties_for_rigid_body(rigid_body)
        self.set_dam_boundary(dam)

        return [rigid_body, dam, fluid]

    def create_scheme(self):
        etvf = ETVFScheme(
            fluids=['fluid'], solids=['dam'], rigid_bodies=['rigid_body'],
            dim=self.dim, rho0=self.fluid_rho, c0=self.c0, nu=None, pb=self.p0,
            h=None, u_max=3. * self.vref, mach_no=self.mach_no,
            internal_flow=False, gy=self.gy, alpha=0.1)

        s = SchemeChooser(default='etvf', etvf=etvf)
        return s

    def configure_scheme(self):
        super().configure_scheme()

        # dt = 0.125*self.h/(co + vref)
        dt = 1e-4
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
        from pysph.solver.utils import iter_output, get_files
        import os
        info = self.read_info(fname)
        files = self.output_files
        t = []
        system_x = []
        system_y = []
        for sd, array in iter_output(files[::10], 'rigid_body'):
            _t = sd['t']
            t.append(_t)
            cm_x = array.xcm[0]
            cm_y = array.xcm[1]

            system_x.append(cm_x)
            system_y.append(cm_y)

        import matplotlib.pyplot as plt
        t = np.asarray(t)

        # gtvf data
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)
        data = np.loadtxt(
            os.path.join(
                directory,
                'qiu_2017_falling_solid_in_water_vertical_displacement_experimental.csv'
            ),
            delimiter=','
        )
        ty, ycom_qiu = data[:, 0], data[:, 1]

        plt.plot(ty, ycom_qiu, "s--", label='Experimental')
        plt.plot(t, system_y, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "ycom.pdf")
        plt.savefig(fig, dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['cylinders']
        b.plot.actor.property.point_size = 2.
        ''')


if __name__ == '__main__':
    app = Qiu2017FallingSolidInWater2D()
    app.run()
    app.post_process(app.info_filename)
