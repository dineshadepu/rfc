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


class CubeFallingInWater(Application):
    def initialize(self):
        # Parameters specific to the world
        self.dim = 2
        spacing = 0.03
        self.hdx = 1.0
        self.alpha = 0.1
        self.gx = 0.
        self.gy = - 9.81
        self.h = self.hdx * spacing

        # Fluid parameters
        self.fluid_column_height = 1.0
        self.fluid_column_width = 1.5
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
        self.rigid_body_radius = 0.2
        self.rigid_body_diameter = 2. * self.rigid_body_radius
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 500

        # Physical parameters of the wall (or tank) body
        self.dam_length = 1.5
        self.dam_height = 1.5
        self.dam_spacing = spacing
        self.dam_layers = 5
        self.dam_rho = 1000.

        # solver data
        self.tf = 0.5
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
            self.rigid_body_diameter, self.rigid_body_spacing, [
                self.rigid_body_radius,
                self.rigid_body_radius + self.rigid_body_spacing / 2.
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
        from geometry import create_circle_1

        x, y = create_circle_1(self.rigid_body_diameter, self.rigid_body_spacing)

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
            self.dam_length, self.dam_height, self.dam_height, self.dam_layers,
            self.rigid_body_spacing, self.rigid_body_spacing)
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

    def adjust_geometry(self, rigid_body, dam, fluid):
        dam.x += 2.0
        fluid.x += min(dam.x) - min(fluid.x) + (self.dam_layers - 1) * self.fluid_spacing
        fluid.x += self.fluid_spacing
        fluid.y += self.fluid_spacing
        fluid.y -= min(fluid.y) - min(dam.y) - self.dam_layers * self.fluid_spacing

        rigid_body.y[:] += max(fluid.y) - min(rigid_body.y) - self.rigid_body_diameter/2.
        rigid_body.y[:] += self.rigid_body_diameter / 1.5
        # rigid_body.x[:] -= self.rigid_body_length
        rigid_body.x[:] -= min(rigid_body.x) - min(fluid.x)
        rigid_body.x[:] += self.fluid_column_width / 2. - self.rigid_body_diameter / 2.

        # for floating case
        rigid_body.y[:] -= self.rigid_body_diameter * 2.
        rigid_body.y[:] += self.fluid_spacing / 2.

        # Remove the fluid particles
        from pysph.tools import geometry as G
        G.remove_overlap_particles(
            fluid, rigid_body, self.fluid_spacing, dim=self.dim
        )

    def create_particles(self):
        fluid = self.create_fluid()
        rigid_body = self.create_rigid_body()
        dam = self.create_dam()

        self.adjust_geometry(rigid_body, dam, fluid)
        self.scheme.setup_properties([rigid_body, dam, fluid])

        # Set the boundary of the rigid body
        self.set_rigid_body_boundary(rigid_body)
        self.set_fluid_coupling_properties_for_rigid_body(rigid_body)
        # self.set_dam_boundary(dam)

        return [rigid_body, dam, fluid]

    def create_scheme(self):
        etvf = ETVFScheme(
            fluids=['fluid'], solids=['dam'], rigid_bodies=['rigid_body'],
            dim=2, rho0=self.fluid_rho, c0=self.c0, nu=None, pb=self.p0,
            h=None, u_max=3. * self.vref, mach_no=self.mach_no,
            internal_flow=False, gy=self.gy, alpha=0.05)

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


if __name__ == '__main__':
    app = CubeFallingInWater()
    # app.create_particles()
    # app.geometry()
    app.run()
    app.post_process(app.info_filename)
