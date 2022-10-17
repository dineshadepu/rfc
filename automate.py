#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
from automan.api import mdict, opts2path

import numpy as np
import matplotlib
from pysph.solver.utils import load, get_files

matplotlib.use('pdf')

n_core = 4
n_thread = 8
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


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


class De2021CylinderRollingOnAnInclinedPlane2d(Problem):
    """
    For pure rigid body problems we use RigidBody3DScheme.
    Scheme used: RigidBody3DScheme
    """
    def get_name(self):
        return 'de_2021_cylinder_rolling_on_an_inclined_plane_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/de_2021_cylinder_rolling_on_an_inclined_plane_2d.py' + backend

        opts = mdict(fric_coeff=[0.3, 0.6])

        self.cases = []
        self.case_info = {}
        for kw in opts:
            name = opts2path(kw)
            name = name.replace(".", "_")
            self.cases.append(
                Simulation(get_path(name), cmd,
                           job_info=dict(n_core=n_core,
                                         n_thread=n_thread), cache_nnps=None,
                           scheme='rb3d', pfreq=300, kr=1e7, kf=1e5, tf=0.6,
                           **kw))
            self.case_info.update({name: rf"$\mu=${kw['fric_coeff']}"})

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            x_analytical = data[name]['x_analytical']

            t = data[name]['t']
            x_com = data[name]['x_com']

            plt.plot(t_analytical, x_analytical, "-", label=self.case_info[name] + ' analytical')
            plt.plot(t, x_com, "^", label=self.case_info[name])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('xcom_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('pdf')

    PROBLEMS = [
        # ========================
        # Only rigid body problems
        # ========================
        # Current paper problem
        De2021CylinderRollingOnAnInclinedPlane2d  # DEM
    ]

    automator = Automator(simulation_dir='outputs',
                          output_dir=os.path.join('manuscript', 'figures'),
                          all_problems=PROBLEMS)

    automator.run()
