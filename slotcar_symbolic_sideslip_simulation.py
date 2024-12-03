# -*- coding: future_fstrings -*-
#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosSim, AcadosSimSolver
import casadi_side_slip_model
import numpy as np
import matplotlib.pyplot as plt
import casadi
import phaseportrait

sim = AcadosSim()
sim.model, theta_no_slip, radius = casadi_side_slip_model.export_slot_car_ode_model()

f = casadi.Function('f', [sim.model.x, sim.model.u], [sim.model.f_expl_expr])

phi = 0



def sideSlip(side_slip, d_side_slip, *, phi_dot=2):
    u = d_side_slip
    #theta = side_slip - np.pi / 2 - theta_no_slip
    #theta_dot = d_side_slip + phi_dot / radius

    theta = side_slip - np.pi / 2
    theta_dot = d_side_slip


    x_dot = [theta, phi, theta_dot, phi_dot]

    values = f( x_dot, u)

    #print("hey")


    return float(values[0]), float(values[2])

 #Unfinished
def stability_angle(theta_0,  *, phi_dot_val=1):


    #theta = side_slip - np.pi / 2 - theta_no_slip
    #theta_dot = d_side_slip + phi_dot / radius

    theta_prev =  -np.pi / 2 + 0.5
    theta_dot_prev = 0

    d_theta = 0
    dd_theta = 0

    for i in range(10000):

        theta = theta_prev + 0.2*d_theta
        theta_dot = theta_dot_prev + 0.2*dd_theta


        x_dot = [theta, phi, theta_dot, phi_dot_val]

        values = f( x_dot, u)

        d_theta = float(values[0])
        dd_theta = float(values[2])

        theta_prev = theta
        theta_dot_prev = theta_dot

    #print("hey")

    return float(theta)

def theta_phase_plot():

    SimplePendulum = phaseportrait.PhasePortrait2D(sideSlip, [[-np.pi / 10 ,np.pi / 10], [-4, 4]], MeshDim=30, Density=1, Title='Theta phase plot for $ \dot \phi = 2$', xlabel=r"$\Theta$ [rad]",
                                                   ylabel=r"$\dot{\Theta}$ [rad/s]")
    #SimplePendulum.add_slider('phi_dot', valinit=1, valinterval=[0.1, 3], valstep=0.1)

    fig, ax = SimplePendulum.plot()

    plt.show()

    print("Wait for Sindre")


def stability_plot():

    Log2 = phaseportrait.Map1D(stability_angle, [-0.1, 2], [0, np.pi / 2], 2000, thermalization=50, Title='Logistic Map',
                 xlabel='Control parameter: "r"', ylabel=r'$x_{n+1}$', color='viridis')

    Log2.plot_over_variable('phi_dot_val', [0.1, 0.2], 0.05)

    fig, ax = Log2.plot()

    fig.ion()

    print("Wait for Sindre")


def main():

    theta_phase_plot()




if __name__ == "__main__":
    main()