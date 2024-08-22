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
import acados_slotcar_model
import numpy as np
import matplotlib.pyplot as plt
import casadi


def main():

    sim = AcadosSim()
    sim.model = acados_slotcar_model.export_slot_car_ode_model()

    jacobian = casadi.jacobian(sim.model.f_expl_expr, sim.model.x)

    #print("Jacobian is:", jacobian)

    Tf = 0.01
    nx = sim.model.x.rows()
    N_sim = 1000


    # set simulation time
    sim.solver_options.T = Tf
    # set options
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 3
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 3 # for implicit integrator
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

    # create
    acados_integrator = AcadosSimSolver(sim)

    x0 = np.array([0.0, 0.0, 0.0, 1.0])
    u0 = np.array([0.1])
    xdot_init = np.zeros((nx,))

    simX = np.zeros((nx, N_sim+1))
    simX[:,0] = x0
    print("X_i: ", simX[:, 0])
    for i in range(N_sim):
        # Note that xdot is only used if an IRK integrator is used
        simX[:, i+1] = acados_integrator.simulate(x=simX[:, i], u=u0, xdot=xdot_init)
        #print("X_i: ", simX[i,:])

    S_forw = acados_integrator.get("S_forw")
    print("S_forw, sensitivities of simulation result wrt x,u:\n", S_forw)

    plt.plot(simX[0, :], label='angle(t)')
    plt.plot(simX[1, :], label='distance(t)')
    plt.plot(simX[2, :], label='ang_vel(t)')
    plt.plot(simX[3, :], label='vel(t)')
    plt.legend(loc='best')
    plt.show()

    #print(simX[1, :])





if __name__ == "__main__":
    main()