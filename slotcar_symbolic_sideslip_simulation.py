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


def pendulum(theta, dTheta):
    sim = AcadosSim()
    sim.model, theta_no_slip = casadi_side_slip_model.export_slot_car_ode_model()

    jacobian = casadi.jacobian(sim.model.f_expl_expr, sim.model.x)

    phi = 0
    phi_dot = 1

    theta_0 = - np.pi / 2 - theta_no_slip
    return dθ, - numpy.sin(θ)

def main():






    SimplePendulum = phaseportrait.PhasePortrait2D(pendulum, [-9, 9], Title='Simple pendulum', xlabel=r"$\Theta$",
                                                   ylabel=r"$\dot{\Theta}$")
    SimplePendulum.plot()





if __name__ == "__main__":
    main()