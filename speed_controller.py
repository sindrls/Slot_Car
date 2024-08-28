import numpy as np


def speed_controller(l_x, l_y, phi, phi_dot, delta_t, horizon):
    V_max = 1
    K = 1
    p = 0.5

    d_phi_l_x = l_x.derivative()
    d_phi_l_y = l_y.derivative()

    dd_phi_l_x = d_phi_l_x.derivative()
    dd_phi_l_y = d_phi_l_y.derivative()

    exp_val = 0

    for i in range(horizon):
        phi_i = i * delta_t * phi_dot + phi
        d_phi_l_x_val = d_phi_l_x(phi_i)
        d_phi_l_y_val = d_phi_l_y(phi_i)

        dd_phi_l_x_val = dd_phi_l_x(phi_i)
        dd_phi_l_y_val = dd_phi_l_y(phi_i)

        kappa = (d_phi_l_x_val * dd_phi_l_y_val - dd_phi_l_x_val * d_phi_l_y_val)

        exp_val = exp_val + np.exp(p, i) * kappa

    return V_max * np.exp(- K * exp_val)


