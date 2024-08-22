# Generating cubic spline for vehicle path
import casadi
import numpy as np
import scipy
import scipy.interpolate
import casadi as ca

import matplotlib.pyplot as plt


def get_interp_path():

    turn_radius = 0.3

    l_x_vals = np.asarray([0, -turn_radius, -1.5 * turn_radius, - 2.5 * turn_radius, - 1.5 * turn_radius, -1 * turn_radius, 1.5 * turn_radius, 2 * turn_radius, 3 * turn_radius, 2 * turn_radius, 1 * turn_radius, 2 * turn_radius, 2.5 * turn_radius,
                            3 * turn_radius, 3.5 * turn_radius, 4.5 * turn_radius, 3.5 * turn_radius, 3 * turn_radius, 0])
    l_y_vals = np.asarray([turn_radius, turn_radius, turn_radius, 0, -turn_radius, -turn_radius, -turn_radius, -turn_radius, - 2 * turn_radius, -3 * turn_radius, -2 * turn_radius, - turn_radius, -turn_radius, -turn_radius, -turn_radius, 0, turn_radius, turn_radius, turn_radius])

    phi_vals = np.asarray([0, turn_radius, 1.5 * turn_radius, 1.5 * turn_radius + (np.pi / 2) * turn_radius, 1.5 * turn_radius + np.pi * turn_radius, 2 * turn_radius + np.pi * turn_radius, 4.5 * turn_radius + np.pi * turn_radius, 5 * turn_radius + np.pi * turn_radius, 1.5 + np.pi * (3.0/2.0) * turn_radius, 5 * turn_radius + np.pi * 2 * turn_radius,
                           5 * turn_radius + np.pi * (5.0/2.0) * turn_radius, 5 * turn_radius + np.pi * 3 * turn_radius, 5.5 * turn_radius + np.pi * 3 * turn_radius, 6 * turn_radius+ np.pi * 3 * turn_radius, 6.5 * turn_radius + np.pi * 3 * turn_radius, 6.5 * turn_radius + np.pi * 3.5 * turn_radius, 6.5 * turn_radius + np.pi * 4 * turn_radius,
                           7 * turn_radius + np.pi * 4 * turn_radius, 10 * turn_radius + np.pi * 4 * turn_radius])


    spline_points = np.linspace(0, phi_vals[-1], 1000)

    l_x = scipy.interpolate.CubicSpline(phi_vals, l_x_vals, bc_type='periodic')
    d_phi_l_x = l_x.derivative()
    dd_phi_l_x = d_phi_l_x.derivative()

    l_y = scipy.interpolate.CubicSpline(phi_vals, l_y_vals, bc_type='periodic')
    d_phi_l_y = l_y.derivative()
    dd_phi_l_y = d_phi_l_y.derivative()



    return l_x, d_phi_l_x, dd_phi_l_x, l_y, d_phi_l_y, dd_phi_l_y, spline_points


def get_ca_interp_path():

    turn_radius = 0.3

    path_length = 10 * turn_radius + np.pi * 4 * turn_radius

    l_x_vals = \
        [turn_radius, 0, -turn_radius, -1.5 * turn_radius, - 2.5 * turn_radius, - 1.5 * turn_radius, -1 * turn_radius,
         1.5 * turn_radius, 2 * turn_radius, 3 * turn_radius, 2 * turn_radius, 1 * turn_radius, 2 * turn_radius,
         2.5 * turn_radius,
         3 * turn_radius, 3.5 * turn_radius, 4.5 * turn_radius, 3.5 * turn_radius, 3 * turn_radius, 1.5 * turn_radius, 0]
    l_y_vals = \
        [turn_radius, turn_radius, turn_radius, turn_radius, 0, -turn_radius, -turn_radius, -turn_radius, -turn_radius,
         - 2 * turn_radius, -3 * turn_radius, -2 * turn_radius, - turn_radius, -turn_radius, -turn_radius, -turn_radius,
         0, turn_radius, turn_radius, turn_radius, turn_radius]

    phi_vals = [-turn_radius, 0, turn_radius, 1.5 * turn_radius, 1.5 * turn_radius + (np.pi / 2) * turn_radius,
                           1.5 * turn_radius + np.pi * turn_radius, 2 * turn_radius + np.pi * turn_radius,
                           4.5 * turn_radius + np.pi * turn_radius, 5 * turn_radius + np.pi * turn_radius,
                           1.5 + np.pi * (3.0 / 2.0) * turn_radius, 5 * turn_radius + np.pi * 2 * turn_radius,
                           5 * turn_radius + np.pi * (5.0 / 2.0) * turn_radius,
                           5 * turn_radius + np.pi * 3 * turn_radius, 5.5 * turn_radius + np.pi * 3 * turn_radius,
                           6 * turn_radius + np.pi * 3 * turn_radius, 6.5 * turn_radius + np.pi * 3 * turn_radius,
                           6.5 * turn_radius + np.pi * 3.5 * turn_radius, 6.5 * turn_radius + np.pi * 4 * turn_radius,
                           7 * turn_radius + np.pi * 4 * turn_radius, 8.5 * turn_radius + np.pi * 4 * turn_radius,
                          10 * turn_radius + np.pi * 4 * turn_radius]

    phi_vals_2 = phi_vals[2:] + path_length * np.ones(len(phi_vals) - 2)

    l_x_vals_tot = np.concatenate([l_x_vals, l_x_vals[2:]])
    l_y_vals_tot = np.concatenate([l_y_vals, l_y_vals[2:]])
    phi_vals_tot = np.concatenate([phi_vals, phi_vals_2])



    l_x = ca.interpolant('l_x', 'bspline', [phi_vals_tot], l_x_vals_tot)
    l_y = ca.interpolant('l_y', 'bspline', [phi_vals_tot], l_y_vals_tot)



    return l_x, l_y, path_length


if __name__ == "__main__":

    l_x, l_y, path_length = get_ca_interp_path()

    x_eval = ca.MX.sym('x_eval')

    l_x_d = casadi.jacobian(l_x(x_eval), x_eval)
    l_x_dd = casadi.jacobian(l_x_d, x_eval)
    l_x_ddd = casadi.jacobian(l_x_dd, x_eval)

    l_x_d_func = ca.Function('l_x_d_func', [x_eval], [l_x_d])
    l_x_dd_func = ca.Function('l_x_dd_func', [x_eval], [l_x_dd])
    l_x_ddd_func = ca.Function('l_x_ddd_func', [x_eval], [l_x_ddd])

    print("l_y(0): ", l_x(0))
    print("d_l_y_phi(0) :", l_x_d_func(0))
    print("dd_l_y_phi(0) :", l_x_dd_func(0))
    print("ddd_l_y_phi(0) :", l_x_ddd_func(0))

    turn_radius = 0.3

    spline_points = np.linspace(0, path_length, 1000)


    plt.plot(l_x(spline_points), l_y(spline_points), lw=3)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()





