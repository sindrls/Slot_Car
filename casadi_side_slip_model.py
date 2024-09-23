from acados_template import AcadosModel
from casadi import MX, Function, vertcat, sin, cos, atan2, fabs, inv, sign, fmax, fmin, gt, ge, if_else, exp
import casadi as ca
import slot_car_path
import numpy as np
from slot_car_path import get_ca_interp_path

def export_slot_car_ode_model() -> AcadosModel:


    #TODO(ss): finne ut om theta rel er god eller ikke

    model_name = 'slot_car_sideslip'

    # constants

    car_weight = 0.114
    car_rot_inert = 0.00023
    car_length = 0.055
    wheel_arm = 0.1

    regular_slide_speed_tresh = 20
    regular_slide_frict_coeff = 0.3
    regular_sin_scaling_coeff = 1.8

    #motor params
    K_p = 1
    D_d = 1

    #circle radius
    radius = 0.3

    # set up states & controls (input)
    theta    = MX.sym('theta')
    phi      = MX.sym('phi')
    dtheta   = MX.sym('dtheta')
    dphi      = MX.sym('dphi')

    x = vertcat(theta, phi, dtheta, dphi)

    F = MX.sym('F')
    u = vertcat(F)

    # xdot (output)
    theta_dot    = MX.sym('theta_dot')
    phi_dot      = MX.sym('phi_dot')
    dtheta_dot   = MX.sym('dtheta_dot')
    dphi_dot      = MX.sym('dphi_dot')

    xdot = vertcat(theta_dot, phi_dot, dtheta_dot, dphi_dot)

    # parameters
    p = []



    #racetrack:

    l_x = radius * cos(phi / radius)
    d_phi_l_x = ca.jacobian(l_x, phi)
    dd_phi_l_x = ca.jacobian(d_phi_l_x, phi)

    l_y = radius * sin(phi / radius)
    d_phi_l_y = ca.jacobian(l_y, phi)
    dd_phi_l_y = ca.jacobian(d_phi_l_x, phi)



    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)

    #alpha = if_else(d_phi_l_x * sin_theta - d_phi_l_y * cos_theta > 1,
    #                1,
    #                if_else(d_phi_l_x * sin_theta - d_phi_l_y * cos_theta < -1,
    #                        -1,
    #                        d_phi_l_x * sin_theta - d_phi_l_y * cos_theta))

    alpha = d_phi_l_x * sin_theta - d_phi_l_y * cos_theta

    vert_vec = vertcat(car_length * dphi * dphi * (dd_phi_l_x * sin_theta - dd_phi_l_y * cos_theta),
                           car_length * dtheta * dtheta * (d_phi_l_x * cos_theta + d_phi_l_y * sin_theta))

    mixing_term = MX.zeros(2,2)

    mixing_term[0, 0] = pow(car_length, 2) + car_rot_inert / car_weight
    mixing_term[1, 0] = -car_length * alpha
    mixing_term[0, 1] = -car_length * alpha
    mixing_term[1, 1] = 1

    # Simplified friction model (check sign of vals):

    cart_vel = MX.zeros(2, 1)

    cart_vel[0, 0] = d_phi_l_x * x[3] - wheel_arm * sin_theta * x[2]
    cart_vel[1, 0] = d_phi_l_y * x[3] + wheel_arm * cos_theta * x[2]

    rot_mat = MX.zeros(2, 2)

    rot_mat[0, 0] = cos_theta
    rot_mat[0, 1] = sin_theta
    rot_mat[1, 0] = -sin_theta
    rot_mat[1, 1] = cos_theta

    body_speed = rot_mat @ cart_vel
    velocity_dir = atan2(body_speed[1], -body_speed[0])

    car_angle = np.arctan2(sin(theta), cos(theta))

    track_vec = vertcat(d_phi_l_x, d_phi_l_y)

    norm_track = track_vec / ca.norm_2(track_vec)

    track_angle_val = ca.arctan2(-norm_track[1], -norm_track[0])

    theta_rel = car_angle - track_angle_val

    body_friction = -regular_slide_frict_coeff * sin(
        regular_sin_scaling_coeff * MX.arctan(regular_slide_speed_tresh * MX.arctan(velocity_dir)))

    tyre_torque = (wheel_arm - sin(theta_rel) * sin(theta_rel) * car_length) * body_friction / car_rot_inert
    tyre_force = sin(theta_rel) * body_friction / car_weight

    tyre_vals = vertcat(tyre_torque,
                        tyre_force)

    # motor dynamics (check sign of vals)

    track_dir = MX.zeros(2, 1)

    track_dir[0] = d_phi_l_x
    track_dir[1] = d_phi_l_y
    #TODO(ss) Hvorfor gjør motor torque at simuleringen divergerer?
    motor_torque = 0.5 * sin(2 * theta_rel) * (K_p * u - D_d * dphi) * car_length
    motor_force = cos(theta_rel) * (K_p * u - D_d * dphi)

    motor_vals = vertcat(motor_torque / car_rot_inert,
                         motor_force / car_weight)

    # Total ODE
    second_order_ode = inv(mixing_term) @ vert_vec + tyre_vals + motor_vals

    f_expl = vertcat(dtheta,
                     dphi,
                     second_order_ode)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    theta_no_slip = np.arcsin(wheel_arm / radius)


    return model, theta_no_slip, radius





