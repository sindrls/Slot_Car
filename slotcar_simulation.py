import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import matplotlib.animation as animation

## TODO(ss): Lag friksjonskoeffesient i slot avhengig av trykk på tvers av slot, Dobbeltsjekke e.o.m.


#test = acados_slotcar_model.export_slot_car_ode_model()


show_plot = False
playback_ratio = 0.01

has_slid = False
track_car_angle_limit = np.pi * (3/5)

car_weight = 0.114
car_rot_inert = 0.001
car_length = 0.055
wheel_arm = 0.1

K_p = 1
D_d = 1

regular_slide_speed_tresh = 20
regular_slide_frict_coeff = 1.5
regular_sin_scaling_coeff = 1.8

u = 5


timesteps = 1000
timestep = 0.001

duration = timestep * timesteps

x_init = [0, 0, 0, 5]
x_t = np.zeros([4, timesteps])
x_t[:,0] = x_init

t = np.arange(0, timesteps * timestep, timestep)


track_angle = np.zeros(timesteps)
constraint_forces = np.zeros(timesteps)

#circle parameters
#turn_radius = 0.3
#def l_x(phi):
#    return turn_radius * np.cos(phi * (1 / turn_radius))

#def l_y(phi):
#    return turn_radius * np.sin(phi * (1 / turn_radius))

#def d_phi_l_x(phi):
#    return -np.sin(phi * (1 / turn_radius))

#def d_phi_l_y(phi):
#    return np.cos(phi * (1 / turn_radius))

#def dd_phi_l_x(phi):
#    return -(1/turn_radius) * np.cos(phi * (1 / turn_radius))

#def dd_phi_l_y(phi):
#    return -(1/turn_radius) * np.sin(phi * (1 / turn_radius))

#l_x_vals = np.asarray([turn_radius, 0, -turn_radius, 0, turn_radius])

#l_y_vals = np.asarray([0, turn_radius, 0, -turn_radius, 0])

#phi_vals = np.asarray([0, turn_radius * np.pi / 2 , turn_radius * np.pi, turn_radius * (3 * np.pi / 2), turn_radius * 2 * np.pi])


# Generating cubic spline for vehicle path

turn_radius = 0.3

l_x_vals = np.asarray([0, -turn_radius, -1.5 * turn_radius, - 2.5 * turn_radius, - 1.5 * turn_radius, -1 * turn_radius, 1.5 * turn_radius, 2 * turn_radius, 3 * turn_radius, 2 * turn_radius, 1 * turn_radius, 2 * turn_radius, 2.5 * turn_radius,
                        3 * turn_radius, 3.5 * turn_radius, 4.5 * turn_radius, 3.5 * turn_radius, 3 * turn_radius, 0])
l_y_vals = np.asarray([turn_radius, turn_radius, turn_radius, 0, -turn_radius, -turn_radius, -turn_radius, -turn_radius, - 2 * turn_radius, -3 * turn_radius, -2 * turn_radius, - turn_radius, -turn_radius, -turn_radius, -turn_radius, 0, turn_radius, turn_radius, turn_radius])

phi_vals = np.asarray([0, turn_radius, 1.5 * turn_radius, 1.5 * turn_radius + (np.pi / 2) * turn_radius, 1.5 * turn_radius + np.pi * turn_radius, 2 * turn_radius + np.pi * turn_radius, 4.5 * turn_radius + np.pi * turn_radius, 5 * turn_radius + np.pi * turn_radius, 1.5 + np.pi * (3.0/2.0) * turn_radius, 5 * turn_radius + np.pi * 2 * turn_radius,
                       5 * turn_radius + np.pi * (5.0/2.0) * turn_radius, 5 * turn_radius + np.pi * 3 * turn_radius, 5.5 * turn_radius + np.pi * 3 * turn_radius, 6 * turn_radius+ np.pi * 3 * turn_radius, 6.5 * turn_radius + np.pi * 3 * turn_radius, 6.5 * turn_radius + np.pi * 3.5 * turn_radius, 6.5 * turn_radius + np.pi * 4 * turn_radius,
                       7 * turn_radius + np.pi * 4 * turn_radius, 10 * turn_radius + np.pi * 4 * turn_radius])




l_x = scipy.interpolate.CubicSpline(phi_vals, l_x_vals, bc_type='periodic')
d_phi_l_x = l_x.derivative()
dd_phi_l_x = d_phi_l_x.derivative()

l_y = scipy.interpolate.CubicSpline(phi_vals, l_y_vals, bc_type='periodic')
d_phi_l_y = l_y.derivative()
dd_phi_l_y = d_phi_l_y.derivative()

spline_points = np.linspace(0, phi_vals[-1], 1000)


def alpha(x):

    alpha_est = d_phi_l_x(x[1]) * np.sin(x[0]) - d_phi_l_y(x[1]) * np.cos(x[0])

    #For numerical stability
    if alpha_est * alpha_est > 1:
        normalization_term = 1 / abs(alpha_est)
    else:
        normalization_term = 1

    return alpha_est * normalization_term


def mixing_term(x, t):
    mixing = np.zeros([2, 2])


    mixing = [[pow(car_length, 2) + car_rot_inert / car_weight, -car_length * alpha(x)],
              [-car_length * alpha(x), 1]]

    return mixing

def dynamics(x, t):
    dynamic = np.asarray( [ car_length * x[3] * x[3] * (dd_phi_l_x(x[1]) * np.sin(x[0]) - dd_phi_l_y(x[1]) * np.cos(x[0])),
                           car_length * x[2] * x[2] * (d_phi_l_x(x[1]) * np.cos(x[0]) + d_phi_l_y(x[1]) * np.sin(x[0]))])

    return dynamic


def tire_friction_body(x):

    cart_vel = np.zeros(2)

    cart_vel[0] = d_phi_l_x(x[1]) * x[3] - wheel_arm * np.sin(x[0]) * x[2]
    cart_vel[1] = d_phi_l_y(x[1]) * x[3] + wheel_arm * np.cos(x[0]) * x[2]

    rot_mat = np.zeros([2,2])

    rot_mat[0, 0] = np.cos(x[0])
    rot_mat[0, 1] = np.sin(x[0])
    rot_mat[1, 0] = -np.sin(x[0])
    rot_mat[1, 1] = np.cos(x[0])

    body_speed = rot_mat@cart_vel
    velocity_dir = np.arctan2(body_speed[1], -body_speed[0])

    car_angle = np.arctan2(np.sin(x[0]), np.cos(x[0]))

    track_vec = np.asarray([d_phi_l_x(x[1]), d_phi_l_y(x[1])])
    norm_track = track_vec / np.linalg.norm(track_vec)

    track_angle_val = np.arctan2(-norm_track[1], -norm_track[0])

    theta_rel = car_angle - track_angle_val

    #print(velocity_dir)

    body_friction = np.zeros(2)

    friction_force = -regular_slide_frict_coeff * np.sin(
        regular_sin_scaling_coeff * np.arctan(regular_slide_speed_tresh * np.arctan(velocity_dir)))

    body_friction[0] = (wheel_arm - np.sin(theta_rel) * np.sin(theta_rel) * car_length) * friction_force
    body_friction[1] = np.sin(theta_rel) * friction_force

    return body_friction


def speed_control(u, x, has_slid):
    force = np.zeros(2)

    car_angle = np.arctan2(np.sin(x[0]), np.cos(x[0]))
    track_vec = np.asarray([d_phi_l_x(x[1]), d_phi_l_y(x[1])])
    norm_track = track_vec / np.linalg.norm(track_vec)

    track_angle_val = np.arctan2(-norm_track[1], -norm_track[0])

    theta_rel = car_angle - track_angle_val

    track_vector = np.asarray([d_phi_l_x(x[1]), d_phi_l_y(x[1])])
    track_vector_norm = track_vector / np.linalg.norm(track_vector)

    car_vector = np.asarray([np.cos(x[0] + np.pi), np.sin(x[0] + np.pi)])

    if abs(np.dot(track_vector_norm, car_vector)) < 0.1:
        has_slid = True

    motor_force = (K_p * u - D_d * x[3])

    force[0] = -0.5 * np.sin(2 * theta_rel) * motor_force * car_length
    force[1] = np.cos(theta_rel) * motor_force

    if has_slid:
        force[0] = 0
        force[1] = 0
    return force, has_slid


def slot_car_ode(x,t, sliding):
    x_dot = np.zeros(4)

    x_dot[0] = x[2]
    x_dot[1] = x[3]

    mixing_mtx = np.linalg.inv(mixing_term(x, t))
    dynamic_forces = dynamics(x, t)
    internal_forces = np.dot(mixing_mtx,dynamic_forces)
    body_friction = tire_friction_body(x)
    speed_force, sliding = speed_control(u, x, sliding)
    #print(x)
    #print(np.linalg.det(mixing_term(x,t)))

    force_inertia_vec = np.asarray([1 / car_rot_inert, 1 / car_weight])

    test = force_inertia_vec * (body_friction + speed_force)

    x_dot[2:4] = np.dot(mixing_mtx, dynamic_forces) + force_inertia_vec * (body_friction + speed_force)

    return x_dot, sliding

for i in range(1, timesteps):
    d_x, has_slid = slot_car_ode(x_t[:, i - 1], 0, has_slid)

    x_t[:, i] = x_t[:, i-1] + timestep * d_x

    track_val = np.arctan2(d_phi_l_y(x_t[1, i]), d_phi_l_x(x_t[1, i])) + np.pi

    car_angle = np.arctan2(np.cos(x_t[0, i]), np.sin(x_t[0, i]))

    track_angle[i] = track_val - car_angle

    constraint_forces[i] = 0.5 * car_weight * (pow(d_phi_l_x(x_t[1, i]) * x_t[3, i] - car_length * np.sin(x_t[0, i]) * x_t[2, i], 2) \
                                + pow(d_phi_l_y(x_t[1, i]) * x_t[3, i] + car_length * np.cos(x_t[0, i]) * x_t[2, i], 2)) \
                                + 0.5 * car_rot_inert * pow(x_t[2, i], 2)







#sol = odeint(slot_car_ode, x_init, t)
#x_t = sol.T

def update(frame):
    # clear
    ax.clear()
    idx = int(np.floor(frame / (timestep / playback_ratio)))
    # plotting line
    slot_pos = [l_x(x_t[1, idx]), l_y(x_t[1, idx])]
    car_pos = [l_x(x_t[1, idx]) + car_length * np.cos(x_t[0, idx]), l_y(x_t[1, idx]) + car_length * np.sin(x_t[0, idx])]
    ax.plot(l_x(spline_points), l_y(spline_points), lw=3)
    ax.plot([slot_pos[0], car_pos[0]], [slot_pos[1], car_pos[1]], lw=3)
    ax.set(xlim=[- 5 * turn_radius - car_length, 5 * turn_radius + car_length],
          ylim=[-5 * turn_radius - car_length, 5 * turn_radius + car_length], xlabel='Meters', ylabel='Meters')

    #ax[1].clear()
    #ax[1].plot(constraint_forces[0:idx])

if show_plot:

    #plt.plot(t, x_t[0, :] / np.pi, 'b', label='angle(t)')

    plt.plot(t, x_t[0, :], 'g', label='angel(t)')

    plt.plot(t, x_t[1, :], 'r', label='distance(t)')

    plt.plot(t, x_t[2, :], 'b', label='angular_speed')

    plt.plot(t, x_t[3, :], 'y', label='speed')

    plt.legend(loc='best')

    plt.xlabel('t')

    plt.grid()

    plt.show()

else:

    fig, ax = plt.subplots(1, 1, layout='constrained')

    ax.set(xlim=[-turn_radius - car_length, turn_radius + car_length], ylim=[-turn_radius - car_length, turn_radius + car_length], xlabel='Meters', ylabel='Meters')
    #ax.legend()

    ani = animation.FuncAnimation(fig=fig, func=update, frames=int(duration / playback_ratio), interval=30)
    plt.show()



