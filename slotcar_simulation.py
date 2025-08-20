
import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from carrera_slot_car_track_spline_creator_class import CarreraTrack, Sections
from speed_controller import speed_controller, MPC
from scipy.integrate import solve_ivp
import tqdm
import time


## TODO(ss): Lag friksjonskoeffesient i slot avhengig av trykk på tvers av slot, Dobbeltsjekke e.o.m.

# Speed control
V_max = 5.0
K =  0.04
p = 1
lookback_distance = 0.0

lowpass_val = 0
prev_throttle = 0

show_plot = True
playback_ratio = 0.01

use_speed_reg = False

has_slid = False
track_car_angle_limit = np.pi * (3/5)

car_weight = 0.114
car_rot_inert = 0.00023
car_length = 0.055
wheel_arm = 0.1

K_p = 1
D_d = 1

regular_slide_speed_tresh = 20
regular_slide_frict_coeff = 0.35
regular_sin_scaling_coeff = 1.8

u = 1.9

num_rounds = 3

lap_times = np.zeros([1,num_rounds])
lap_idx = 0

prev_lap = 0

timesteps = 1000
timestep = 0.005

duration = timestep * timesteps

x_init = [np.pi, 0, 0, 3]
x_t = np.zeros([4, timesteps])
x_t[:,0] = x_init

t = np.arange(0, timesteps * timestep, timestep)

turn_radius  = 0.25

throttle_values = np.zeros(timesteps)

track_angle = np.zeros(timesteps)


trackSectionList = [
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.CW,
        Sections.CW,
        Sections.CW,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.CW,
        Sections.CW,
        Sections.CW,
    ]


track = CarreraTrack(trackSectionList)

x_vals, y_vals, phi_vals, end_section_positions = track.generate_track_spline(True)


l_x = CubicSpline(phi_vals, x_vals, bc_type='periodic')
d_phi_l_x = l_x.derivative()
dd_phi_l_x = d_phi_l_x.derivative()

l_y = CubicSpline(phi_vals, y_vals, bc_type='periodic')
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

    body_friction = np.zeros(2)

    friction_force = -regular_slide_frict_coeff * np.sin(
        regular_sin_scaling_coeff * np.arctan(regular_slide_speed_tresh * np.arctan(velocity_dir)))

    body_friction[0] = (wheel_arm - np.sin(theta_rel) * np.sin(theta_rel) * car_length) * friction_force
    body_friction[1] = -np.sin(theta_rel) * friction_force

    return body_friction

def speed_control(u, x, x_meas, has_slid):
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

    if use_speed_reg:
        delta_phi = 0.12
        horizon = 10
        predict_t = 0.2
        u_val = speed_controller(
            V_max, K, p, d_phi_l_x, d_phi_l_y,
            dd_phi_l_x, dd_phi_l_y, x_meas[1], x_meas[3], 
            delta_phi, horizon, predict_t, phi_vals[-1])
        u = u_val

    motor_force = (K_p * u - D_d * x[3])
    force[0] = -0.5 * np.sin(2 * theta_rel) * motor_force * car_length
    force[1] = np.cos(theta_rel) * motor_force

    if has_slid:
        force[0] = 0
        force[1] = 0
    return force, has_slid, u

def slot_car_dynamics(t, x, external_force):
    x_dot = np.zeros(4)
    x_dot[0] = x[2]
    x_dot[1] = x[3]

    mixing_mtx = np.linalg.inv(mixing_term(x, t))
    dynamic_forces = dynamics(x, t)
    internal_forces = np.dot(mixing_mtx,dynamic_forces)

    force_inertia_vec = np.asarray([1 / car_rot_inert, 1 / car_weight])
    x_dot[2:4] = internal_forces + force_inertia_vec * external_force

    return x_dot


# Tracking time delay simulation
base_delay = 0.4
base_delay_steps = int(base_delay / timestep)

tracking_frequency = 10.0
update_steps = int((1.0 / tracking_frequency) / timestep)
state_buffer_size = 10000
state_buffer = np.zeros([4, state_buffer_size])

measured_state_t = np.zeros([4,timesteps])

measured_state = x_init

delay_steps = 0
delay_buffer_idx = 0

mpc_horizon = 15
mpc_dt = 0.005
mpc_Q = np.array([5.0, 0.1, 2.5])

mpc = MPC(V_max, K_p, D_d, mpc_horizon, mpc_dt, mpc_Q, 
          phi_vals[-1], l_x, l_y, 
          car_length, car_rot_inert, car_weight, wheel_arm,
          regular_slide_frict_coeff, regular_slide_speed_tresh,
          regular_sin_scaling_coeff)

mpc_time_record = np.zeros(timesteps)

for i in tqdm.tqdm(range(1, timesteps)):

    state_buffer_idx = np.mod(i-1, state_buffer_size)

    if delay_steps == update_steps:
        delay_steps = 0

        delay_buffer_idx = state_buffer_idx - base_delay_steps
        measured_state = state_buffer[:, delay_buffer_idx]

    else:
        delay_steps = delay_steps + 1

    current_state = x_t[:, i-1]
    if not use_speed_reg:
        t0 = time.time()
        last_optimal_U, optimal_X = mpc.mpc_controller(current_state)
        t1 = time.time()
        mpc_time_record[i] = t1-t0
        u = last_optimal_U[0]


    body_friction = tire_friction_body(current_state)
    speed_force, sliding, u = speed_control(u, current_state, measured_state, has_slid)
    throttle_values[i] = u

    external_force = body_friction + speed_force

    t_start = t[i-1]
    t_end = t[i]

    sol = solve_ivp(
        fun=slot_car_dynamics,
        t_span=[t_start, t_end],
        y0=current_state,
        method="RK45",
        args=(external_force,),
        dense_output=True
    )

    x_t[:,i] = sol.y[:,-1]

    state_buffer[:, state_buffer_idx] = x_t[:, i]
    measured_state_t[:,i] = measured_state

    if x_t[1,i] > phi_vals[-1]:
        lap_times[0,lap_idx] = i * timestep - prev_lap
        prev_lap = i * timestep
        lap_idx = lap_idx + 1
        if lap_idx >= (num_rounds):
            lap_idx = 0
        x_t[1,i] = x_t[1,i] - phi_vals[-1]


    track_val = np.arctan2(d_phi_l_y(x_t[1, i]), d_phi_l_x(x_t[1, i])) + np.pi
    car_angle = np.arctan2(np.cos(x_t[0, i]), np.sin(x_t[0, i]))
    track_angle[i] = track_val - car_angle


    if sliding:
        print("Has slid at step: ", i)
        break



print("avg Lap time is ", np.mean(lap_times))
print("Variance  is: ",np.std(lap_times) )
print("vals  are: ",lap_times )
print("Avg speed is: ", np.mean(x_t[3,:]))
print("Avg throttle is: ", np.mean(throttle_values))
print("Has slid: ", sliding)
print("Average MPC calculation duration: ", mpc_time_record.mean())
print("Max MPC calculation duration: ", mpc_time_record.max())



def update(frame):
    # clear
    ax.clear()
    idx = int(np.floor(frame / (timestep / playback_ratio)))
    # plotting line
    slot_pos = [l_x(x_t[1, idx]), l_y(x_t[1, idx])]
    car_pos = [l_x(x_t[1, idx]) + car_length * np.cos(x_t[0, idx]), l_y(x_t[1, idx]) + car_length * np.sin(x_t[0, idx])]
    ax.plot(l_x(spline_points), l_y(spline_points), lw=3)
    ax.plot([slot_pos[0], car_pos[0]], [slot_pos[1], car_pos[1]], lw=3)
    ax.set(xlim=[-1 * turn_radius - car_length, 10 * turn_radius + car_length],
          ylim=[-5 * turn_radius - car_length, 5 * turn_radius + car_length], xlabel='Meters', ylabel='Meters')


if show_plot:

    plt.plot(t, x_t[0, :], 'g', label='angle(t)')
    plt.plot(t, x_t[1, :], 'r', label='distance(t)')
    plt.plot(t, x_t[2, :], 'b', label='angular_speed')
    plt.plot(t, x_t[3, :], 'y', label='speed')
    plt.plot(t, measured_state_t[3,:], label='Measured state')
    plt.plot(t, throttle_values, "black", label="throttle")

    plt.legend(loc='best')

    plt.xlabel('t')

    plt.grid()

    plt.show()


fig, ax = plt.subplots(1, 1, layout='constrained')

ax.set(xlim=[-turn_radius - car_length, turn_radius + car_length], ylim=[-turn_radius - car_length, turn_radius + car_length], xlabel='Meters', ylabel='Meters')

ani = animation.FuncAnimation(fig=fig, func=update, frames=int(duration / playback_ratio), interval=30)
plt.show()



