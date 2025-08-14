import numpy as np
from scipy.optimize import minimize
import time

def speed_controller(
    V_max, K, p, d_phi_l_x, d_phi_l_y, dd_phi_l_x, dd_phi_l_y, phi, phi_dot, delta_phi, horizon, predict_t, path_length):
    exp_val = 0

    for i in range(horizon):
        phi_i = i * delta_phi + predict_t * phi_dot + phi
        phi_i = np.mod(phi_i, path_length)
        d_phi_l_x_val = d_phi_l_x(phi_i)
        d_phi_l_y_val = d_phi_l_y(phi_i)

        dd_phi_l_x_val = dd_phi_l_x(phi_i)
        dd_phi_l_y_val = dd_phi_l_y(phi_i)

        kappa = abs((d_phi_l_x_val * dd_phi_l_y_val - dd_phi_l_x_val * d_phi_l_y_val))

        exp_val = exp_val + (p**i) * kappa

    return V_max * np.exp(- K * exp_val)

class MPC:
    def __init__(self, V_max, K_p, D_d, N, dt, Q, path_length,
                 d_phi_l_x, d_phi_l_y, dd_phi_l_x, dd_phi_l_y, 
                 car_length, car_rot_inert, car_weight, wheel_arm,
                 slide_frict_coeff, slide_speed_thresh, sin_scaling_coeff):
        self.V_max = V_max

        self.K_p = K_p
        self.D_d = D_d

        self.N = N
        self.dt = dt
        self.Q = Q

        self.path_length = path_length

        self.d_phi_l_x = d_phi_l_x
        self.d_phi_l_y = d_phi_l_y
        self.dd_phi_l_x = dd_phi_l_x
        self.dd_phi_l_y = dd_phi_l_y

        self.car_length = car_length
        self.car_rot_inert = car_rot_inert
        self.car_weight = car_weight
        self.wheel_arm = wheel_arm

        self.slide_frict_coeff = slide_frict_coeff
        self.slide_speed_thresh = slide_speed_thresh
        self.sin_scaling_coeff = sin_scaling_coeff

        self.motor_force_time = []
        self.tire_friction_time = []
        self.dynamics_time = []

        self.last_optimal_U = np.ones(self.N)*self.V_max

    def alpha(self, x):

        alpha_est = self.d_phi_l_x(x[1]) * np.sin(x[0]) - self.d_phi_l_y(x[1]) * np.cos(x[0])
        #For numerical stability
        if alpha_est * alpha_est > 1:
            normalization_term = 1 / abs(alpha_est)
        else:
            normalization_term = 1

        return alpha_est * normalization_term


    def mixing_term(self, x, t):
        # mixing = np.zeros([2, 2])
        alpha_est = self.alpha(x)
        mixing = np.array([[pow(self.car_length, 2) + self.car_rot_inert / self.car_weight, -self.car_length * alpha_est],
                [-self.car_length * alpha_est, 1]])

        return mixing
    
    def matrix_inv_2x2(self, mtx):
        a, b = mtx[0, 0], mtx[0, 1]
        c, d = mtx[1, 0], mtx[1, 1]
        det = a * d - b * c
        inv = np.array([[d, -b], [-c, a]]) / det
        return inv

    def dynamics(self, x, t):
        dynamic = np.asarray( [ self.car_length * x[3] * x[3] * (self.dd_phi_l_x(x[1]) * np.sin(x[0]) - \
                                                                 self.dd_phi_l_y(x[1]) * np.cos(x[0])),
                            self.car_length * x[2] * x[2] * (self.d_phi_l_x(x[1]) * np.cos(x[0]) + \
                                                             self.d_phi_l_y(x[1]) * np.sin(x[0]))])
        
        return dynamic

    def body_speed(self, x):
        
        cart_vel = np.zeros(2)

        cart_vel[0] = self.d_phi_l_x(x[1]) * x[3] - self.wheel_arm * np.sin(x[0]) * x[2]
        cart_vel[1] = self.d_phi_l_y(x[1]) * x[3] + self.wheel_arm * np.cos(x[0]) * x[2]

        rot_mat = np.zeros([2,2])

        rot_mat[0, 0] = np.cos(x[0])
        rot_mat[0, 1] = np.sin(x[0])
        rot_mat[1, 0] = -np.sin(x[0])
        rot_mat[1, 1] = np.cos(x[0])

        return rot_mat@cart_vel
    
    def theta_rel(self, x):
        car_angle = np.mod(x[0] + np.pi, 2*np.pi) - np.pi
        track_vec = np.asarray([self.d_phi_l_x(x[1]), self.d_phi_l_y(x[1])])
        track_angle_val = np.arctan2(-track_vec[1], -track_vec[0])

        return car_angle - track_angle_val
    
    def tire_friction_body(self, x, body_speed, theta_rel):
        velocity_dir = np.arctan2(body_speed[1], -body_speed[0])

        body_friction = np.zeros(2)

        friction_force = - self.slide_frict_coeff * np.sin(
            self.sin_scaling_coeff * np.arctan(self.slide_speed_thresh * np.arctan(velocity_dir)))

        body_friction[0] = (self.wheel_arm - np.sin(theta_rel) * np.sin(theta_rel) * self.car_length) * friction_force
        body_friction[1] = -np.sin(theta_rel) * friction_force

        return body_friction

    def slot_car_dynamics(self, t, x, external_force):
        x_dot = np.zeros(4)
        x_dot[0] = x[2]
        x_dot[1] = x[3]


        mixing_mtx = self.matrix_inv_2x2(self.mixing_term(x, t))
        dynamic_forces = self.dynamics(x, t)

        internal_forces = np.dot(mixing_mtx, dynamic_forces)

        force_inertia_vec = np.asarray([1 / self.car_rot_inert, 1 / self.car_weight])
        x_dot[2:4] = internal_forces + force_inertia_vec * external_force

        return x_dot
    
    def motor_model(self, x, u, theta_rel):
        force = np.zeros(2)

        motor_force = (self.K_p * u - self.D_d * x[3])
        force[0] = -0.5 * np.sin(2 * theta_rel) * motor_force * self.car_length
        force[1] = np.cos(theta_rel) * motor_force

        return force
    
    def x_dot(self, x, u):

        body_speed = self.body_speed(x)
        theta_rel = self.theta_rel(x)
        speed_force = self.motor_model(x, u, theta_rel)
        body_friction = self.tire_friction_body(x, body_speed, theta_rel)
        external_force = speed_force + body_friction

        x_dot = self.slot_car_dynamics(0, x, external_force)


    
        return x_dot, body_speed[1]
    
    def mpc_predict(self, x, u):
        # RK4 
        # k1 = self.x_dot(x, u)
        # k2 = self.x_dot(x + k1 * self.dt / 2.0, u)
        # k3 = self.x_dot(x + k2 * self.dt / 2.0, u)
        # k4 = self.x_dot(x + k3 * self.dt, u)

        # x_next = x + (k1 + 2*k2 + 2*k3 + k4) / 6.0 * self.dt

        # Forward euler
        x_dot, side_speed = self.x_dot(x, u)
        x_next = x + x_dot * self.dt

        return x_next, side_speed


    def mpc_cost_func(self, U, x_current):
        cost = 0.0
        predicted_x = np.copy(x_current)
        
        for i in range(self.N):
            u_i = U[i]
            predicted_x, side_speed = self.mpc_predict(predicted_x, u_i)
            
            # Cost of sideways slipping

            side_speed_cost = self.Q[0] * side_speed ** 2

            # Reward for high speed
            speed_cost = - self.Q[1] * predicted_x[3]


            cost += side_speed_cost + speed_cost
        return cost

    def mpc_controller(self, x_current):        
        bounds = [(0.0, self.V_max)] * self.N
        u_guess = np.roll(self.last_optimal_U, -1)
        u_guess[-1:] = self.V_max
        u_guess = self.last_optimal_U

        options_dict = {
            "maxiter": 7, # Max iterations
            "disp": False # Debug covergence stats
        }

        result = minimize(
            fun= self.mpc_cost_func,
            x0 = u_guess,
            args = (x_current),
            method = "SLSQP",
            bounds=bounds,
            tol=1e-3,
            options=options_dict
        )

        optimal_U = result.x

        self.last_optimal_U = optimal_U
        return optimal_U[0], optimal_U
