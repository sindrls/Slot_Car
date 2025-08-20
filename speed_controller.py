import numpy as np
import casadi as ca
import sys

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
    def __init__(self, V_max, K_p, D_d, N, dt, Q, path_length, l_x, l_y, 
                 car_length, car_rot_inert, car_weight, wheel_arm,
                 slide_frict_coeff, slide_speed_thresh, sin_scaling_coeff):
        
        # Store params
        self.V_max = V_max
        self.K_p = K_p
        self.D_d = D_d
        self.N = N
        self.dt = dt
        self.Q = Q
        self.path_length = path_length
        self.l_x = l_x
        self.d_phi_l_x = self.l_x.derivative()
        self.dd_phi_l_x = self.d_phi_l_x.derivative()

        self.l_y = l_y
        self.d_phi_l_y = self.l_y.derivative()
        self.dd_phi_l_y = self.d_phi_l_y.derivative()

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

        # Setup for casadi
        phi_grid = np.linspace(0, path_length, 100000, endpoint=False)
        x_grid = l_x(phi_grid)
        y_grid = l_y(phi_grid)
        dx_grid = self.d_phi_l_x(phi_grid)
        dy_grid = self.d_phi_l_y(phi_grid)
        ddx_grid = self.dd_phi_l_x(phi_grid)
        ddy_grid = self.dd_phi_l_y(phi_grid)

        self.l_x_casadi = ca.interpolant('l_x', 'linear', [phi_grid], x_grid)
        self.l_y_casadi = ca.interpolant('l_y', 'linear', [phi_grid], y_grid)
        self.d_phi_l_x_casadi = ca.interpolant('d_phi_l_x', 'linear', [phi_grid], dx_grid)
        self.d_phi_l_y_casadi = ca.interpolant('d_phi_l_y', 'linear', [phi_grid], dy_grid)
        self.dd_phi_l_x_casadi = ca.interpolant('dd_phi_l_x', 'linear', [phi_grid], ddx_grid)
        self.dd_phi_l_y_casadi = ca.interpolant('dd_phi_l_y', 'linear', [phi_grid], ddy_grid)

        self.last_optimal_U = np.ones(self.N)
        self.last_optimal_X = np.tile([np.pi, 0, 0, 3], (self.N+1, 1)).T

        self._build_optimizer()

    def _build_optimizer(self):
        self.opti = ca.Opti()

        self.X = self.opti.variable(4, self.N+1)
        self.U = self.opti.variable(self.N)

        self.x0 = self.opti.parameter(4)
        self.opti.subject_to(self.X[:, 0] == self.x0)

        # Calculating the cost
        cost = 0
        for k in range(self.N):
            x_k = self.X[:,k]
            u_k = self.U[k]

            _, side_speed_k = self.x_dot_casadi(x_k, u_k)

            cost += self.Q[0] * side_speed_k ** 2
            cost -= self.Q[1] * self.X[3, k+1]
            
        cost += self.Q[2] * ca.sumsqr(self.U[1:] - self.U[:-1])

        self.opti.minimize(cost)

        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[k]
            x_next = self.step(x_k, u_k)
            self.opti.subject_to(self.X[:, k+1] == x_next)

        self.opti.subject_to(self.opti.bounded(0, self.U, self.V_max))

        p_opts = {
            "expand": True,
            "print_time": 0
        }

        s_opts = {
            "max_iter": 50,
            "print_level": 0,
            "acceptable_tol": 1e-3, 
            "max_cpu_time": 0.3
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    
    def mpc_controller(self, x_current):       
        self.opti.set_value(self.x0, x_current)

        self.opti.set_initial(self.U, self.last_optimal_U)
        self.opti.set_initial(self.X, self.last_optimal_X)


        try:
            sol = self.opti.solve()

            optimal_U = sol.value(self.U)
            optimal_X = sol.value(self.X)

    
        except RuntimeError as e:
            if "Maximum_Iterations_Exceeded" in str(e) or "Maximum_CpuTime_Exceeded" in str(e):
                optimal_U = self.opti.debug.value(self.U)
                optimal_X = self.opti.debug.value(self.X)

            else:
                print(f"MPC solver failed with an unexpected error: {e}")
                sys.exit(0)
        
        roll_length = 1
        self.last_optimal_U = np.roll(optimal_U, -roll_length)
        self.last_optimal_U[-roll_length:] = np.ones(roll_length) * optimal_U[-1]
        self.last_optimal_X = np.roll(optimal_X, -roll_length, axis=1)
        self.last_optimal_X[:,-roll_length:] = np.tile(optimal_X[:,-1], (roll_length,1)).T

        return optimal_U, optimal_X

    # Dynamic model

    def step(self, x, u):
        # RK4
        # k1, _ = self.x_dot_casadi(x, u)
        # k2, _ = self.x_dot_casadi(x + self.dt / 2 * k1, u)
        # k3, _ = self.x_dot_casadi(x + self.dt / 2 * k2, u)
        # k4, _ = self.x_dot_casadi(x + self.dt * k3, u)
        # x_next = x + self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

        # Forward Euler
        x_dot, _ = self.x_dot_casadi(x, u)
        x_next = x + self.dt * x_dot

        return x_next

    def alpha_casadi(self, x):
        alpha_est = self.d_phi_l_x_casadi(x[1]) * ca.sin(x[0]) - self.d_phi_l_y_casadi(x[1]) * ca.cos(x[0])
        normalization_term = ca.if_else(alpha_est * alpha_est > 1, 1 / ca.fabs(alpha_est), 1)
        return alpha_est * normalization_term

    def mixing_term_casadi(self, x):
        alpha_est = self.alpha_casadi(x)
        mixing = ca.vertcat(
            ca.horzcat(self.car_length**2 + self.car_rot_inert / self.car_weight, -self.car_length * alpha_est),
            ca.horzcat(-self.car_length * alpha_est, 1)
        )
        return mixing
    

    def matrix_inv_2x2_casadi(self, mtx):
        a, b = mtx[0, 0], mtx[0, 1]
        c, d = mtx[1, 0], mtx[1, 1]
        det = a * d - b * c
        inv = ca.vertcat(
            ca.horzcat(d, -b),
            ca.horzcat(-c, a)
        ) / det
        return inv


    def dynamics_casadi(self, x):
        dyn0 = self.car_length * x[3]**2 * (self.dd_phi_l_x_casadi(x[1]) * ca.sin(x[0]) - self.dd_phi_l_y_casadi(x[1]) * ca.cos(x[0]))
        dyn1 = self.car_length * x[2]**2 * (self.d_phi_l_x_casadi(x[1]) * ca.cos(x[0]) + self.d_phi_l_y_casadi(x[1]) * ca.sin(x[0]))
        return ca.vertcat(dyn0, dyn1)


    def body_speed_casadi(self, x):
        cart_vel_0 = self.d_phi_l_x_casadi(x[1]) * x[3] - self.wheel_arm * ca.sin(x[0]) * x[2]
        cart_vel_1 = self.d_phi_l_y_casadi(x[1]) * x[3] + self.wheel_arm * ca.cos(x[0]) * x[2]
        cart_vel = ca.vertcat(cart_vel_0, cart_vel_1)
        rot_mat = ca.vertcat(
            ca.horzcat(ca.cos(x[0]), ca.sin(x[0])),
            ca.horzcat(-ca.sin(x[0]), ca.cos(x[0]))
        )
        body_speed = rot_mat @ cart_vel
        return body_speed
    

    def theta_rel_casadi(self, x):
        car_angle = ca.fmod(x[0] + np.pi, 2 * np.pi) - np.pi
        track_vec_0 = self.d_phi_l_x_casadi(x[1])
        track_vec_1 = self.d_phi_l_y_casadi(x[1])
        track_angle_val = ca.atan2(-track_vec_1, -track_vec_0)
        return car_angle - track_angle_val
    

    def tire_friction_body_casadi(self, x, body_speed, theta_rel):
        velocity_dir = ca.atan2(body_speed[1], -body_speed[0])
        friction_force = - self.slide_frict_coeff * ca.sin(
            self.sin_scaling_coeff * ca.atan(self.slide_speed_thresh * ca.atan(velocity_dir)))
        bf0 = (self.wheel_arm - ca.sin(theta_rel)**2 * self.car_length) * friction_force
        bf1 = -ca.sin(theta_rel) * friction_force
        return ca.vertcat(bf0, bf1)

    def slot_car_dynamics_casadi(self, x, external_force):
        x_dot = ca.MX.zeros(4)
        x_dot[0] = x[2]
        x_dot[1] = x[3]
        mixing_mtx = self.matrix_inv_2x2_casadi(self.mixing_term_casadi(x))
        dynamic_forces = self.dynamics_casadi(x)
        internal_forces = mixing_mtx @ dynamic_forces
        force_inertia_vec = ca.vertcat(1 / self.car_rot_inert, 1 / self.car_weight)
        x_dot[2] = internal_forces[0] + force_inertia_vec[0] * external_force[0]
        x_dot[3] = internal_forces[1] + force_inertia_vec[1] * external_force[1]
        return x_dot
    

    def motor_model_casadi(self, x, u, theta_rel):
        motor_force = (self.K_p * u - self.D_d * x[3])
        f0 = -0.5 * ca.sin(2 * theta_rel) * motor_force * self.car_length
        f1 = ca.cos(theta_rel) * motor_force
        return ca.vertcat(f0, f1)
    
    def x_dot_casadi(self, x, u):
        body_speed = self.body_speed_casadi(x)
        theta_rel = self.theta_rel_casadi(x)
        speed_force = self.motor_model_casadi(x, u, theta_rel)
        body_friction = self.tire_friction_body_casadi(x, body_speed, theta_rel)
        external_force = speed_force + body_friction

        x_dot = self.slot_car_dynamics_casadi(x, external_force)
    
        return x_dot, body_speed[1]


