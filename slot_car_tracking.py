
from acados_template import AcadosSim, AcadosSimSolver
import acados_slotcar_model
import numpy as np
import matplotlib.pyplot as plt
import casadi

class slot_car_tracker:

    def __init__(self):
        """ Create a new point at the origin """

        self.model = acados_slotcar_model.export_slot_car_ode_model()
        self.state = np.zeros([4, 1])
        self.cov = np.zeros([4, 4])
        self.Q = np.zeros([4, 4])
        self.curr_timestamp= 0
        self.H_jac = casadi.Function('H_jac',[self.model.x],[casadi.jacobian(self.model.z, self.model.x)])
        self.A = casadi.Function('A',[self.model.x],[casadi.jacobian(self.model.f_expl_expr, self.model.x)])
        self.h = casadi.Function('h',[self.model.x],[self.model.z])
        dt = casadi.SX.sym("dt")
        self.x_next = casadi.Function('x_next',[self.model.x], [self.model.x + self.model.f_expl_expr * dt])

    def measurement_update(self, measurement):

        H_mat = self.H_jac(self.state)
        K_inv = H_mat @ self.cov @ H_mat.T + measurement.R
        K = self.cov * H_mat.T @ np.linalg.inv(K_inv)

        self.cov = self.cov - K @ H_mat * self.cov

        self.state = self.state + K @ (measurement.z - self.h(self.state))

    def predict_state(self, delta_time):

        self.state = self.x_next(np.vstack((self.state, delta_time)))
        A_mat = self.A(self.state)

        self.cov = self.cov + delta_time * (A_mat @ self.cov + self.cov @ A_mat.T) + self.Q * delta_time







