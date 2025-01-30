
import numpy as np
from carrera_slot_car_track_spline_creator_class import CarreraTrack, Sections
import scipy
import time



class SlotCarKalmanTracker:
    def __init__(self, carrera_track_list):
        """ Create a new point at the origin """

        self.state = np.zeros([2, 1])
        self.cov = np.zeros([2, 2])

        self.cov[0, 0] = 0.1
        self.cov[1, 1] = 0.1

        self.Q = np.zeros([2, 2])
        self.Q[1, 1] = 4

        self.curr_timestamp = 0

        self.carrera_track = CarreraTrack(carrera_track_list)

        x_vals, y_vals, phi_vals, end_section_positions = self.carrera_track.generate_track_spline(True)

        self.x_vals = x_vals
        self.y_vals = y_vals
        self.phi_vals = phi_vals
        self.end_section_positions = end_section_positions

        self.l_x = scipy.interpolate.CubicSpline(phi_vals, x_vals)

        self.l_y = scipy.interpolate.CubicSpline(phi_vals, y_vals)

        self.track_length = phi_vals[-1]

        self.prev_time = time.time()

        self.num_laps_avg = 10
        self.lap_idx = 0
        self.lap_times = np.zeros(self.num_laps_avg)

    def measurement_update(self, z, R):
        z_val = np.zeros((2, 1))
        z_val[0] = z[0]
        z_val[1] = z[1]
        d_phi_l_x = self.l_x.derivative()
        d_phi_l_y = self.l_y.derivative()

        H_mat = np.zeros((2, 2))

        H_mat[0, 0] = d_phi_l_x(self.state[0])
        H_mat[1, 0] = d_phi_l_y(self.state[0])

        K_inv = H_mat @ self.cov @ np.transpose(H_mat) + R

        K = self.cov @ np.transpose(H_mat) @ np.linalg.inv(K_inv)

        self.cov = self.cov - K @ H_mat @ self.cov
        h = np.zeros((2, 1))
        h[0] = self.l_x(self.state[0])
        h[1] = self.l_y(self.state[0])

        self.state = self.state + K @ (z_val - h)

    def predict_state(self, delta_time):
        A_mat = np.zeros((2, 2))
        A_mat[0, 1] = delta_time

        self.state = self.state + A_mat @ self.state

        self.cov = self.cov + (A_mat @ self.cov + self.cov @ np.transpose(A_mat)) + self.Q * delta_time

        if self.state[0, 0] > self.track_length:
            self.state[0, 0] = self.state[0, 0] - self.track_length
            self.lap_idx = self.lap_idx + 1

            if self.lap_idx >= self.num_laps_avg:
                self.lap_idx = 0

            new_time = time.time()
            self.lap_times[self.lap_idx] = new_time - self.prev_time
            self.prev_time = new_time



    def get_state(self):
        return self.state

    def get_xy_pos(self, phi):
        meas = np.zeros((2, 1))

        meas[0] = self.l_x(phi)
        meas[1] = self.l_y(phi)

        return meas

    def get_track_length(self):
        return self.track_length

    def get_lap_time_mean(self):
        return np.mean(self.lap_times)

    def get_lap_time_std(self):
        return np.std(self.lap_times)


if __name__ == "__main__":

    trackSectionList = []

    timesteps = 100
    timestep = 0.02
    car_state = np.zeros((2, 1))
    car_state[0] = 0.1

    trackSectionList.append(Sections.CW)
    trackSectionList.append(Sections.CW)
    trackSectionList.append(Sections.CW)

    trackSectionList.append(Sections.STRAIGHT)

    trackSectionList.append(Sections.CW)
    trackSectionList.append(Sections.CW)
    trackSectionList.append(Sections.CW)

    trackSectionList.append(Sections.STRAIGHT)

    tracker = SlotCarKalmanTracker(trackSectionList)

    for i in range(timesteps):
        car_state[1] = 1
        car_state[0] = car_state[0] + car_state[1] * timestep

        tracker.predict_state(timestep)

        print("Curr state is: ", car_state)
        print("Kalman est is: ", tracker.get_state())

        R = np.zeros((2, 2))
        R[0, 0] = 0.1
        R[1, 1] = 0.1

        tracker.measurement_update(tracker.get_xy_pos(car_state[0]), R)

        if car_state[0] > tracker.get_track_length():
            car_state[0] = car_state[0] - tracker.get_track_length()
