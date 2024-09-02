
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from enum import Enum


class Sections(Enum):
    STRAIGHT = 0
    CW = 1
    CCW = 2


class CarreraTrack:
    def __init__(self, trackSectionTypeList):

        self.trackSectionList = []
        for i in trackSectionTypeList:
            if i == 0:
                self.trackSectionList.append(StraightTrack())
            elif i == 1:
                self.trackSectionList.append(ClockWiseTrack_60_25())
            elif i == 2:
                self.trackSectionList.append(CounterClockWiseTrack_60_25())

    def generate_track_spline(self, is_cw_inner_slot):

        x_vals = np.zeros(1)
        y_vals = np.zeros(1)
        phi_vals = np.zeros(1)
        end_section_positions = np.zeros(1)
        rotation_val = 0

        if not is_cw_inner_slot:
            y_vals[0] = 0.1

        for section in self.trackSectionList:
            rot_mat = np.zeros([2, 2])

            rot_mat[0, 0] = np.cos(-rotation_val)
            rot_mat[0, 1] = np.sin(-rotation_val)
            rot_mat[1, 0] = -np.sin(-rotation_val)
            rot_mat[1, 1] = np.cos(-rotation_val)

            last_pos = np.zeros([2, 1])
            last_pos[0, 0] = x_vals[-1]
            last_pos[1, 0] = y_vals[-1]

            section_xy_vals, section_phi_vals = section.get_xy_phi_vals(is_cw_inner_slot)

            rotated_xy = np.squeeze(rot_mat @ section_xy_vals + np.repeat([last_pos], 10, axis=0).T)

            added_phi_vals = section_phi_vals + np.repeat(phi_vals[-1], 10)

            x_vals = np.concatenate((x_vals, rotated_xy[0, :]))
            y_vals = np.concatenate((y_vals, rotated_xy[1, :]))
            phi_vals = np.concatenate((phi_vals, added_phi_vals))
            end_section_positions = np.concatenate((end_section_positions, [phi_vals[-1]]))

            rotation_val = rotation_val + section.get_delta_angle()

        l_x = sp.interpolate.CubicSpline(phi_vals, x_vals)

        l_y = sp.interpolate.CubicSpline(phi_vals, y_vals)

        spline_points = np.linspace(0, phi_vals[-1], 1000)

        plt.plot(l_x(spline_points), l_y(spline_points), lw=3)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        plt.show()

        return x_vals, y_vals, phi_vals, end_section_positions



class CarreraTrackSection:

    def __init__(self):

        self.distance_param = np.empty(0)
        self.xy_param = np.empty([2, 0])
        self.delta_angle = 0.0

    def get_phi_vals(self):
        return self.distance_param

    def get_delta_angle(self):
        return self.delta_angle


class StraightTrack(CarreraTrackSection):

    def __init__(self):

        CarreraTrackSection.__init__(self)
        self.delta_angle = 0

    def get_xy_phi_vals(self, is_cw_inner_slot):
        distance_params = np.linspace(0, 0.345, 11)
        self.distance_param = distance_params[1:]
        x_vals = np.linspace(0, 0.345, 11)

        if is_cw_inner_slot:
            y_vals = np.repeat(0,10)
            self.xy_param = np.vstack((x_vals[1:], y_vals))
        else:
            y_vals = np.repeat(0.0, 10)
            self.xy_param = np.vstack((x_vals[1:], y_vals))

        return self.xy_param, self.distance_param



class ClockWiseTrack_60_25(CarreraTrackSection):

    def __init__(self):

        CarreraTrackSection.__init__(self)
        self.delta_angle = -np.pi / 3.0

    def get_xy_phi_vals(self, is_cw_inner_slot):
        if is_cw_inner_slot:
            track_radius = 0.25
            angle_params = np.linspace(0, self.delta_angle, 11)[1:]
            distance_params = abs(angle_params) * track_radius
            self.distance_param = distance_params
            x_vals = np.sin(-angle_params) * track_radius
            y_vals = (np.cos(angle_params) - 1) * track_radius
            self.xy_param = np.vstack((x_vals, y_vals))
        else:
            track_radius = 0.35
            angle_params = np.linspace(0, self.delta_angle, 11)[1:]
            distance_params = abs(angle_params) * track_radius
            self.distance_param = distance_params
            x_vals = np.sin(-angle_params) * track_radius
            y_vals = (np.cos(angle_params) - 1) * track_radius
            self.xy_param = np.vstack((x_vals, y_vals))

        return self.xy_param, self.distance_param


class CounterClockWiseTrack_60_25(CarreraTrackSection):

    def __init__(self):

        CarreraTrackSection.__init__(self)
        self.delta_angle = np.pi / 3.0

    def get_xy_phi_vals(self, is_cw_inner_slot):
        if is_cw_inner_slot:
            track_radius = 0.35
            angle_params = np.linspace(0, self.delta_angle, 11)[1:]
            distance_params = angle_params * track_radius
            self.distance_param = distance_params
            x_vals = np.sin(angle_params) * track_radius
            y_vals = (1 - np.cos(angle_params)) * track_radius
            self.xy_param = np.vstack((x_vals, y_vals))
        else:
            track_radius = 0.25
            angle_params = np.linspace(0, self.delta_angle, 11)[1:]
            distance_params = angle_params * track_radius
            self.distance_param = distance_params
            x_vals = np.sin(angle_params) * track_radius
            y_vals = (1 - np.cos(angle_params)) * track_radius
            self.xy_param = np.vstack((x_vals, y_vals))

        return self.xy_param, self.distance_param


class CrossTrackAuto(CarreraTrackSection):

    def __init__(self):
        CarreraTrackSection.__init__(self)


class CrossTrackManual_start(CarreraTrackSection):

    def __init__(self):
        CarreraTrackSection.__init__(self)


class CrossTrackManual_end(CarreraTrackSection):

    def __init__(self):
        CarreraTrackSection.__init__(self)


class NarrowingTrack(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self)


class WideningTrack(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self)


class ClockWiseTrack_30_50(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self)


class CounterClockWiseTrack_30_50(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self)



#TODO: Finnish the track types
#TODO: Add logic to automatically detect if you need one or twoloops to get a closed curve (odd or even cross tracks)
#TODO: Add logic for manual slot cange (different spline)






if __name__ == "__main__":

    trackSectionList = []

    trackSectionList.append(0)
    trackSectionList.append(0)


    trackSectionList.append(1)
    trackSectionList.append(1)
    trackSectionList.append(1)

    trackSectionList.append(0)
    trackSectionList.append(0)

    trackSectionList.append(2)
    trackSectionList.append(2)
    trackSectionList.append(2)
    trackSectionList.append(2)
    trackSectionList.append(2)
    trackSectionList.append(2)

    trackSectionList.append(0)

    trackSectionList.append(1)
    trackSectionList.append(1)
    trackSectionList.append(1)

    trackSectionList.append(0)

    carreraTrack = CarreraTrack(trackSectionList)

    carreraTrack.generate_track_spline(False)


