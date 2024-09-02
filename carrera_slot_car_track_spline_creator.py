
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt




class CarreraTrackSection:

    def __init__(self, is_cw_inner_slot):

        self.distance_param = np.empty(0)
        self.xy_param = np.empty([2, 0])
        self.delta_angle = 0.0
        self.is_cw_inner_slot = is_cw_inner_slot

    def get_xy_vals(self):
        return self.xy_param

    def get_phi_vals(self):
        return self.distance_param

    def get_delta_angle(self):
        return self.delta_angle

    def is_cw_inner_slot_val(self):
        return self.is_cw_inner_slot

class StraightTrack(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):

        CarreraTrackSection.__init__(self, is_cw_inner_slot)
        self.delta_angle = 0
        distance_params = np.linspace(0, 0.345, 11)
        self.distance_param = distance_params[1:]
        x_vals = np.linspace(0, 0.345, 11)

        if is_cw_inner_slot:
            y_vals = np.repeat(0,10)
            self.xy_param = np.vstack((x_vals[1:], y_vals))
        else:
            y_vals = np.repeat(0.0, 10)
            self.xy_param = np.vstack((x_vals[1:], y_vals))



class ClockWiseTrack_60_25(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):

        CarreraTrackSection.__init__(self, is_cw_inner_slot)
        self.delta_angle = -np.pi / 3.0

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


class CounterClockWiseTrack_60_25(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):

        CarreraTrackSection.__init__(self, is_cw_inner_slot)
        self.delta_angle = np.pi / 3.0

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


class CrossTrackAuto(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self, is_cw_inner_slot)


class CrossTrackManual_start(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self, is_cw_inner_slot)


class CrossTrackManual_end(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self, is_cw_inner_slot)


class NarrowingTrack(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self, is_cw_inner_slot)


class WideningTrack(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self, is_cw_inner_slot)


class ClockWiseTrack_30_50(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self, is_cw_inner_slot)


class CounterClockWiseTrack_30_50(CarreraTrackSection):

    def __init__(self, is_cw_inner_slot):
        CarreraTrackSection.__init__(self, is_cw_inner_slot)



#TODO: Finnish the track types
#TODO: Add logic to automatically detect if you need one or twoloops to get a closed curve (odd or even cross tracks)
#TODO: Add logic for manual slot cange (different spline)




def generateTrackSpline(trackSectionList):

    x_vals = np.zeros(1)
    y_vals = np.zeros(1)
    phi_vals = np.zeros(1)
    rotation_val = 0

    if not trackSectionList[0].is_cw_inner_slot_val():
        y_vals[0] = 0.1

    for section in trackSectionList:

        rot_mat = np.zeros([2, 2])

        rot_mat[0, 0] = np.cos(-rotation_val)
        rot_mat[0, 1] = np.sin(-rotation_val)
        rot_mat[1, 0] = -np.sin(-rotation_val)
        rot_mat[1, 1] = np.cos(-rotation_val)

        last_pos = np.zeros([2, 1])
        last_pos[0, 0] = x_vals[-1]
        last_pos[1, 0] = y_vals[-1]

        rotated_xy = np.squeeze(rot_mat @ section.get_xy_vals() + np.repeat([last_pos], 10, axis=0 ).T)

        added_phi_vals = section.get_phi_vals() + np.repeat(phi_vals[-1], 10)

        x_vals = np.concatenate((x_vals, rotated_xy[0, :]))
        y_vals = np.concatenate((y_vals, rotated_xy[1, :]))
        phi_vals = np.concatenate((phi_vals, added_phi_vals))

        test = section.get_delta_angle()
        rotation_val = rotation_val + section.get_delta_angle()

    #x_vals[-1] = x_vals[0]
    #y_vals[-1] = y_vals[0]

    l_x = sp.interpolate.CubicSpline(phi_vals, x_vals)

    l_y = sp.interpolate.CubicSpline(phi_vals, y_vals)

    spline_points = np.linspace(0, phi_vals[-1], 1000)

    plt.plot(l_x(spline_points), l_y(spline_points), lw=3)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.show()

    return x_vals, y_vals, phi_vals






if __name__ == "__main__":
    cwTrackPart = ClockWiseTrack_60_25(False)
    ccwTrackPart = CounterClockWiseTrack_60_25(False)
    straightTrackPart = StraightTrack(False)

    trackSectionList = []

    trackSectionList.append(straightTrackPart)
    trackSectionList.append(straightTrackPart)


    trackSectionList.append(cwTrackPart)
    trackSectionList.append(cwTrackPart)
    trackSectionList.append(cwTrackPart)

    trackSectionList.append(straightTrackPart)
    trackSectionList.append(straightTrackPart)

    trackSectionList.append(ccwTrackPart)
    trackSectionList.append(ccwTrackPart)
    trackSectionList.append(ccwTrackPart)
    trackSectionList.append(ccwTrackPart)
    trackSectionList.append(ccwTrackPart)
    trackSectionList.append(ccwTrackPart)

    trackSectionList.append(straightTrackPart)

    trackSectionList.append(cwTrackPart)
    trackSectionList.append(cwTrackPart)
    trackSectionList.append(cwTrackPart)

    trackSectionList.append(straightTrackPart)

    generateTrackSpline(trackSectionList)

