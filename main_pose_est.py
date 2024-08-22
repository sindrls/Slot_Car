# import the opencv library
import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation
import casadi as ca
import time
import matplotlib.pyplot as plt
import scipy

import carrera_slot_car_track_spline_creator_class



def transform_mtx_inv(mtx):
    transformation_inverse = np.zeros([4, 4])
    transformation_inverse[0:3, 0:3] = mtx[0:3, 0:3].T
    transformation_inverse[0:3, 3] = -mtx[0:3, 0:3].T @ mtx[0:3, 3]
    transformation_inverse[3, 3] = 1

    return transformation_inverse


def ippe_pose_est(H):
    v = H[0:2, 2]
    J = np.asarray([[H[0, 0] - H[2, 0] * H[0, 2], H[0, 1] - H[2, 1] * H[0, 2]],
                    [H[1, 0] - H[2, 0] * H[1, 2], H[1, 1] - H[2, 1] * H[1, 2]]])

    v_vec = np.asarray([v[0], v[1], 1])

    c_theta_norm = np.linalg.norm(v_vec)
    cos_theta = 1 / c_theta_norm

    sin_theta = np.sqrt(1 - (1 / np.pow(c_theta_norm, 2)))

    test2 = cos_theta * cos_theta + sin_theta * sin_theta

    k_skew = (1 / np.linalg.norm(v)) * np.asarray([[0, 0, v[0]],
                                                          [0, 0, v[1]],
                                                          [-v[0], -v[1], 0]])

    R_v = np.identity(3) + sin_theta * k_skew + (1 - cos_theta) * np.matmul(k_skew, k_skew)


    B_mtx_pre = np.asarray([[1, 0, -v[0]], [0, 1, -v[1]]])

    B_pre = B_mtx_pre @ R_v

    A = np.linalg.solve(B_pre[:, 0:2], J)

    lambda_mtx = A @ A.T

    lambda_partial_val = np.pow(lambda_mtx[0, 0] - lambda_mtx[1, 1], 2) + 4 * lambda_mtx[1, 0] * lambda_mtx[1, 0]

    lambda_val = np.sqrt(0.5 * (lambda_mtx[0, 0] + lambda_mtx[1, 1] + np.sqrt(lambda_partial_val)))

    R_22 = (1 / lambda_val) * A

    U, S, V = np.linalg.svd(np.eye(2) - R_22.T @ R_22)

    b = U[:, 0]

    param_vec_cross = np.asarray([[R_22[0, 0], R_22[0, 1]],
                            [R_22[1, 0], R_22[1, 1]],
                            [b[0], b[1]]])

    distance = np.linalg.norm((1 / lambda_val) * v_vec)

    param_vec = np.linalg.cross(param_vec_cross[:, 0], param_vec_cross[:, 1])

    R_1 =  np.matmul(R_v, np.asarray([[R_22[0, 0], R_22[0, 1], param_vec[0]],
                                    [R_22[1, 0], R_22[1, 1], param_vec[1]],
                                    [b[0], b[1], param_vec[2]]]))

    R_2 = np.matmul(R_v, np.asarray([[R_22[0, 0], R_22[0, 1], -param_vec[0]],
                                     [R_22[1, 0], R_22[1, 1], -param_vec[1]],
                                     [-b[0], -b[1], param_vec[2]]]))

    print("Distance est is: ", distance)
    #print("Det is: ", np.linalg.det(R_v))











class SlotCarTracker:
    def __init__(self):
        # define a video capture object
        self.l_y = None
        self.l_x = None
        self.end_section_positions = None
        self.phi_vals = None
        self.y_vals = None
        self.x_vals = None
        self.carrera_track = None
        self.vid = cv2.VideoCapture(2)  # this is the magic!

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.frame = self.vid.read()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Create the ArUco detector
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.race_track_reference_points = np.empty([2, 0])

        self.camera_to_race_track = np.zeros([4, 4])

        self.aruco_markings = np.asarray(
            [[0.0265, -0.0265, 0], [0.0265, 0.0265, 0], [-0.0265, 0.0265, 0], [-0.0265, -0.0265, 0]])

        self.camera_mtx = np.asarray([[801.97181397, 0.00000000e+00, 642.52529483],
                                 [0.00000000e+00, 801.64865261, 365.19553175],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.distortion_mtx = np.asarray([[0.03649306, -0.06757704, -0.00136245, 0.0067907, 0.05633966]])

        self.slot_car_length = 0.1

        self.first_image = True

        self.curve_closed = False

        self.out_of_start = False

        self.start_radius = 0.2

        self.base_mtx = np.eye(4)

        self.track_detections = np.empty((3, 0))

        self.track_detections_params = np.empty(0)

        self.use_time_parametrisation = True


    def detect_aruco_marker_pose(self):
        # Capture the video frame
        # by frame
        ret, self.frame = self.vid.read()
        timestamp = time.time() #Seconds since epoch

        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        imgpoints = []

        # Display the resulting frame
        cv2.imshow('frame', self.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ywy")

        transformation_mtx = np.zeros([4, 4])
        got_detection = False

        # Detect the markers
        corners, ids, rejected = self.detector.detectMarkers(gray_image)
        if len(corners) != 0:
            got_detection = True
            corners2 = cv2.cornerSubPix(gray_image, corners[0], (11, 11), (-1, -1), self.criteria)

            ret, rvecs, tvecs = cv2.solvePnP(self.aruco_markings, corners2, self.camera_mtx, self.distortion_mtx)

            transformation_mtx = self.get_transformation_mtx(rvecs, tvecs)

            if self.first_image:
                self.base_mtx = transformation_mtx
                self.first_image = False
                self.test_pos = tvecs[:, 0]

        return got_detection, transformation_mtx, timestamp


    def get_aruco_centre_pos(self):
        # Capture the video frame
        # by frame
        ret, self.frame = self.vid.read()
        timestamp = time.time() #Seconds since epoch

        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        imgpoints = []
        centre_pos = np.empty([2,0])

        # Display the resulting frame
        cv2.imshow('frame', self.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ywy")

        transformation_mtx = np.zeros([4, 4])
        got_detection = False

        # Detect the markers
        test = np.zeros([2, 1])
        corners, ids, rejected = self.detector.detectMarkers(gray_image)
        if len(corners) != 0:
            got_detection = True
            corners2 = np.squeeze(cv2.cornerSubPix(gray_image, corners[0], (11, 11), (-1, -1), self.criteria)).T

            centre_pos = np.mean(np.asarray(corners2), axis=1)

            test[0] = centre_pos[0]
            test[1] = centre_pos[1]

        return got_detection, test


    def calculate_camera_to_race_track_transform(self):

        measured_track_x_pos = np.empty(0)
        measured_track_y_pos = np.empty(0)
        track_x_pos = np.empty(0)
        track_y_pos = np.empty(0)


        section_idx = 0
        for i in self.end_section_positions:
            not_finished = True
            num_tries = 0
            print("Section idx: ", section_idx)
            name = input("press when ready to take image, at pos: ")
            while not_finished or num_tries > 100:
                got_detection, centre_pos = self.get_aruco_centre_pos()
                num_tries = num_tries + 1



                if got_detection:
                    measured_track_x_pos = np.concatenate([measured_track_x_pos, centre_pos[0, :]])
                    measured_track_y_pos = np.concatenate([measured_track_y_pos, centre_pos[1, :]])
                    track_x_pos = np.concatenate([track_x_pos, [self.l_x(i)]])
                    track_y_pos = np.concatenate([track_y_pos, [self.l_y(i)]])
                    not_finished = False
                    print("Got detection at: ", centre_pos)
            section_idx = section_idx + 1
        measured_track_pos = np.vstack([measured_track_x_pos, measured_track_y_pos])
        track_pos = np.vstack([track_x_pos, track_y_pos, np.zeros(track_x_pos.size)])

        measured_track_pos = np.transpose(measured_track_pos.astype('float32'))
        track_pos = np.transpose(track_pos.astype('float32'))

        ret, rvecs, tvecs = cv2.solvePnP(track_pos, measured_track_pos, self.camera_mtx, self.distortion_mtx)

        self.camera_to_race_track = self.get_transformation_mtx(rvecs, tvecs)


    def calculate_camera_to_race_track_transform_test(self):

        track_pos = np.array([])
        measured_track_pos = np.array([])

        ret, rvecs, tvecs = cv2.solvePnP(track_pos, measured_track_pos, self.camera_mtx, self.distortion_mtx)

        self.camera_to_race_track = self.get_transformation_mtx(rvecs, tvecs)

    def get_transformation_mtx(self, rvecs, tvecs):

        transformation_mtx = np.zeros( [4, 4])

        rotationMtx = cv2.Rodrigues(rvecs)

        transformation_mtx[0:3, 0:3] = rotationMtx[0]
        transformation_mtx[3, 3] = 1
        transformation_mtx[0:3, 3] = tvecs[:, 0]

        return transformation_mtx


    def load_track(self, carrera_track_list):
        self.carrera_track = carrera_slot_car_track_spline_creator_class.CarreraTrack(carrera_track_list)

        x_vals, y_vals, phi_vals, end_section_positions = self.carrera_track.generate_track_spline(True)

        self.x_vals = x_vals
        self.y_vals = y_vals
        self.phi_vals = phi_vals
        self.end_section_positions = end_section_positions

        self.l_x = scipy.interpolate.CubicSpline(phi_vals, x_vals)

        self.l_y = scipy.interpolate.CubicSpline(phi_vals, y_vals)


    def update_base_mtx(self):

        self.first_image = True
        self.detect_aruco_marker_pose()

    def get_slot_car_measurement(self):

        got_detection, transform_mtx_meas, timestamp = self.detect_aruco_marker_pose()

        new_pos = np.zeros([4, 1])
        new_pos[0:3, 0] = [- self.slot_car_length, 0, 0]
        new_pos[3, 0] = 1
        delta_pos = self.base_mtx @ transform_mtx_inv(transform_mtx_meas) @ new_pos

        delta_rot = self.base_mtx[0:3, 0:3].T @ transform_mtx_meas[0:3, 0:3]

        r = Rotation.from_matrix(delta_rot)

        angles = r.as_euler("zyx", degrees=True)

        return delta_pos[0:2], angles[0], timestamp

    def detect_closed_loop_track(self, use_time_parametrisation, delta_param):

        self.use_time_parametrisation = use_time_parametrisation

        prev_timestamp = time.time()
        start_time = prev_timestamp
        prev_pos = np.asarray([0,0,0])
        tot_dist = 0
        add_detection = False
        param_val = 0

        while not(self.curve_closed):
            got_detection, transformation_mtx, timestamp = self.detect_aruco_marker_pose()

            if got_detection:
                if np.linalg.norm(prev_pos) == 0:
                    prev_pos = transformation_mtx[0:3, 3]

                delta_time = timestamp - prev_timestamp
                prev_timestamp = timestamp
                delta_dist = prev_pos - transformation_mtx[0:3, 3]
                dist_norm = np.linalg.norm(delta_dist, 2)

                test_pos = np.zeros([4, 1])
                test_pos[0:3, 0] = [0, 0, 0]
                test_pos[3, 0] = 1

                print("Dist norm is:", dist_norm)

                if not use_time_parametrisation and (dist_norm > delta_param):
                    add_detection = True
                    param_val = tot_dist + dist_norm
                    tot_dist = tot_dist + dist_norm

                if use_time_parametrisation and (delta_time > delta_param):
                    add_detection = True
                    param_val = timestamp - start_time

                if add_detection:
                    add_detection = False
                    prev_pos = transformation_mtx[0:3, 3]

                    new_detection = np.zeros([3, 1])
                    new_detection[0:3,0] = (transformation_mtx @ transform_mtx_inv(self.base_mtx) ) [0:3,3]

                    #new_pos = np.zeros([4, 1])
                    #ew_pos[0:3, 0] = [- self.slot_car_length, 0, 0][0:3, 3]
                    #new_pos[3, 0] = 1
                    #new_transform = self.base_mtx @ transform_mtx_inv(transformation_mtx)
                    #pos_meas = self.base_mtx @ transform_mtx_inv(transformation_mtx) @ new_pos

                    self.track_detections = np.append(self.track_detections, new_detection, axis=1)
                    self.track_detections_params = np.append(self.track_detections_params, param_val)

                    start_distance = np.linalg.norm(self.base_mtx[0:3, 3] - transformation_mtx[0:3, 3], 2)
                    print("Added pos, start distance: ", start_distance)
                    if start_distance > self.start_radius:
                        self.out_of_start = True

                    if self.out_of_start and (start_distance < self.start_radius * 0.5):
                        self.curve_closed = True

        plt.plot(self.track_detections[0,:], self.track_detections[1, :])

        spline_points = np.linspace(0, self.phi_vals[-1], 1000)

        plt.plot(self.l_x(spline_points), self.l_y(spline_points), lw=3)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.ylabel('track_detections')
        plt.show()



    def finnish_tracking(self):
        # Destroy all the windows
        cv2.destroyAllWindows()


if __name__ == "__main__":

    trackSectionList = []


    trackSectionList.append(0)
    trackSectionList.append(1)
    trackSectionList.append(1)
    trackSectionList.append(1)
    trackSectionList.append(0)
    trackSectionList.append(0)
    trackSectionList.append(0)
    trackSectionList.append(1)
    trackSectionList.append(1)
    trackSectionList.append(1)
    trackSectionList.append(0)
    trackSectionList.append(0)



    slotCarTracker = SlotCarTracker()

    slotCarTracker.load_track(trackSectionList)

    #slotCarTracker.detect_closed_loop_track(False, 0.05)


    #slotCarTracker.calculate_camera_to_race_track_transform_test() #Debug testing

    slotCarTracker.calculate_camera_to_race_track_transform() #cameraa code











