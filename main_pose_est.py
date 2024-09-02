# import the opencv library
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
import math

from carrera_slot_car_track_spline_creator_class import CarreraTrack, Sections
import simple_kalman_tracker
from udp_sender import CarreraUDPSender
from utl import p_save, p_load



def transform_mtx_inv(mtx):
    transformation_inverse = np.zeros([4, 4])
    transformation_inverse[0:3, 0:3] = np.transpose(mtx[0:3, 0:3])
    transformation_inverse[0:3, 3] = -np.transpose(mtx[0:3, 0:3]) @ mtx[0:3, 3]
    transformation_inverse[3, 3] = 1

    return transformation_inverse


def speed_controller(l_x, l_y, phi, phi_dot, delta_phi, horizon, predict_t):
    V_max = 150
    K = 0.05
    p = 0.1

    d_phi_l_x = l_x.derivative()
    d_phi_l_y = l_y.derivative()

    dd_phi_l_x = d_phi_l_x.derivative()
    dd_phi_l_y = d_phi_l_y.derivative()

    exp_val = 0

    for i in range(horizon):
        phi_i = i * delta_phi + predict_t * phi_dot + phi
        d_phi_l_x_val = d_phi_l_x(phi_i)
        d_phi_l_y_val = d_phi_l_y(phi_i)

        dd_phi_l_x_val = dd_phi_l_x(phi_i)
        dd_phi_l_y_val = dd_phi_l_y(phi_i)

        kappa = abs((d_phi_l_x_val * dd_phi_l_y_val - dd_phi_l_x_val * d_phi_l_y_val))

        exp_val = exp_val + (p**i) * kappa

    return V_max * (math.e**(- K * exp_val))


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
        self.car_tracker = None
        self.vid = cv2.VideoCapture(0)  # this is the magic!

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.udpSender = CarreraUDPSender()

        self.frame = self.vid.read()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Create the ArUco detector
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.race_track_reference_points = np.empty([2, 0])

        self.camera_to_race_track = p_load("camera_to_race_track")
        self.aruco_markings = p_load("aruco_markings")
        self.camera_mtx = p_load("camera_mtx")
        self.distortion_mtx = p_load("distortion_mtx")

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
        timestamp = time.time()  # Seconds since epoch

        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        #cv2.imshow('frame', self.frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    print("ywy")

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

        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        centre_pos = np.empty([2, 0])
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
            for j in range(30):
                ret, self.frame = self.vid.read()
            while not_finished and num_tries < 10:
                got_detection, centre_pos = self.get_aruco_centre_pos()
                num_tries = num_tries + 1
                print("New try: ", num_tries)

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

        print("Camera to track transform is: ", self.camera_to_race_track)
        print("Camera to track translation is: ", tvecs)


    def get_transformation_mtx(self, rvecs, tvecs):

        transformation_mtx = np.zeros([4, 4])

        rotationMtx = cv2.Rodrigues(rvecs)

        transformation_mtx[0:3, 0:3] = rotationMtx[0]
        transformation_mtx[3, 3] = 1
        transformation_mtx[0:3, 3] = tvecs[:, 0]

        return transformation_mtx


    def load_track(self, carrera_track_list):
        self.carrera_track = CarreraTrack(carrera_track_list)

        x_vals, y_vals, phi_vals, end_section_positions = self.carrera_track.generate_track_spline(True)

        self.x_vals = x_vals
        self.y_vals = y_vals
        self.phi_vals = phi_vals
        self.end_section_positions = end_section_positions

        self.l_x = scipy.interpolate.CubicSpline(phi_vals, x_vals)

        self.l_y = scipy.interpolate.CubicSpline(phi_vals, y_vals)

        self.car_tracker = simple_kalman_tracker.SlotCarKalmanTracker(carrera_track_list)


    def update_base_mtx(self):

        self.first_image = True
        self.detect_aruco_marker_pose()

    def get_slot_car_measurement(self):

        got_detection, transform_mtx_meas, timestamp = self.detect_aruco_marker_pose()

        new_pos = np.zeros([4, 1])
        new_pos[0:3, 0] = transform_mtx_meas[0:3, 3]
        new_pos[3, 0] = 1
        delta_pos = transform_mtx_inv(self.camera_to_race_track) @ new_pos

        print("meas_pos is: ", new_pos)

        #delta_rot = self.base_mtx[0:3, 0:3].T @ transform_mtx_meas[0:3, 0:3]

        #r = Rotation.from_matrix(delta_rot)

        #angles = r.as_euler("zyx", degrees=True)

        return delta_pos[0:2],  # angles[0], timestamp

    def detect_closed_loop_track(self, use_time_parametrisation, delta_param):

        self.use_time_parametrisation = use_time_parametrisation

        prev_timestamp = time.time()
        start_time = prev_timestamp
        prev_pos = np.asarray([0, 0, 0])
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
                    new_detection[0:3, 0] = (transformation_mtx @ transform_mtx_inv(self.base_mtx))[0:3, 3]

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

        plt.plot(self.track_detections[0, :], self.track_detections[1, :])

        spline_points = np.linspace(0, self.phi_vals[-1], 1000)

        plt.plot(self.l_x(spline_points), self.l_y(spline_points), lw=3)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.ylabel('track_detections')
        plt.show()

    def get_pixel_to_track_plane_intersection(self, pixel_position):

        pixels = np.zeros((1, 1, 2))

        pixels[0, 0, 0] = pixel_position[0, 0]
        pixels[0, 0, 1] = pixel_position[0, 1]

        direction_vector = np.squeeze(cv2.undistortPoints(pixels, self.camera_mtx, self.distortion_mtx))

        pixel_direction = np.asarray([direction_vector[0], direction_vector[1], 1])

        track_plane_norm_vec = self.camera_to_race_track[2, 0:3]
        track_plane_centre = self.camera_to_race_track[0:3, 3]

        intersection_distance = np.dot(track_plane_centre, track_plane_norm_vec) / np.dot(pixel_direction, track_plane_norm_vec)

        intersection_pos = pixel_direction * intersection_distance

        new_pos = np.asarray([intersection_pos[0], intersection_pos[1], intersection_pos[2], 1])

        return (transform_mtx_inv(self.camera_to_race_track) @ new_pos)[0:2]

    def get_xy_position(self):
        while True:
            pixel_position = self.get_aruco_centre_pos()
            track_position = self.get_pixel_to_track_plane_intersection(pixel_position)
            print("Position is: ", track_position)

    def finnish_tracking(self):
        # Destroy all the windows
        cv2.destroyAllWindows()

    def get_moving_objects(self):
        backSub = cv2.createBackgroundSubtractorMOG2()
        prev_timestamp = time.time()
        while True:
            new_timestamp = time.time()
            time_step = new_timestamp - prev_timestamp
            prev_timestamp = new_timestamp

            self.car_tracker.predict_state(time_step)

            ret, self.frame = self.vid.read()
            if ret:
                fg_mask = backSub.apply(self.frame)
                contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                retval, mask_thresh = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)

                min_contour_area = 2000  # Define your minimum area threshold
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

                frame_out = self.frame.copy()
                for cnt in large_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    frame_out = cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 200), 3)
                    centre_pos = np.zeros((1, 2))
                    centre_pos[0, 0] = x + w / 2.0
                    centre_pos[0, 1] = y + h / 2.0
                    pos = self.get_pixel_to_track_plane_intersection(centre_pos)

                    R = np.zeros((2, 2))
                    R[0, 0] = 0.5
                    R[1, 1] = 0.5

                    self.car_tracker.measurement_update(pos, R)
                    #print("Car pos is: ", pos)
                    car_state = self.car_tracker.get_state()
                    print("Car state is: ", car_state)

                    v_ref = speed_controller(self.l_x, self.l_y, car_state[0], car_state[1], 0.5, 10, 0.4)
                    print("V_ref is: ", v_ref)

                    self.udpSender.send(int(v_ref))

                # Display the resulting frame
                cv2.imshow('Frame_final', frame_out)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":

    trackSectionList = [
        Sections.CW,
        Sections.CW,
        Sections.CW,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.CW,
        Sections.CW,
        Sections.CW,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
        Sections.STRAIGHT,
    ]

    slotCarTracker = SlotCarTracker()

    slotCarTracker.load_track(trackSectionList)

    slotCarTracker.get_moving_objects()
