# Slot_Car

This is the repo containing all slot car related code, including, simulator, symbolic Casadi slot car model,
analysis models and real time tracker and controller. It also contains tests of Model Predictive Control models, that is currently unfinished.

The plan for the repo is to gradually improve the usability of the code and include more functionality as we get feedback from users.


# slotcar_simulation.py
slotcar_simulation.py is a simulator using a hard coded model as described in [], with slight numerical modifications to prevent any singularities. The simulator is currently a simple first order Runge Kutta integrator, where the slot car, parameterized track and speed controller is hard coded in the Python script. To run a simulation, simply run the script. :)

# main_pose_est.py
main_pose_est is the main pose estimation routine. This script uses the camera and pre-defined track model to estimate the slot car's position in the track.

# main_calibration.py
The main calibration routine uses OpenCV's calibrateCamera() to generate the camera matrix and distortion matrix for the camera. This routine only needs to be run once per camera, given that the camera's resolution and other parameters are constant. Changes in the camera's position and orientation do not require a recalibration. Read more about camera calibration here: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html 

# slotcar_symbolic_sideslip_simulation.py
slotcar_symbolic_sideslip_simulation.py was created to generate the phase plots in []. This is a simple tool to analyse the stability of the slot car in a corner. This code uses the Casadi symbolic equation tools, which is not a trivial to include as regular packages, as the solver is wAgain, all parameters are hard coded into the script, so change parameters in the script to test the phase plot for different values. Simply run slotcar_symbolic_sideslip_simulation.py and the phase plot will pop up. 

To calibrate, run main_calibration.py
