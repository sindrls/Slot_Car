# Slot_Car

This is the repo containing all slot car related code, including, simulator, symbolic Casadi slot car model,
analysis models and real time tracker and controller. It also contains tests of Model Predictive Control models, that is currently unfinnished.

The plan for the repo is to gradually improve tha usability of the code and include more functionality as we get feedback from users.


# slotcar_simulation.py
slotcar_simulation.py is a simulator using a hardcoded model as described in [], with slight mumerical modifications to prevent any singularities. The simulator is currently a simple first order runge kutta integrator, where the slot car, parametrised track and speed controller is har coded in the python script. To run a simulation simply rin the script:)

# main_pose_est.py

# main_calibration.py

# slotcar_symbolic_sideslip_simulation.py
slotcar_symbolic_sideslip_simulation.py was created to generate the phase plots in []. This is a simple tool to analyse the stability of the slot car in a corner. This code uses the casiadi symolic equation tools, which is not s trivial to include as regular pachages, as the solver is wAgain, all parameters are hardcoded into the script, so change oparameters in the script to test the phase plot for different values. simply run slotcar_symbolic_sideslip_simulation.py and the phase plot will pop up. 

To calibrate, run main_calibration.py
