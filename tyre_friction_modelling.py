import numpy as np
from matplotlib import pyplot as plt

atan_slide_speed_tresh = 0.01
atan_slide_frict_coeff = 1

regular_slide_speed_tresh = 20
regular_slide_frict_coeff = 2
regular_sin_scaling_coeff = 1.8

data_points = 1000

test_velocities = np.linspace(-np.pi /3, np.pi /3, data_points)

atan_friction = np.zeros(data_points)
regular_friction = np.zeros(data_points)


def dynamic_atan_friction_model(vel):
    return atan_slide_frict_coeff * np.arctan((1 / (np.pi * atan_slide_speed_tresh)) * vel)


def dynamic_regularized_friction_model(vel):
    return regular_slide_frict_coeff * np.sin(
        regular_sin_scaling_coeff * np.arctan(regular_slide_speed_tresh * vel))


def lift_Cd(angle):
    return np.sin(0)


idx = 0
for vel in test_velocities:
    atan_friction[idx] = dynamic_atan_friction_model(vel)
    regular_friction[idx] = dynamic_regularized_friction_model(vel)
    idx = idx + 1

#plt.plot(test_velocities, atan_friction, 'b', label='atan friction')

plt.plot(test_velocities * (180 / np.pi), regular_friction, 'g', label='regular friction')

plt.legend(loc='best')

plt.xlabel('t')

plt.grid()

plt.show()
