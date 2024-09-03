import numpy as np
from matplotlib import pyplot as plt

atan_slide_speed_tresh = 0.01
atan_slide_frict_coeff = 1

regular_slide_speed_tresh = 0.01
regular_slide_frict_coeff = 0.8
regular_sin_scaling_coeff = 1.9

data_points = 1000

test_velocities = np.linspace(-0, np.pi, data_points)

atan_friction = np.zeros(data_points)
regular_friction = np.zeros(data_points)


def dynamic_atan_friction_model(vel):
    return atan_slide_frict_coeff * np.arctan((1 / (np.pi * atan_slide_speed_tresh)) * vel)


def dynamic_regularized_friction_model(vel):
    return regular_slide_frict_coeff * np.sin(
        regular_sin_scaling_coeff * np.arctan((1 / (np.pi * regular_slide_speed_tresh)) * vel))


def lift_Cd(angle):
    return np.sin()


idx = 0
for vel in test_velocities:
    atan_friction[idx] = dynamic_atan_friction_model(vel)
    regular_friction[idx] = lift_Cd(vel)
    idx = idx + 1

#plt.plot(test_velocities, atan_friction, 'b', label='atan friction')

plt.plot(test_velocities, regular_friction, 'g', label='regular friction')

plt.legend(loc='best')

plt.xlabel('t')

plt.grid()

plt.show()
