from actuonix_driver import LinearDriver
import numpy as np
from time import sleep

da = LinearDriver(f"/dev/ttyACM0")

# PRESET POSITIONS
p = np.ones((8, 12)) * 0.0012
p[0, 3:7] = np.array([0.01, 0.01, 0.01, 0.01])
p[1, 3:7] = np.array([0.01, 0.02, 0.02, 0.02])
p[2, 3:7] = np.array([0.01, 0.02, 0.03, 0.03])
p[3, 3:7] = np.array([0.01, 0.02, 0.03, 0.04])

p[4, 3:7] = np.array([0.01, 0.02, 0.03, 0.04])
p[5, 3:7] = np.array([0.01, 0.02, 0.03, 0.03])
p[6, 3:7] = np.array([0.01, 0.02, 0.02, 0.02])
p[7, 3:7] = np.array([0.01, 0.01, 0.01, 0.01])


def print_posn():
    posn = da.get_joint_positions()
    print(posn[3:6])  # Motors 4-6


def retract():
    da.reset()
    print_posn()


retract()
# print_posn()
for i in range(0, 8): # LOOP THROUGH ALL PRESET POSITIONS
    duration = [1.0]
    print(f"i:{i}, {p[i,:]}")
    da.move_joint_position(p[i, :].reshape(1,12), duration)
    sleep(1)
    # print_posn()

# RESET TO FULLY RETRACTED ACTUATORS
retract()

da.close()
