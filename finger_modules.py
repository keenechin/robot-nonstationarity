from dynamixel_driver import dxl
import numpy as np
import time


class Fingers():
    def __init__(self, ids=[40, 41, 42, 43, 44, 45, 50], motor_type="X",
                 device="/dev/ttyUSB0", baudrate=57600, protocol=2):
        self.servos = dxl(motor_id=ids, motor_type=motor_type,
                          devicename=device, baudrate=baudrate, protocol=protocol)
        self.servos.open_port()
        self.mid = np.pi
        range = 2.75
        self.min = {"left": self.mid - range/2, "right": self.mid + range/2}
        self.max = {"left": self.mid + range/4, "right": self.mid - range/4}

        self.reset()

    def reset(self):
        default = ((self.min["left"], self.min["right"]),) * 3
        self.all_move(default)
        self.servos.set_des_pos([self.servos.motor_id[-1]], [self.mid])
        err_thresh = 0.05
        errs = np.array([np.inf] * 1)
        while np.any(errs > err_thresh):
            curr = self.get_pos()
            errs = np.abs(curr[-1] - np.pi)
        self.servos.engage_motor([50], False)

    def finger_delta(self, finger_num, dir):
        movements = {"up": np.array([0.1, -0.1]),
                     "down": np.array([-0.1, 0.1]),
                     "left": np.array([0.1, 0.1]),
                     "right": np.array([-0.1, -0.1])}
        assert dir in movements.keys()
        assert finger_num in [0, 1, 2]
        curr = self.get_pos()
        left = (finger_num)*2
        right = (finger_num)*2+1
        pos = np.array(curr[left:right+1])
        delta = movements[dir]
        new_pos = pos + delta
        self.finger_move(finger_num, new_pos)

    def finger_move(self, finger_num, pos, err_thresh=0.1):
        assert finger_num in [0, 1, 2]
        left = (finger_num)*2
        right = (finger_num)*2+1
        self.servos.set_des_pos(self.servos.motor_id[left:right+1], pos)
        errs = np.array([np.inf, np.inf])
        while np.any(errs > err_thresh):
            curr = self.get_pos()
            errs = np.abs(curr[left:right+1] - pos)

    def all_move(self, pos, err_thresh=0.1):
        for i in range(3):
            self.finger_move(i, pos[i])

    def get_pos(self):
        return self.servos.get_pos(self.servos.motor_id)


manipulator = Fingers()
for i in range(12):
    dir = input("Move left, right, up, or down:")
    manipulator.finger_delta(0, dir)
    manipulator.finger_delta(1, dir)
    manipulator.finger_delta(2, dir)
# i = 0
# while True:
#     pos = manipulator.get_pos()
#     print(pos)
#     trange = np.pi/4
#     manipulator.move(*[np.pi -trange/2 + trange * np.random.random_sample()]*6)
#     i = i + 1
    
