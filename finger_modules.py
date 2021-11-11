from dynamixel_driver import dxl
import numpy as np


class Fingers():
    def __init__(self, ids=[40, 41, 42, 43, 44, 45, 50], motor_type="X",
                 device="/dev/ttyUSB0", baudrate=57600, protocol=2):
        self.servos = dxl(motor_id=ids, motor_type=motor_type,
                          devicename=device, baudrate=baudrate, protocol=protocol)
        self.servos.open_port()
        self.servos.engage_motor([50], False)

    def move(self, pos1, pos2, pos3, pos4, pos5, pos6, err_thresh=0.1):
        self.servos.set_des_pos(self.servos.motor_id[:-1],
                                [pos1, pos2, pos3, pos4, pos5, pos6])
        errs = np.array([np.inf] * 6)
        while np.any(errs > err_thresh):
            curr = self.get_pos()
            errs = np.abs(
                curr[:-1] - np.array([pos1, pos2, pos3, pos4, pos5, pos6]))

    def get_pos(self):
        return self.servos.get_pos(self.servos.motor_id)


manipulator = Fingers()
i = 0
while True:
    pos = manipulator.get_pos()
    print(pos)
    manipulator.move(*[np.pi + 0.2 * np.cos(np.pi * i/10)]*6)
    i = i + 1
    
