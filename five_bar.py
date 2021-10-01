from actuonix_driver import LinearDriver
from dynamixel_driver import DynamixelDriver
import numpy as np
import time


class FiveBar():
    def __init__(self):
        self.servos = DynamixelDriver(num_servos=2)
        self.linear = LinearDriver()
        self.linear_min = 0.01
        self.linear_zeros = np.ones((1, 12))*self.linear_min
        self.linear_max = 0.0375
        self.theta_range = 0.25
        self.theta1_mid = 0.5+0.06
        self.theta2_mid = 0.5-0.06
        self.theta1_min = self.theta1_mid - self.theta_range/2
        self.theta1_max = self.theta1_mid + self.theta_range/2
        self.theta2_min = self.theta2_mid - self.theta_range/2
        self.theta2_max = self.theta2_mid + self.theta_range/2
        self.reset()
    
    def reset(self):
        self.linear.move_joint_position(self.linear_zeros, 1.0)
        self.move_abs(self.theta1_mid, self.theta2_mid)
        time.sleep(0.1)

    def drift(self, pos):
        assert pos >= self.linear_min
        assert pos <= self.linear_max
        p = self.linear_zeros
        p[0, 3:7] = pos
        self.linear.move_joint_position(p, 1.0)
        time.sleep(0.01)
        # self.linear.wait_until_done_moving()

    def move_abs(self, pos1, pos2):
        print(f"Raw val: {pos1:.3f},{pos2:.3f}")
        pos1 = max(pos1, self.theta1_min)
        pos1 = min(pos1, self.theta1_max)
        pos2 = max(pos2, self.theta2_min)
        pos2 = min(pos2, self.theta2_max)
        print(f"Bounded: {pos1:.3f},{pos2:.3f}")
        assert pos1 >= self.theta1_min
        assert pos2 >= self.theta2_min
        assert pos1 <= self.theta1_max
        assert pos2 <= self.theta2_max
        self.servos.servo_move([int(pos1 * self.servos.DXL_RANGE + self.servos.DXL_MINIMUM_POSITION_VALUE),
                                int(pos2 * self.servos.DXL_RANGE + self.servos.DXL_MINIMUM_POSITION_VALUE)])

    def move_delta(self, delta1, delta2):
        curr1, curr2 = self.get_pose()
        new_pos1 = curr1 + delta1
        new_pos2 = curr2 + delta2
        self.move_abs(new_pos1, new_pos2)

    def get_pose(self):
        _, _, curr1 = self.servos.get_pos(self.servos.DXL_IDS[0])
        _, _, curr2 = self.servos.get_pos(self.servos.DXL_IDS[1])
        curr1 = (curr1 - self.servos.DXL_MINIMUM_POSITION_VALUE)/self.servos.DXL_RANGE
        curr2 = (curr2 - self.servos.DXL_MINIMUM_POSITION_VALUE)/self.servos.DXL_RANGE
        return curr1, curr2

    def primitive(self, idx, mag=0.1):
        if idx == 1:
            self.move_delta(-mag, -mag)
        if idx == 2:
            self.move_delta(-mag, 0.0)
        if idx == 3:
            self.move_delta(-mag, mag)
        if idx == 4:
            self.move_delta(0.0, -mag)
        if idx == 5:
            self.move_delta(0.0, 0.0)
        if idx == 6:
            self.move_delta(0.0, mag)
        if idx == 7:
            self.move_delta(mag, -mag)
        if idx == 8:
            self.move_delta(mag, 0.0)
        if idx == 9:
            self.move_delta(mag, mag)

    def trajectory(self, primitive_list):
        for idx in primitive_list:
            self.primitive(idx)

    def test_motion(self):
        self.drift(robot.linear_max)
        # self.trajectory([1, 9, 2, 8, 3, 7, 4, 6, 5])
        self.trajectory([2, 8, 8, 2, 2, 8])
        self.trajectory([4, 6, 6, 4, 4, 6])
        self.drift(robot.linear_min)
        # self.trajectory([1, 3, 9, 7]*2)
        self.trajectory([2, 8, 8, 2, 2, 8])
        self.trajectory([4, 6, 6, 4, 4, 6])
        self.reset()


if __name__ == "__main__":
    robot = FiveBar()
    robot.test_motion()
