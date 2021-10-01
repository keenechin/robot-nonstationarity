from actuonix_driver import LinearDriver
from dynamixel_py import dxl
import numpy as np
import time
import click


class FiveBar():
    def __init__(self, ids, motor_type, device, baudrate, protocol):
        self.servos = dxl(motor_id=ids, motor_type=motor_type,
                          devicename=device, baudrate=baudrate, protocol=protocol)
        self.servos.open_port()
        self.theta_range = 2*np.pi*(0.25)
        self.theta1_mid = 2*np.pi*(0.5+0.06)
        self.theta2_mid = 2*np.pi*(0.5-0.06)
        self.theta1_min = self.theta1_mid - self.theta_range/2
        self.theta1_max = self.theta1_mid + self.theta_range/2
        self.theta2_min = self.theta2_mid - self.theta_range/2
        self.theta2_max = self.theta2_mid + self.theta_range/2

        self.linear = LinearDriver()
        self.linear_min = 0.01
        self.linear_zeros = np.ones((1, 12))*self.linear_min
        self.linear_max = 0.0375

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

    def move_abs(self, pos1, pos2, thresh=0.01):
        print(f"Raw val: {pos1:.3f},{pos2:.3f}")
        pos1 = np.clip(pos1, self.theta1_min, self.theta1_max)
        pos2 = np.clip(pos2, self.theta2_min, self.theta2_max)
        print(f"Bounded: {pos1:.3f},{pos2:.3f}")
        self.servos.set_des_pos(self.servos.motor_id, [pos1, pos2])
        err1 = np.inf
        err2 = np.inf
        while err1 > thresh or err2 > thresh:
            curr = self.get_pos()
            err1 = np.abs(curr[0]-pos1)
            err2 = np.abs(curr[1]-pos2)
            time.sleep(0.001)

    def move_delta(self, delta1, delta2):
        curr = self.get_pos()
        # new_pos1 = curr1 + delta1
        # new_pos2 = curr2 + delta2

        self.move_abs(curr[0]+delta1, curr[1]+delta2)

    def get_pos(self):
        curr = self.servos.get_pos(self.servos.motor_id)
        return curr

    def primitive(self, idx, mag=0.2*np.pi):
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
        self.trajectory([5, 5, 5, 1, 1, 1, 1, 9, 9, 9, 9])
        self.drift(self.linear_max)
        # self.trajectory([1, 9, 2, 8, 3, 7, 4, 6, 5])
        self.trajectory([2, 8, 8, 2, 2, 8])
        self.trajectory([4, 6, 6, 4, 4, 6])
        self.drift(self.linear_min)
        # self.trajectory([1, 3, 9, 7]*2)
        self.trajectory([2, 8, 8, 2, 2, 8])
        self.trajectory([4, 6, 6, 4, 4, 6])
        self.reset()


DESC = ''' 
USAGE:
python five_bar.py --motor_id "[1,2]" --motor_type "X" --baudrate 1000000 --device /dev/ttyUSB0 --protocol 2
'''


@click.command(help=DESC)
@click.option('--motor_id', '-i', type=str, help='motor ids', default="[1, 2]")
@click.option('--motor_type', '-t', type=str, help='motor type', default="X")
@click.option('--baudrate', '-b', type=int, help='port baud rate', default=1000000)
@click.option('--device', '-n', type=str, help='port name', default="/dev/ttyUSB0")
@click.option('--protocol', '-p', type=int, help='communication protocol 1/2', default=2)
def main(motor_id, motor_type, device, baudrate, protocol):
    ids = eval(motor_id)
    robot = FiveBar(ids, motor_type, device, baudrate, protocol)
    robot.test_motion()


if __name__ == "__main__":
    main()
