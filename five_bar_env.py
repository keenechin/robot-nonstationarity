from five_bar import FiveBar
from realsense_camera import RealsenseCamera


class FiveBarEnv():
    def __init__(self):
        self.camera = RealsenseCamera()
        self.camera_feed = self.camera.get_feed()
        self.hardware = FiveBar()
        self.camera.process.start()

    def get_state(self):
        servo_state = self.hardware.get_pos()
        pendulum_state = self.camera_feed.get()
        state = [*servo_state, *pendulum_state]
        return state

    def __del__(self):
        self.camera.process.terminate()


if __name__ == "__main__":
    env = FiveBarEnv()

    env.hardware.test_motion()
