import pyrealsense2 as rs
import cv2
from multiprocessing import Process, Queue
import os
import signal
import time
import numpy as np
from queue import Empty


class RealsenseCamera():
    def __init__(self, mode="Tracking"):
        camera = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth)
        self.config.enable_stream(
            rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
        self.camera = camera
        self.feed = Queue()
        if mode == "Tracking":
            self.process = Process(target=self.track_points, args=(self.feed,))
        if mode == "Detection":
            self.process = Process(
                target=self.detect_points, args=(self.feed,))

    def start(self):
        self.process.start()

    def create_tracker(self, tracker_name):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        return OPENCV_OBJECT_TRACKERS[tracker_name]()

    def initialize_tracking(self, trackerType="mosse"):  # medianflow
        self.camera.start(self.config)
        bboxes = []
        multi_tracker = cv2.MultiTracker_create()
        for i in range(100):  # give camera time to autoadjust
            frame = self.get_frame()

        while True:
            init_window = 'MultiTracker selections (MAKE BIG BOXES)'
            bbox = cv2.selectROI(init_window, frame,
                                 fromCenter=False, showCrosshair=True)
            bboxes.append(bbox)
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == ord('q')):  # q is pressed
                break
        cv2.destroyWindow(init_window)

        bboxes = bboxes[:2]
        print(bboxes)
        for bbox in bboxes:
            multi_tracker.add(self.create_tracker(trackerType), frame, bbox)
        time.sleep(2)
        return bboxes, multi_tracker

    def get_frame(self):
        frames = self.camera.wait_for_frames()
        ir_frame = frames.get_infrared_frame(0)
        frame = np.asanyarray(ir_frame.get_data())
        return frame

    def initialize_detection(self):
        profile = self.camera.start(self.config)

        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, 1)
        init_window = "Draw box around relevant area"
        for i in range(100):
            frame = self.get_frame()
        cv2.imshow(init_window, frame)
        bbox = cv2.selectROI(init_window, frame,
                             fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(init_window)
        return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

    def rescale(self, image, scale):
        if len(image.shape) == 2:
            height, width = image.shape
        if len(image.shape) == 3:
            height, width, _ = image.shape
        new_height = int(np.around(scale * height))
        new_width = int(np.around(scale * width))
        scaled = cv2.resize(
            image, (new_width, new_height))
        return scaled

    def detect_points(self, queue, visualize=True):
        rect = self.initialize_detection()
        last_state = None
        while True:
            try:
                frame = self.get_frame()
                cropped = frame[rect[1]:rect[3], rect[0]:rect[2]]
                original = cropped
                
                brightness_thresh = (0.6 * np.mean(cropped[:]) + 0.4 * np.max(cropped[:]))
                _, cropped = cv2.threshold(
                    cropped, brightness_thresh, 255, cv2.THRESH_BINARY)

                cropped = 255-cropped
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = True
                params.minArea = 4
                params.maxArea = 400
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.filterByInertia = False
                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(cropped)

                if len(keypoints) == 2:
                    x1 = np.round(keypoints[0].pt[0], 1)
                    y1 = np.round(keypoints[0].pt[1], 1)
                    x2 = np.round(keypoints[1].pt[0], 1)
                    y2 = np.round(keypoints[1].pt[1], 1)
                    state = [x1, y1, x2, y2]

                    if last_state is not None:
                        last_x1 = last_state[0]
                        last_y1 = last_state[1]
                        cis_dist = ((x1 - last_x1)**2 + (y1 - last_y1)**2)**0.5
                        trans_dist = ((x2 - last_x1)**2 + (y2 - last_y1)**2)**0.5
                        if trans_dist < cis_dist:
                            print("Inverted.")
                            state = [x2, y2, x1, y1]

                    last_state = state
                    queue.put(state)

                    if visualize:
                        im_with_keypoints = cv2.drawKeypoints(original, keypoints, np.array(
                            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        scaled = self.rescale(im_with_keypoints, 4)
                        cv2.imshow('Live Feed', scaled)
                        k = cv2.waitKey(1)
                        if (k == ord('q')):  # q is pressed
                            raise KeyboardInterrupt
                else:
                    print("Detection Fail.")
            except KeyboardInterrupt:
                print("User exit during detection.")
                break

    def track_points(self, queue, visualize=True):
        bboxes, multi_tracker = self.initialize_tracking()
        while True:
            frame = self.get_frame()
            success, bboxes = multi_tracker.update(frame)
            if success:
                if not queue.empty():
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

                states = []
                for bbox in bboxes:
                    state = [int(bbox[0] + bbox[2]/2),
                             int(bbox[1] + bbox[3]/2)]
                    states = states + state

                queue.put(states)
            else:
                print("Tracking Fail.")
            if visualize:
                cv2.circle(img=frame, center=state, radius=5,
                           color=(0, 0, 200), thickness=-1)
                cv2.imshow('Live Feed', frame)
                k = cv2.waitKey(1)
                if (k == ord('q')):  # q is pressed
                    break

    def __del__(self):
        os.kill(self.process.pid, signal.SIGKILL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = RealsenseCamera(mode="Detection")
    camera.start()
    while True:
        try:
            print(camera.feed.get())
            pass
        except KeyboardInterrupt:
            print("\nUser exit main.")
            break
