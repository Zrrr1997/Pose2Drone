from tello_command import Tello
import pygame
from pygame.locals import *
import numpy as np
import time
import threading
from pid import PID

# ROS
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.msg import Image as ros_image
from geometry_msgs.msg import Pose
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String


def get_dist(x, y):
    dist = np.sqrt((np.square(np.array([x, y]) - np.array([240, 180]))).sum())
    return dist


def round_to_80(x: int):
    if x > 80:
        return 80
    if x < -80:
        return -80
    return x


class MovementControl:
    """
    Start face bbox:
        Input: Face bounding boxes from openpose
        Output: Actions(List) of drone keep face in the center of frame


    Start Gesture:
        Input: Gesture from gesture node
        Output: gesture(String)


    Maintains the Tello display and moves it through the keyboard keys.
    Press escape key to quit.
    The controls are:
        - T: Takeoff
        - L: Land
        - Space:    Emergency Shutdown
        - Arrow keys: Forward, backward, left and right.
        - A and D: Counter clockwise and clockwise rotations
        - W and S: Up and down.
        - K_1: Enter/Exit Face follow mode
        - K_2: Enter/Exit Gesture control mode
    """

    def __init__(self):
        self.box_centers = []
        self.gesture = None
        self.gesture_side = None
        self.received_face_bbox = []
        self.received_distance = 200

        self.face_bbox_topic = 'face_bbox'

        self.gesture_topic = 'gesture'

        self.dist_topic = 'distance'

        self.gesture_topic_side = 'gesture_side'

        self.movement_control_topic = 'movement'
        self.pub_movement_control = rospy.Publisher(self.movement_control_topic, numpy_msg(Floats), queue_size=100)

        self.hz = 100

        rospy.init_node('movement_control', anonymous=True)

        # Init pygame
        pygame.init()

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        self.tello.send('command')
        self.tello.send('streamon')

        # general config
        self.S = 40
        self.FPS = 20
        self.hud_size = (480, 360)

        # other params (no need to config)
        self.mid_x = int(self.hud_size[0] / 2)
        self.mid_y = int(self.hud_size[1] / 2)
        self.x_offset = 0
        self.y_offset = 0
        self.target_distance = 200
        self.h_pid = PID(0.2, 0.00001, 0.01)
        self.v_pid = PID(0.4, 0.001, 0.01)
        self.dist_pid = PID(0.4, 0.001, 0.01)

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

        self.send_rc_control = False
        self.Face_running = False

        self.gesture_cnt = 0
        self.Gesture_running = False

        self.box_cnt = 0
        self.last_yaw = 0
        self.last_up_down_velocity = 0
        self.last_for_back_velocity = 0
        # Creat pygame window
        pygame.display.set_caption("Tello Control")
        self.screen = pygame.display.set_mode(self.hud_size)

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

        print('init done')

    def start_face_tracking(self):
        rospy.init_node('movement_control', anonymous=True)
        rate = rospy.Rate(self.hz)
        face_bbox_sub = rospy.Subscriber(self.face_bbox_topic, numpy_msg(Floats), self.receive_face_bbox,
                                         queue_size=100)
        print("Face BBox subscriber successfully started!")
        face_dist_sub = rospy.Subscriber(self.dist_topic, numpy_msg(Floats), self.receive_distance,
                                         queue_size=100)
        print("Face Distance subscriber successfully started!")

    def receive_face_bbox(self, data):
        face_bboxes = np.copy(data.data)
        self.received_face_bbox = face_bboxes.reshape(-1, 4)

        if len(self.received_face_bbox) > 0:
            bbox = self.received_face_bbox[0]
            # calculate center of bounding box
            bbox_x = int((bbox[2] - bbox[0]) / 2 + bbox[0])
            bbox_y = int((bbox[3] - bbox[1]) / 2 + bbox[1]) - 70
            self.box_centers = [bbox_x, bbox_y]

    def receive_distance(self, data):
        self.received_distance = data.data[0]

    def start_gesture(self):
        rospy.init_node('movement_control', anonymous=True)
        rate = rospy.Rate(self.hz)
        gesture_sub = rospy.Subscriber(self.gesture_topic, String, self.receive_gesture, queue_size=100)
        print("Gesture subscriber successfully started!")
        gesture_side_sub = rospy.Subscriber(self.gesture_topic_side, String, self.receive_gesture_side, queue_size=100)
        print("Gesture Side subscriber successfully started!")

    def receive_gesture(self, data):
        self.gesture = data.data

    def receive_gesture_side(self, data):
        self.gesture_side = data.data

    def run(self):
        print('front to end started!')
        should_stop = False
        while not should_stop:

            if self.Face_running:
                self.face_follow()

            if self.Gesture_running:
                self.gesture_control()
                time.sleep(0.5)

            # handle input from drone or user
            for event in pygame.event.get():
                if event.type == USEREVENT + 1 and not self.Gesture_running:
                    self.send_input()
                elif event.type == QUIT:
                    self.should_stop = True
                elif event.type == KEYDOWN:
                    if (event.key == K_ESCAPE) or (event.key == K_BACKSPACE):
                        self.should_stop = True
                        pygame.quit()
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)

            # wait a little
            time.sleep(1 / self.FPS)

        # always call before finishing to deallocate resources
        self.tello.send('streamoff')

        pygame.quit()

    def face_follow(self):
        if len(self.box_centers) > 0 and 200 > get_dist(self.box_centers[0], self.box_centers[1]) > 30:
            self.box_cnt = 0
            self.last_yaw = 0
            self.last_up_down_velocity = 0

            print('center distance:{}'.format(get_dist(self.box_centers[0], self.box_centers[1])))
            print('---------------')
            # distance
            dist_error = self.target_distance - self.received_distance
            dist_control = self.dist_pid.control(dist_error)
            print('z position:{}'.format(self.received_distance))
            print('z target:{}'.format(self.target_distance))
            print('z offset:{}'.format(dist_error))
            print('z speed:{}'.format(-dist_control))
            print('---------------')

            # rotation
            self.x_offset = self.box_centers[0] - self.mid_x
            h_control = self.h_pid.control(self.x_offset)
            print('x position:{}'.format(self.box_centers[0]))
            print('x target:{}'.format(self.mid_x))
            print('x offset:{}'.format(self.x_offset))
            print('x speed:{}'.format(h_control))
            print('---------------')

            # up an down
            self.y_offset = self.mid_y - self.box_centers[1]
            v_control = self.v_pid.control(self.y_offset)
            print('y position:{}'.format(self.box_centers[1]))
            print('y target:{}'.format(self.mid_y))
            print('y offset:{}'.format(self.y_offset))
            print('y speed:{}'.format(v_control))
            print('---------------')

            #self.for_back_velocity = 0
            self.for_back_velocity = -int(round_to_80(dist_control))
            self.yaw_velocity = int(round_to_80(h_control))
            self.up_down_velocity = int(round_to_80(v_control))
            self.left_right_velocity = 0

            self.last_yaw = self.yaw_velocity
            self.last_up_down_velocity = self.up_down_velocity
            self.last_for_back_velocity = self.for_back_velocity

        if len(self.box_centers) > 0 and get_dist(self.box_centers[0], self.box_centers[1]) < 30:
            self.box_cnt = 0
            self.up_down_velocity = 0
            self.yaw_velocity = 0
            self.for_back_velocity = 0
            self.left_right_velocity = 0

        if len(self.box_centers) > 0 and get_dist(self.box_centers[0], self.box_centers[1]) > 230:
            self.box_centers = []

        else:
            self.box_cnt += 1
            self.h_pid.reset()
            self.v_pid.reset()
            self.dist_pid.reset()
            if self.box_cnt > 10:
                self.yaw_velocity = self.last_yaw
                self.up_down_velocity = self.last_up_down_velocity
                self.for_back_velocity = self.last_for_back_velocity
            else:
                self.left_right_velocity = 0

    def gesture_control(self):
        if not (self.gesture is None):
            if self.gesture == 'Up':
                self.tello.send('up 40')
                time.sleep(3)
            elif self.gesture == 'Down':
                self.tello.send('down 40')
                time.sleep(3)
            elif self.gesture == 'Right':
                self.tello.send('left 40')
                time.sleep(3)
            elif self.gesture == 'Left':
                self.tello.send('right 40')
                time.sleep(3)
            elif self.gesture == 'Cw':
                self.tello.send('cw 90')
                time.sleep(3)
            elif self.gesture == 'Ccw':
                self.tello.send('ccw 90')
                time.sleep(3)
            elif self.gesture == 'Forward':
                self.tello.send('forward 40')
                time.sleep(3)
            elif self.gesture == 'Back':
                self.tello.send('back 40')
                time.sleep(3)
            print('gesture control on : {} !'.format(self.gesture))
            self.gesture = None
        if self.gesture_side == 'Side right':
            x_1 = int(self.received_distance * 0.3)
            X_2 = int(self.received_distance)
            y_1 = int(-self.received_distance * 0.7)
            y_2 = int(-self.received_distance)
            self.tello.send('curve {} {} {} {} {} {} 30'.format(x_1, y_1, 0, X_2, y_2, 0))
            time.sleep(7)
            self.tello.send('ccw 90')
            print('move from side to front')
            self.gesture_side = 'None'
        if self.gesture_side == 'Side left':
            x_1 = int(self.received_distance * 0.3)
            X_2 = int(self.received_distance)
            y_1 = int(self.received_distance * 0.7)
            y_2 = int(self.received_distance)
            self.tello.send('curve {} {} {} {} {} {} 30'.format(x_1, y_1, 0, X_2, y_2, 0))
            time.sleep(7)
            self.tello.send('cw 90')
            print('move from side to front')
            self.gesture_side = 'None'
        else:
            print('gesture control on : {} !'.format(self.gesture))
            pass

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.Face_running = False
            self.Gesture_running = False
            self.for_back_velocity = self.S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.Face_running = False
            self.Gesture_running = False
            self.for_back_velocity = -self.S
        elif key == pygame.K_LEFT:  # set left velocity
            self.Face_running = False
            self.Gesture_running = False
            self.left_right_velocity = -self.S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.Face_running = False
            self.Gesture_running = False
            self.left_right_velocity = self.S
        elif key == pygame.K_w:  # set up velocity
            self.Face_running = False
            self.Gesture_running = False
            self.up_down_velocity = self.S
        elif key == pygame.K_s:  # set down velocity
            self.Face_running = False
            self.Gesture_running = False
            self.up_down_velocity = -self.S
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.Face_running = False
            self.Gesture_running = False
            self.yaw_velocity = self.S
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.Face_running = False
            self.Gesture_running = False
            self.yaw_velocity = -self.S
        elif key == pygame.K_t:  # takeoff
            self.tello.send('takeoff')
            self.send_rc_control = True
            self.left_right_velocity = 0
            self.for_back_velocity = 0
            self.yaw_velocity = 0
            self.up_down_velocity = 0
        elif key == pygame.K_l:  # land
            self.Face_running = False
            self.Gesture_running = False
            self.tello.send('land')
            self.send_rc_control = False
            self.left_right_velocity = 0
            self.for_back_velocity = 0
            self.yaw_velocity = 0
            self.up_down_velocity = 0
        elif key == pygame.K_SPACE:  # emergency shutdown
            self.Face_running = False
            self.Gesture_running = False
            self.tello.send('emergency')
            self.send_rc_control = False
            self.should_stop = True
        elif key == pygame.K_BACKSPACE:  # emergency shutdown
            self.Face_running = False
            self.Gesture_running = False
            self.send_rc_control = False
            self.should_stop = True
        elif key == pygame.K_1:  # Face follow
            self.Face_running = not self.Face_running
            self.Gesture_running = False
            self.left_right_velocity = 0
            self.for_back_velocity = 0
            self.yaw_velocity = 0
            self.up_down_velocity = 0
        elif key == pygame.K_2:  # Gesture control
            self.Gesture_running = not self.Gesture_running
            self.Face_running = False
            self.send_rc_control = True
            self.left_right_velocity = 0
            self.for_back_velocity = 0
            self.yaw_velocity = 0
            self.up_down_velocity = 0

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0

    def send_input(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send(
                'rc {a} {b} {c} {d}'.format(a=self.left_right_velocity,b=self.for_back_velocity,c=self.up_down_velocity,
                                            d=self.yaw_velocity))


def thread_job():
    rospy.spin()


# start thread
m = MovementControl()

face_bbox_p = threading.Thread(target=m.start_face_tracking())
add_thread_face = threading.Thread(target=thread_job)

gesture_p = threading.Thread(target=m.start_gesture())
add_thread_gesture = threading.Thread(target=thread_job)

front_end = threading.Thread(target=m.run())

face_bbox_p.start()
add_thread_face.start()

gesture_p.start()
add_thread_gesture.start()

front_end.start()
