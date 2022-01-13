import threading
import socket
import cv2

# ROS
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.msg import Image as ros_image
from geometry_msgs.msg import Pose
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import time
import numpy as np

class Tello:
    def __init__(self):
        self._running = True
        self.video = cv2.VideoCapture("udp://@0.0.0.0:11111")

        self.rgb_topic = 'rgb'
        self.pub = rospy.Publisher(self.rgb_topic, numpy_msg(Floats), queue_size=100)
        rospy.init_node('tello_RGB_stream', anonymous=True)


    def recv(self):
        """ Handler for Tello states message """
        while self._running:
            try:
                ret, frame = self.video.read()
                if ret:
                    # Resize frame
                    height, width, _ = frame.shape
                    new_h = int(height) # dont reshape?
                    new_w = int(width) # dont reshape?

                    # Resize for improved performance
                    new_frame = cv2.resize(frame, (new_w, new_h))
                    print("Tello frame shape: ", new_frame.shape)
                    self.pub.publish(np.array(new_frame.flatten(), dtype=np.float32))

                    #cv2.imwrite('demo/im' + str(time.time()) + '.png', new_frame)
            except Exception as err:
                print(err)
        exit()
    def terminate(self):
        while self._running:
                inp = input()
                if inp == 'stop':
                        self._running = False
                        print("Stopping drone cam...")

""" Start new thread for receive Tello response message """
t = Tello()
recvThread = threading.Thread(target=t.recv)
recvThread.start()

term_p = threading.Thread(target=t.terminate)
term_p.start()

recvThread.join()
term_p.join()

