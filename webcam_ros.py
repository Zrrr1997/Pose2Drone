# ROS
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.msg import Image as ros_image
from geometry_msgs.msg import Pose
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import cv2
import numpy as np
import threading
import time

pub = rospy.Publisher('rgb', numpy_msg(Floats), queue_size=100)
rospy.init_node('webcam_stream', anonymous=True)
cap = cv2.VideoCapture(0)
running = True

def stream():
	counter = 0
	global running
	while running:
		ret, frame = cap.read()
		if ret:
			pub.publish(np.array(frame.flatten(), dtype=np.float32))
			print(counter + 1)
			counter += 1
			time.sleep(1.0 / 4.0) # Delay for debugging

def terminate():
	global running
	while running:
		inp = input()
		if inp == 'stop':
			running = False
			print("Stopping webcam...")

stream_p = threading.Thread(target=stream)
stream_p.start()

term_p = threading.Thread(target=terminate)
term_p.start()

term_p.join()
stream_p.join()


        
