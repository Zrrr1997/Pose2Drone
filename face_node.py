from std_msgs.msg import String
import threading
import socket
import cv2
import sys

# ROS
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.msg import Image as ros_image
from geometry_msgs.msg import Pose
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from matplotlib import pyplot as plt
from recognition import *

import argparse
import logging
import time
import math

import numpy as np
import face_recognition
import os


class Face_Recognition:

	def __init__(self, width=640, height=480):
		self._running = True
		self.rgb_topic = 'rgb'
		self.recognition_topic = 'face_recogn'
		self.pub = rospy.Publisher(self.recognition_topic, String, queue_size=100)
		self.hz = 100

		(self.known_face_encodings, self.known_face_names) = get_known_face_encodings_and_names()

		self.w = width
		self.h = height

	def start_subscriber(self):
		rospy.init_node('face_recogn', anonymous=True)
		rate = rospy.Rate(self.hz)
		rgb_sub = rospy.Subscriber(self.rgb_topic, numpy_msg(Floats), self.receive_pose, queue_size = 100)
		print("RGB subscriber successfully started!")
		rospy.spin()
		print("Subscriber is dead...")

	def receive_pose(self, data):
		stream_data = np.array(data.data, dtype=np.float32)
		rgb_data = stream_data[:self.h * self.w *3].reshape(self.h, self.w, 3)
		rgb = np.array(rgb_data, dtype=np.uint8)
		self.recognize_faces(rgb)

	''' Face recognition '''
	def recognize_faces(self, frame):
		recognize_face(frame, self.known_face_encodings, self.known_face_names)


width = int(sys.argv[1])
height = int(sys.argv[2])

face_recogn = Face_Recognition(width, height)
face_recogn.start_subscriber()
