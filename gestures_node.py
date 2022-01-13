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

import argparse
import logging
import time
import math

import numpy as np
from collections import Counter

class Gestures:
	def __init__(self, width, height, FPS):
		self._running = True
		self.pose_topic = 'pose'
		self.gesture_topic = 'gesture'
		self.gesture_topic_side = 'gesture_side'
		self.pub = rospy.Publisher(self.gesture_topic, String, queue_size=100)
		self.pub_side = rospy.Publisher(self.gesture_topic_side, String, queue_size=100)
		self.hz = 100

		self.w = width
		self.h = height

		self.fps = FPS
		self.needed_frames_for_gesture = 1 * self.fps
		self.buf = self.needed_frames_for_gesture * [None]
		self.side_buf = self.needed_frames_for_gesture * [None]

		self.last_gesture = None
		self.start_time = 0.0
		self.last_side_gesture = None
		self.start_side_time = 0.0

		self.frame_counter = 0.0
		self.up = 0.0

		# cheese gesture parameters
		self.cheese = False
		self.cheese_time = self.needed_frames_for_gesture
        

	def start_subscriber(self):
		rospy.init_node('gestures', anonymous=True)
		rate = rospy.Rate(self.hz)
		rgb_sub = rospy.Subscriber(self.pose_topic, numpy_msg(Floats), self.receive_pose, queue_size = 100)
		print("Pose subscriber successfully started!")
		rospy.spin()
		print("Subscriber is dead...")

	def receive_pose(self, data):	
		pose_data = np.array(data.data, dtype=np.float32)
		self.frame_counter += 1.0

		human = self.get_human_joints(pose_data)
		self.recognize_gesture(human, None)
		
	def get_human_joints(self, data):
		human_joints = {}
		i = 0
		while i < len(data):
			joint_id = data[i]
			x = data[i + 1]
			y = data[i + 2]
			score = data[i + 3]
			human_joints[joint_id] = [x,y,score]
			i += 4
		return human_joints

	def check_in(self, i, human):
		return i in human

	def horizontal_position(self, i, j, human):
		part_i = human[i]
		part_j = human[j]
		
		if part_i[1]  < part_j[1]:
			return 1
		if part_i[1]  > part_j[1]:
			return 0
    
	""" Distance between two points """
	def distance(self, i, j, human):
		part_i = human[i] 
		part_j = human[j]
           
		return np.linalg.norm((np.array([part_i[0], part_i[1]]) - np.array([part_j[0], part_j[1]])))

	""" Check if all points are detected in the skeleton pose """
	def validations(self, points, human):
		for i in points:
			if not self.check_in(i, human):
				return False
		return True 
	
	""" Calculate angle between two vectors described by the skeleton poses """
	def angle_between_two_points(self, a, b, c, human, d=-1):
		part_a = human[a]
		part_b = human[b]
		part_c = human[c]
		
		if d == -1:
			part_d = human[a]
		else:
			part_d = human[d]
		
		x = self.get_vector(part_a, part_b)
		y = self.get_vector(part_d, part_c)
		val = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) 
		if val > 1.0:
			val = 1.0
		if val < -1.0:
			val = -1.0
		return np.arccos(val) 

	def get_vector(self, origin, direction):
		return np.array([direction[0] - origin[0], direction[1] - origin[1]])
		

	def get_orientation(self, a, b, c, human, d=-1):
		angle = math.degrees(self.angle_between_two_points(a, b, c, human, d))

		#perpendicular
		if angle > 60.0 and angle < 120.0:
			return 1
		#straight
		elif angle > 140.0:
			return -1
		#nothing is recognized
		else:
			return None

	def get_status(self, a, b, c, human, d=-1):
		#angle = math.degrees(self.angle_between_two_points(a, b, c, human))
		# up and down
		hemisphere = self.horizontal_position(b, a, human)
		# orientation according to angle
		orientation = self.get_orientation(a, b, c, human, d)

		if orientation == 1:
			return hemisphere
		elif orientation == -1:
			return orientation
		else:
			# no gesture 
			return None

	def check_forearm(self, a, b, c, human, d=-1):
		if self.validations(np.array([a,b,c]), human):
			return self.get_status(a, b, c, human, d)

		#forearm not visible
		return None

	def check_complex_arm_gesture(self, upper_arm, forearm):
		if forearm is None or upper_arm is None:
			return None
		if upper_arm == -1:
			if forearm == -1:
				return -1
			else:
				return forearm - 2 * upper_arm		
		else:
			return upper_arm
			

	def temporal_stability(self, gestures):
		max_same_gesture = Counter(gestures.copy()).most_common(1)

		if max_same_gesture[0][1] > len(gestures) / 2 :
			return max_same_gesture
		else:
			return [(None, len(gestures))]

	def combined_hand_gesture(self, right_position, left_position, side=0):
		if not side:
			return self.pose(right_position, left_position)
		else:
			return self.pose_side(right_position, left_position)

	def pose(self, i,j):
		switcher={
			(3,3):'Up',
			(2,2):'Down',
			(1,1):'Cheese',
			(-1,3):'Right',
			(3,-1):'Left',
			(3,1):'Cw',
			(1,3):'Ccw',
			(-1,2):'Forward',
			(3,2):'Back'
		}
		return switcher.get((i,j), None)

	def pose_side(self, i,j):
		if i == 1 or i == 0: 
			return 'Side right'
		elif j == 1 or j == 0:
			return 'Side left'
		else:
			return None

	# frames with the same gesture on it are filtered every needed_frames_for_gesture seconds
	def filter_gestures(self, stable_gesture, side=-1):
		gesture = stable_gesture[0][0]

		if side == - 1:
			if gesture is not None:
				if self.last_gesture != gesture:
					self.last_gesture = gesture
					self.start_time = time.time()
					print(gesture)
					self.pub.publish(gesture)
					return gesture
				elif time.time() - self.start_time > self.needed_frames_for_gesture :
					self.start_time = time.time()
					self.pub.publish(gesture)
					return gesture
			else:
				self.last_gesture = None
				return None
		#side		
		else:
			if gesture is not None:
				if self.last_side_gesture != gesture:
					self.last_side_gesture = gesture
					self.start_side_time = time.time()
					self.pub_side.publish(gesture)
					return gesture
				elif time.time() - self.start_side_time > self.needed_frames_for_gesture :
					self.start_side_time = time.time()
					self.pub_side.publish(gesture)
					return gesture
			else:
				self.last_side_gesture = None
				return None


	def handle_cheese_gesture(self, rgb): 
		if self.cheese == True:
			self.cheese_time -= 1
			if self.cheese_time < 0 and self.cheese == True:
				self.cheese = False
				self.cheese_time = self.needed_frames_for_gesture / 2.0
				#cv2.imwrite('cheese/' + str(time.time()) + '.png', rgb)

	""" Recognize up/down position of left and right arms """
	def recognize_gesture(self, human, rgb):
		
		if self.validations([1, 5, 6], human) and self.validations(np.array([2,3,1]), human):
			
			# Right hand status
			right_position = self.get_status(2, 3, 1 , human)
			right_forearm = self.check_forearm(3, 4, 2, human)
			right_final_gesture = self.check_complex_arm_gesture(right_position, right_forearm)

			#Left hand status
			left_position = self.get_status(5, 6, 1, human)
			left_forearm = self.check_forearm(6, 7, 5, human)
			left_final_gesture = self.check_complex_arm_gesture(left_position, left_forearm)
			
			# combine gestures from left and right hand
			gestures = self.combined_hand_gesture(right_final_gesture, left_final_gesture)		
			self.buf.append(gestures) 
				
			if len(self.buf) > self.needed_frames_for_gesture:
				self.buf.pop(0)

		stable_gesture = self.temporal_stability(self.buf)
		gesture = self.filter_gestures(stable_gesture)
		
		if stable_gesture == 'Cheese':
			self.cheese = True
		#self.handle_cheese_gesture(rgb)


		''' side gestures '''
		if self.validations([11,1,5,6], human) or self.validations([8,1,2,3,4], human):

			if self.validations([8,1,2,3,4], human):
				# Right hand status
				right_position = self.get_status(2, 3, 8 , human, 1)
				right_forearm = self.check_forearm(3, 4, 2, human)
				right_final_gesture = self.check_complex_arm_gesture(right_position, right_forearm)
			else:
				right_final_gesture = None

			if self.validations([11,1,5,6], human):
				#Left hand status
				left_position = self.get_status(5, 6, 11, human,1)
				left_forearm = self.check_forearm(6, 7, 5, human)
				left_final_gesture = self.check_complex_arm_gesture(left_position, left_forearm)
			else:
				left_final_gesture = None
			
			# combine gestures from left and right hand
			gestures = self.combined_hand_gesture(right_final_gesture, left_final_gesture, 1)		
			self.side_buf.append(gestures) 

			if len(self.side_buf) > self.needed_frames_for_gesture:
				self.side_buf.pop(0)

		stable_gesture_side = self.temporal_stability(self.side_buf)
		gesture_side = self.filter_gestures(stable_gesture_side, 1)
		

		

""" Command line arguments """
if len(sys.argv) != 3 and len(sys.argv) != 4:
   print("python gestures_node.py [width] [height] ([FPS])")
   exit()

width = int(sys.argv[1])
height = int(sys.argv[2])
if len(sys.argv) !=3:
	FPS = int(sys.argv[3])
else:
	FPS = 5
g = Gestures(width, height, FPS)
g.start_subscriber()
