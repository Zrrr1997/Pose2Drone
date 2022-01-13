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

import numpy as np

sys.path.insert(1, '../../tf-pose-estimation/') # Make sure to have tf-openpose cloned at this path

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from recognition import *

from multiprocessing import Manager, Pool, Process


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class OpenPose:
    def __init__(self, width=640, height=480, user_name="Zdravko"):
        self._running = True
        self.e = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368))
        self.w, self.h = model_wh('432x368')

        self.manager = Manager()
        self.received_frames = self.manager.list()

        self.rgb_topic = 'rgb'

        self.pose_topic = 'pose'
        self.pub_pose = rospy.Publisher(self.pose_topic, numpy_msg(Floats), queue_size=100)

        self.bbox_topic = 'bbox'
        self.pub_bbox = rospy.Publisher(self.bbox_topic, numpy_msg(Floats), queue_size=100)

        self.face_bbox_topic = 'face_bbox'
        self.pub_face_bbox = rospy.Publisher(self.face_bbox_topic, numpy_msg(Floats), queue_size=100)

        self.w_h_topic = 'w_h'
        self.pub_w_h = rospy.Publisher(self.w_h_topic, numpy_msg(Floats), queue_size=100)

        self.view_topic = 'view'
        self.pub_view = rospy.Publisher(self.view_topic, numpy_msg(Floats), queue_size=100)

        self.crop_topic = 'crop' # crop of whole body published
        self.pub_crop = rospy.Publisher(self.crop_topic, numpy_msg(Floats), queue_size=100)

        self.im_h = height 
        self.im_w = width 

        self.user_name = user_name

        self.hz = 100

        (self.known_face_encodings, self.known_face_names) = get_known_face_encodings_and_names()
        
        rospy.init_node('openpose', anonymous=True)

    ''' start rgb frame reader process '''
    def start_rgb_stream(self):
        rospy.init_node('openpose', anonymous=True)
        rate = rospy.Rate(self.hz)
        rgb_sub = rospy.Subscriber(self.rgb_topic, numpy_msg(Floats), self.receive_frame, queue_size = 100)
        print("RGB subscriber successfully started!")
        rospy.spin()
        print("Subscriber is dead...")

    def receive_frame(self, data):
         frame = np.array(data.data.reshape(self.im_h, self.im_w, 3), dtype=np.float32)
         """ Callback method for receiving RGB stream from Tello drone """

         # Resize frame if needed
         height, width, _ = frame.shape
         new_h = int(height / 2)
         new_w = int(width / 2)

         # Resize for improved performance
         new_frame = cv2.resize(frame, (new_w, new_h))
         if len(self.received_frames) > 30:
                self.received_frames[:] = [self.received_frames[-1]]
         self.received_frames.append(new_frame)

    def get_pose(self):
        """ Estimate the openpose of the last frame in shared buffer and visualize it"""
        loop = 0
        rec_frames = 0
        guessed_users = 0

        last_user_bbox = np.array([0,0,0,0], dtype = np.float32)
        while self._running:
            if len(self.received_frames) > 0:
                loop += 1
                new_frame = self.received_frames.pop() # last frame
                before_estimation = time.time()
                humans = self.e.inference(new_frame, resize_to_default=(self.w > 0 and self.h > 0), upsample_size = 4.0) 
                print("Estimation time FPS ", int(1.0 / (time.time() - before_estimation))) # for debugging and performance benchmarking

                means, stds, new_humans = self.filter_humans(humans) # filter non-human objects with confidence <30%

                if len(new_humans) == 0:
                    print("No humans")
                    continue # no humans left

                new_frame = TfPoseEstimator.draw_humans(new_frame, new_humans, imgcopy = True)

                new_frame = cv2.resize(new_frame, (self.im_w, self.im_h), interpolation = cv2.INTER_AREA) # resized frame to input resolution

                bboxs_arr = [] # only body
                bboxs_arr_face = []
                head_neck_waist_distances = []

                scores = [] # confidence scores
                xys = [] # xy positions
                idxs = [] # joint indices

                human_bboxs = {} # body and face bboxes
                human_proportions = {} # head-neck-waist distance and face bbox - 4 parameters used for distance estimation
                human_views = {} # viewpoint of the human (Side, Front, Back, None) - (1, 2, 3, 4)
                
                user_found = False
                possible_bboxs = []

                for j, human in enumerate(new_humans):
                    for i in range(26):
                       if i not in human.body_parts.keys():
                          continue
                       body_part = human.body_parts[i]
                       xy = [body_part.x * self.im_w + 0.5, body_part.y * self.im_h + 0.5]
                       new_frame = cv2.putText(new_frame, str(i), (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                    if self.get_bbox_face(human, self.im_w, self.im_h) is not None: # face, neck and waist are visible
                       bbox_face = self.get_bbox_face(human, self.im_w, self.im_h) 
                       face_patch = new_frame[int(bbox_face[1]):int(bbox_face[3]), int(bbox_face[0]):int(bbox_face[2])] 


                       if face_patch.shape[0] != 0 and face_patch.shape[1] != 0 and face_patch.shape[2] != 0:

                           if get_name_from_frame(self.resize_to_256(np.array(face_patch, dtype=np.uint8)), self.known_face_encodings, self.known_face_names) == self.user_name):
                               rec_frames += 1.0
                               guessed_users += 1.0
                               print("User found!")
                               last_user_bbox = np.array(bbox_face, dtype = np.float32) / 2
                               user_found = True

                               start_point = (int(last_user_bbox[0]) , int(last_user_bbox[1]))
                               end_point = (int(last_user_bbox[2]) , int(last_user_bbox[3]))
                               cv2.imwrite('./demo/' + str(time.time()) + '_found.png', new_frame)

                               self.pub_face_bbox.publish(np.array(self.get_bbox_face(human, self.im_w, self.im_h), dtype=np.float32) / 2) # publish face bbox 

                           else:
                               rec_frames += 1
                               possible_bboxs.append(np.array(bbox_face, dtype = np.float32) / 2)
                               #continue # not the user we are tracking
                           print('User ID accuracy: ', guessed_users / rec_frames)
                    else:
                       continue

                    human_views[j] = self.get_view(human, self.im_w, self.im_h)
                    

                    idxs = self.get_idxs(human)
                    xys += self.get_xys(human, new_frame.shape[0], new_frame.shape[1])
                    scores += self.get_scores(human)
                    
                    body_bbox = [int(el) for el in self.get_bbox(human, self.im_w, self.im_h)]

                    hnw_dist = self.get_distances(human, new_frame.shape[0], new_frame.shape[1])

                    if hnw_dist is not None: # Add only if head, neck and waist are visible
                       bboxs_arr += body_bbox
                       head_neck_waist_distances += hnw_dist
                    else:
                       print("Head, neck or waist are not visible")

                    if self.get_bbox_face(human, self.im_w, self.im_h) is not None and hnw_dist is not None: # face, neck and waist are visible

                       body_face_bbox = [int(el) for el in self.get_bbox_face(human, self.im_w, self.im_h)]
                       bboxs_arr_face += self.get_bbox_face(human, self.im_w, self.im_h) 

                       human_bboxs[j] = self.get_bbox(human, self.im_w, self.im_h) + self.get_bbox_face(human, self.im_w, self.im_h)
                       human_proportions[j] = hnw_dist + self.get_bbox_face(human, self.im_w, self.im_h)

                if not user_found:
                   min_dist = 1000000.0 # just a high distance
                   best_bbox = None
                   for i, el in enumerate(possible_bboxs):
                       if np.linalg.norm(el - last_user_bbox) < min_dist:
                          
                          min_dist = np.linalg.norm(el - last_user_bbox)
                          best_bbox = el
                   print(min_dist)
                   if best_bbox is not None:
                      print('TRACKING')

                      start_point = (int(best_bbox[0]) , int(best_bbox[1]))
                      end_point = (int(best_bbox[2]) , int(best_bbox[3]))
                      self.pub_face_bbox.publish(best_bbox) # publish face bbox
                      last_user_bbox = best_bbox

                # Test if bbox of face is okay
                ind = 0
                while ind < len(bboxs_arr_face):
                      start_point = (int(bboxs_arr_face[ind] / 2) , int(bboxs_arr_face[ind+1] / 2))
                      end_point = (int(bboxs_arr_face[ind+2] / 2) , int(bboxs_arr_face[ind+3] / 2 ))

                      ind += 4
                

		# publish width and height of face bounding boxes
                for el in human_proportions.values():#human_bboxs.values():
                    wh = [el[1] ,el[4] - el[2], el[5] - el[3], (el[4] - el[2]) * (el[5] - el[3])] + human_views[0] # nw dist and face-bbox dimensions
                    self.pub_w_h.publish(np.array(wh , dtype=np.float32)) # TODO: publishing the image leads to large overhead!

                
                self.pub_bbox.publish(np.array(bboxs_arr, dtype=np.float32)) # publish whole body bbox
  
                # Create pose array with joint ids and confidence scores
                pose_arr = []
                for i, el in enumerate(idxs):
                    pose_arr.append(el)
                    pose_arr.append(xys[2*i])
                    pose_arr.append(xys[2*i + 1])
                    pose_arr.append(scores[i])

                # Pose published in format: [index, x, y, score, index, x, y, score, ....] etc.
                self.pub_pose.publish(np.array(pose_arr, dtype=np.float32)) # TODO: publishing the image leads to large overhead



    """ Filter bad estimations - under 30% confidece in pose"""
    def filter_humans(self, humans):
        means, stds, new_humans = [], [], []
        for human in humans:
            scores = self.get_scores(human)
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            if np.mean(scores) > 0.3:
               new_humans.append(human)
        return means, stds, new_humans


    """ Get head-to-neck and neck-to-waist distances """
    def get_distances(self, human, w, h):
        for i in [0, 1, 8]:
            if i not in human.body_parts.keys():
               return None
        head = np.array([human.body_parts[0].x * w,human.body_parts[0].y * h])
        neck = np.array([human.body_parts[1].x * w,human.body_parts[1].y * h]) 
        waist = np.array([human.body_parts[8].x * w,human.body_parts[8].y * h]) 
        head_neck = np.linalg.norm(head - neck)
        neck_waist = np.linalg.norm(neck - waist)
        return [head_neck, neck_waist]

    """ Get confidence scores """
    def get_scores(self, human):
        return [x.score for _, x in human.body_parts.items()]

    """ Get (x,y) coordinates of joints """
    def get_xys(self, human, w, h):
        xys = []
        for i in range(26):
            if i not in human.body_parts.keys():
               continue
            body_part = human.body_parts[i]
            xy = [body_part.x * w + 0.5, body_part.y * h + 0.5]
            xys += xy
        return xys

    """ Get available joint indices """
    def get_idxs(self, human):
        return human.body_parts.keys()

    """ Get bbox of human """
    def get_bbox(self, human, w, h):
        centers = []
        for i in range(26):
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            center = [body_part.x * w, body_part.y * h] 
            centers.append(center)

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_xs = max(min(xs) - 0.1 * (max(xs) - min(xs)) ,0) # 10% extension
        min_ys = max(min(ys) - 0.1 * (max(ys) - min(ys)) ,0) # 10% extension

        max_xs = min(max(xs) + 0.1 * (max(xs) - min(xs)) ,self.im_w) # 10% extension
        max_ys = min(max(ys) + 0.1 * (max(ys) - min(ys)) ,self.im_h) # 10% extension

        top_left = [min_xs, min_ys] 
        bottom_right = [max(xs), max(ys)]
        return top_left + bottom_right

    """ Get bbox of face """
    def get_bbox_face(self, human, w, h):
        # Make sure we have at least some face features
        if 1 not in human.body_parts.keys() or (14 not in human.body_parts.keys() and 15 not in human.body_parts.keys() and 16 not in human.body_parts.keys() and 17 not in human.body_parts.keys()):
            return None

        centers = []
        for i in [1, 14, 15, 16, 17]:
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            center = [body_part.x * w, body_part.y * h] 
            centers.append(center)

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_xs = max(min(xs) - 0.2 * (max(xs) - min(xs)) ,0) # 20% extension
        min_ys = max(min(ys) - 0.6 * (max(ys) - min(ys)) ,0) # 60% extension

        max_xs = min(max(xs) + 0.2 * (max(xs) - min(xs)) ,self.im_w) # 20% extension
        max_ys = min(max(ys) - 0.2 * (max(ys) - min(ys)) ,self.im_h) # -20% extension


        top_left = [min_xs, min_ys] 
        bottom_right = [max_xs, max_ys]
        return top_left + bottom_right

    """ Get view - side, front, back or None if it is ambiguous """ 
    def get_view(self, human, w ,h):
        centers = {}
        front = True
        back = True
        side = True
        for i in [0, 1, 2, 5, 16, 17]:
            if i not in human.body_parts.keys():
               front = False
               if i != 0:
                  back = False
               if i != 16 and i != 17:
                  side = False
               continue
            body_part = human.body_parts[i]
            center = np.array([body_part.x * w, body_part.y * h])
            centers[i] = center
        if side:
            if np.linalg.norm(centers[2] - centers[5]) < 0.5 * np.linalg.norm(centers[0] - centers[1]): # shoulders are at a very high angle
                return [0,1,0] #"Side"

        if front:
            if centers[16][0] < centers[17][0] and centers[2][0] < centers[5][0] and front:
                return [1, 0, 0] #"Front"
        if back:
            if centers[16][0] > centers[17][0] and centers[2][0] > centers[5][0]:
                return [0,0,1] #2 #"Back"

        return [0,0,1] #3 #"None" # ambiguous view
   
    def resize_to_256(self, frame):
        if frame.shape[1] < 256:
           frame = cv2.resize(frame, (0, 0), fx = 2.0, fy = 2.0,  interpolation = cv2.INTER_AREA)
           return self.resize_to_256(frame)
        return frame

    ''' Kill OpenPose-ROS Node and related processes '''
    def terminate(self):
        self._running = False
        cv2.destroyAllWindows()


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


if len(sys.argv) != 4:
   print("python openpose_ros [width] [height] [user_name]")
   exit()

width = int(sys.argv[1])
height = int(sys.argv[2])
user_name = str(sys.argv[3])


""" Start new thread for receive Tello response message """
t = OpenPose(width, height, user_name)

read_buffer_p = threading.Thread(target=t.get_pose)
open_pose_p = threading.Thread(target=t.start_rgb_stream)

read_buffer_p.start()
open_pose_p.start()
read_buffer_p.join()
open_pose_p.join()


