import os 
import numpy as np
import face_recognition
import cv2
import time

''' Face recognition '''
def recognize_face(frame, known_face_encodings, known_face_names):

	#frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	face_locations = face_recognition.face_locations(frame, model='cnn')
	face_encodings = face_recognition.face_encodings(frame, face_locations, model='large')

	face_names = []

	for face_encoding in face_encodings:
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
		name = "Unknown"

		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]
		face_names.append(name)
	#print(face_names)

	# get coordinates of the bounding box if any
	print(get_BB_location_by_name("Stanka", face_names, face_locations))

	# just for debugging - remove for better performance
	get_BB(frame, face_locations, face_names)


def get_name_from_frame(frame, known_face_encodings, known_face_names):
	face_locations = face_recognition.face_locations(frame)
	face_encodings = face_recognition.face_encodings(frame, face_locations)

	name = "Unknown"

	for face_encoding in face_encodings:
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]

	return name

def get_known_face_encodings_and_names():
	encodings = []
	names = []
	folder_name = "./template_images/"
	for image in os.listdir(folder_name):
		print(image)
		recgn_image = face_recognition.load_image_file(folder_name + image)
		recgn_face_encoding = face_recognition.face_encodings(recgn_image)[0]
		encodings.append(recgn_face_encoding)
		# name convention to get the accurate names of the people in the template database
		names.append(image.split('@')[0])
	return (encodings, names)

def get_BB(frame, locations, names):
	if names != []:
		for (top, right, bottom, left), name in zip(locations, names):
			# Draw a box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

			 # Draw a label with a name below the face
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
	cv2.imwrite('demo/' + str(time.time()) + '.png', frame)

def get_BB_location_by_name(name, face_names, face_locations):
	if name in face_names and name != "Unknown":
		index = face_names.index(name)
		return face_locations[index]
	else:
		return -1

