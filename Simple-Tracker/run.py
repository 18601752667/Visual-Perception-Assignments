import numpy as np 
import cv2
import sys
from time import time

import basetracker


initTracking = True
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
inteval = 1
duration = 0.01

if __name__ == '__main__':
	
	if(len(sys.argv)==1):
		cap = cv2.VideoCapture(0)
	elif(len(sys.argv)==2):
		if(sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
			cap = cv2.VideoCapture(int(sys.argv[1]))
		else:
			cap = cv2.VideoCapture(sys.argv[1])
			inteval = 30
	else:  assert(0), "too many arguments"

	# tracker = kcftracker.KCFTracker(False, True, False)  # hog, fixed_window, multiscale
	tracker = basetracker.BaseTracker(False, True, False)  # hog, fixed_window, multiscale
	#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

	cv2.namedWindow('tracking')

	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret:
			break

		if(initTracking):
			print('start roi')
			r = cv2.selectROI('tracking',frame)
			ix, iy, w, h = r
			# print(ix, iy, w, h)
			# print(len(frame))
			# print("#########")
			tracker.init([ix,iy,w,h], frame)
			initTracking = False


		else:
			t0 = time()
			boundingbox = tracker.update(frame)
			t1 = time()

			boundingbox = list(map(int, boundingbox))
			# print(boundingbox)
			cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,255), 1)
			
			duration = 0.8*duration + 0.2*(t1-t0)
			#duration = t1-t0
			cv2.putText(frame, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

		cv2.imshow('tracking', frame)
		c = cv2.waitKey(inteval) & 0xFF

	cap.release()
	cv2.destroyAllWindows()
