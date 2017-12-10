#!/usr/bin/env python

import cv2
import numpy as numpy
from matplotlib import pyplot as plt
import imutils
from imutils.perspective import four_point_transform
from sudoku import Sudoku


cap = cv2.VideoCapture(0)
corners = []
captured = False

while(cap.isOpened()):
	ret,frame = cap.read()

	## PREPROCESSING

	img = imutils.resize(frame,height=400)
	# cv2.imshow("Original",img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray,(5,5),0)
	blurred = cv2.bilateralFilter(gray, 11, 17, 17)
	#apply adaptive thresholding
	thres = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,4)
	cv2.imshow("Thresholded",thres)

	## EXTRACTION

	key = cv2.waitKey(1) & 0xFF 
	if key == ord('q'):
		#quit
		cap.release()
		cv2.destroyAllWindows()
		exit(0)

	elif key == ord('c'):
		#capture
		# find contours in the thresholded image
		_,cnts,_ = cv2.findContours(thres.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# store largest 4 contours since sudoku contour must be large
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4]
		cntSudoku = None
		
		#since cv2.findContours is a destructive method
		imgCp1 = img.copy() 

		for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)

			# if our approximated contour has four points, then
			# we can assume that we have found our screen
			if len(approx) == 4:
				cntSudoku = approx     

				cv2.drawContours(imgCp1,[cntSudoku],-1,(0,0,255),3)
				captured = True
				cv2.imshow("Contours",imgCp1)
				break

		if captured:
			break

#apply perspective transform
imgSudoku = four_point_transform(img,cntSudoku.reshape(-1,2))
cv2.imshow("Sudoku Warped",imgSudoku)


cv2.waitKey(0)


cap.release()
cv2.destroyAllWindows()
exit(0)