#!/usr/bin/env python

import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from sudoku import Sudoku
from keras.models import load_model

import os
import time
timestr = time.strftime("%Y%m%d%H%M%S")

cap = cv2.VideoCapture(0)
corners = []
captured = False

while(cap.isOpened()):
	ret,frame = cap.read()

	## PREPROCESSING

	img = imutils.resize(frame,height=400)
	# cv2.imshow("Original",img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray,(5,5),0)
	# blurred = cv2.bilateralFilter(gray, 11, 17, 17)
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
				cap.release()
				cv2.imshow("Contours",imgCp1)
				break

		if captured:
			break

#apply perspective transform
imgSudoku = four_point_transform(img,cntSudoku.reshape(-1,2))
cv2.imshow("Sudoku Warped",imgSudoku)
key = cv2.waitKey(0)

# Proceed only if the image is a Sudoku
if key&0xFF == ord("q"):
	exit()

## IMAGE RECOGNITION (Using PreTrained Model)

imrecModel = load_model("../models/MNIST_keras_CNN.h5")

imgSudoku = cv2.cvtColor(imgSudoku,cv2.COLOR_BGR2GRAY)
# imgSudoku = cv2.GaussianBlur(imgSudoku,(3,3),0)
# imgSudoku = cv2.adaptiveThreshold(imgSudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,4)
slideX = int(imgSudoku.shape[1]/9.0)
slideY = int(imgSudoku.shape[0]/9.0)
print imgSudoku.shape, slideX, slideY
cv2.imshow("Sudoku",imgSudoku)
key = cv2.waitKey(0)
cv2.imwrite("../ims/"+timestr+"_Unsolved.jpg",imgSudoku)


grid = []
mask = []

for i in range(0,imgSudoku.shape[0],slideY):
	row = []
	maskrow = []
	for j in range(0,imgSudoku.shape[1],slideX):
		S_cell = imgSudoku[i:i+slideY,j:j+slideX]
		if S_cell.shape[0] != slideY or S_cell.shape[1] != slideX:
			continue

		#Resize the cell to MNIST dimensions
		digit = cv2.resize(S_cell,(28,28))
		# Apply Adaptive Thresholding on inv image
		digit = cv2.adaptiveThreshold(digit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,3)
		digit = clear_border(digit)
		
		# cv2.imshow("digit",digit)

		n_zero = cv2.countNonZero(digit)

		if n_zero<50:
			maskrow.append(0)
			pred_class = 0
		else:
			maskrow.append(1)
			_,cnts,_ = cv2.findContours(digit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
			if len(cnts)>0:
				cnt = cnts[0]
				x,y,w,h = cv2.boundingRect(cnt)
				bound = digit[y:y+h,x:x+w]

				bound = cv2.erode(bound,np.ones((2,2),np.uint8),iterations = 1)

				bound = imutils.resize(bound,height=24)
				w = bound.shape[1]
				st = (28-w)/2
				digit = np.zeros((28,28))
				digit[2:26,st:st+w] = bound
				# cv2.imshow("digit",digit)

			pred = imrecModel.predict([digit.reshape(1,28,28,1)/255.])
			pred_class = pred.argmax(axis=-1)[0]
			if (pred_class == 0 or pred_class == 8 or pred_class == 5):
				digit = cv2.erode(digit,np.ones((2,2),np.uint8),iterations = 1)
				# cv2.imshow("erodeddigit",digit)
				pred = imrecModel.predict([digit.reshape(1,28,28,1)/255.])
				pred_class = pred.argmax(axis=-1)[0]

		# print pred_class,n_zero
		# key = cv2.waitKey(0)
		# if key&0xFF == ord("q"):
		# 	exit()

		row.append(pred_class)
	grid.append(row)
	mask.append(maskrow)
 
S = Sudoku(grid)

S.show()
key = cv2.waitKey(0)
if key&0xFF == ord("q"):
	os.remove("../ims/"+timestr+"_Unsolved.jpg")
	exit()

if(S.solve()):
	S.show()

	for i in range(0,9):
		for j in range(0,9):
			if not mask[i][j]:
				num = grid[i][j]
				digim = cv2.imread("../digits/"+str(num)+".jpg")
				digim = cv2.cvtColor(digim,cv2.COLOR_BGR2GRAY)
				digim = cv2.resize(digim,(slideX,slideY))
				imgSudoku[i*slideY:(i+1)*slideY,j*slideX:(j+1)*slideX] = digim
	cv2.imshow("Solved",imgSudoku)
	cv2.imwrite("../ims/"+timestr+"_Solved.jpg",imgSudoku)
	key = cv2.waitKey(0)
else:
	os.remove("../ims/"+timestr+"_Unsolved.jpg")
	print "No Solution Exists"

cv2.destroyAllWindows()
exit(0)
