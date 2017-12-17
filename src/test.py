#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from sudoku import Sudoku
from keras.models import load_model




## PREPROCESSING
frame = cv2.imread("../ims/sudoku7.jpg")
img = imutils.resize(frame,height=400)
# cv2.imshow("Original",img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
# blurred = cv2.bilateralFilter(gray, 11, 17, 17)
#apply adaptive thresholding
thres = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,4)
# cv2.imshow("Thresholded",thres)

## EXTRACTION

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
		# cv2.imshow("Contours",imgCp1)
		break


#apply perspective transform
imgSudoku = four_point_transform(img,cntSudoku.reshape(-1,2))
# cv2.imshow("Sudoku Warped",imgSudoku)
# key = cv2.waitKey(0)

# # Proceed only if the image is a Sudoku
# if key&0xFF == ord("q"):
# 	exit()

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

grid = []

for i in range(0,imgSudoku.shape[0],slideY):
	row = []
	for j in range(0,imgSudoku.shape[1],slideX):
		S_cell = imgSudoku[i:i+slideY,j:j+slideX]
		if S_cell.shape[0] != slideY or S_cell.shape[1] != slideX:
			continue

		#Resize the cell to MNIST dimensions
		digit = cv2.resize(S_cell,(28,28))
		# Apply Adaptive Thresholding on inv image
		digit = cv2.adaptiveThreshold(digit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,4)
		digit = clear_border(digit)
		
		# # digit = cv2.morphologyEx(digit,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
		digit = cv2.erode(digit,np.ones((2,2),np.uint8),iterations = 1)
		
		cv2.imshow("digit",digit)

		n_zero = cv2.countNonZero(digit)

		if n_zero<30:
			pred_class = 0
		else:
			pred = imrecModel.predict([digit.reshape(1,28,28,1)/255.])
			pred_class = pred.argmax(axis=-1)[0]
		print pred_class,n_zero

		key = cv2.waitKey(0)
		if key&0xFF == ord("q"):
			exit()

	# 	row.append(pred_class)
	# grid.append(row)
 
S = Sudoku(grid)
S.show()

key = cv2.waitKey(0)
if key&0xFF == ord("q"):
	exit()

if(S.solve()):
	S.show()
else:
	print "No Solution Exists"

cv2.destroyAllWindows()
exit(0)
