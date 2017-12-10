# Sudoku Solver

## Description

The completion of the project requires the following 5 steps:

---Preprocess Image

---Extract Sudoku

---Digit Recognition and Sudoku Matrix formation

---Solve the sudoku

### Preprocess Image
The image is first blurred slightly using a Bilateral or Gaussian filter to remove noise. Bilateral filter was seen to produce better results.

In case an image has different lighting conditions in different areas, adaptive thresholding is preferred. The algorithm calculates the threshold for a small region of the image. So we get different thresholds for different regions of the same image and it gives us better results for images with varying illumination. Here, we use adaptive gaussian thresholding where the region threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.

### Extract Sudoku
To extract just the Sudoku from the image, we need to find contours/outline of the puzzle. cv2.findContours is used here to find the largest rectangular contour (use a copy of the image).

We approximate the contour using cv2.arcLength and cv2.approxPolyDP, which give polygonal curves of a contour. 2% of the perimeter of the contour is used as the precision. The ROI is valid if no of points in the contour is 4 (rectangle), and a bounding box (contour) is drawn around the Sudoku using drawContour.

Finally a perspective transform is applied on the marked region in the original image using the four_point_transform function of the imutils.perspective module to extract the Sudoku in a top-down view.

### Digit Recognition and Sudoku Matrix formation

### Solve the sudoku
For solving the sudoku, a backtracking algorithm is used, which is a special case type of Brute Force search. 

A cell is tried with digits from 1-9 and checked for validity. If not a valid move, the next digit is placed and checked for again. If valid, then the next empty cell is searched for and filled the same way. In this way, it is a depth first search with depth N (no. of unfilled cells) and a maximum branching factor 9. Worst case performance can be O(9^N).

## Libraries used
OpenCV-3.2.0
numpy
matplotlib.pyplot
imutils
imutils.perspective

## Authors
Kunal Jain
