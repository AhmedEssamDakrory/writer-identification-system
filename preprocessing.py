import numpy as np
import cv2
import math

def paragraphExtraction(gray):
	# Get edges
	edges = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=5)

	# Get horizontal lines
	lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 100, None, 600, 1);

	# Remove noise with opening operator
	kernel = np.ones((5,5),np.uint8)
	kernel[0,:] = 0
	kernel[4,:] = 0
	opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
	
	# Get horizontal lines rows
	rows = []
	for line in lines:
		rows.append(line[0][1])
	rows.sort()

	# Get upper, lower bounds
	lower, upper = 2850, 0
	for row in rows:
		if row > 400 and row < 1100:
			upper = max(upper, row)
		elif row >2500 and row < 2850:
			lower = min(lower, row)
	return gray[upper: lower - 20, 200:] # Margin to remove edge pixels
