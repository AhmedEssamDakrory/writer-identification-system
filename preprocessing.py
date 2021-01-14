import numpy as np
import cv2
import math
import time
import os

def paragraphExtraction(gray):
	# Get edges
	edges = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=5)

	# Get horizontal lines
	lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 100, None, 600, 1);

	# Get horizontal lines rows
	rows = []
	for line in lines:
		rows.append(line[0][1])
	rows.sort()

	# Get upper, lower bounds
	lower, upper = 2800, 2800
	for row in rows:
		if row > 500:
			upper = min(upper, row)
	return gray[upper:lower, :]
