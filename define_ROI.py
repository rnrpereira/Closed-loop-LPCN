#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:36:15 2018

@author: administrador
"""

import cv2
from skimage import data

def define_rect(image):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    rect_pts = [] # Starting and ending points
    win_name = "image" # Window name

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 800,20);
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    # close the open windows
    cv2.destroyWindow(win_name)

    return rect_pts


# Prepare an image for testing
#img = gray_img # A image array with RGB color channels
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert RGB to BGR

# Points of the target window
points = define_rect(bckg_frame)

print("--- target window ---")
print("Starting point is ", points[0])
print("Ending   point is ", points[1])