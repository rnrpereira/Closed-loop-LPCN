# -*- coding: utf-8 -*-
#"""
#Created on Mon Apr 30 15:20:49 2018
#
#@author: Rodrig Neves Romcy Pereira
#"""
## ######################################################
 #DEFINE A ROI IN AN IMAGE - FUNCIONA (# CELL 1)
import cv2
import numpy as np

def define_roi3(image,nzones):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (400,15)
    fontScale              = 0.5
    fontColor              = [(0,0,255),(0,255,0),(0,255,255),(255,0,0),(0,0,255),(0,255,0),(0,255,255),(255,0,0)]
    lineType               = 1
    
    cv2.putText(image,'DEFINE TARGET ZONES', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor[0],
        lineType,
        cv2.LINE_AA)

    clone = image.copy()
    rect_pts = [] # Starting and ending points
    zones_xy = np.zeros((nzones,1,4),dtype=int)
    win_name = "frame0" # Window name
    zone_count = 0
    
    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]
            
        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))            
            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], fontColor[zone_count], 2)
            cv2.imshow(win_name, clone)
            cv2.moveWindow(win_name, 1200,360);
            #plt.imshow(clone)
            #plt.show()

    cv2.namedWindow(win_name)
    if nzones > 0:
        for k in range(nzones):
            cv2.setMouseCallback(win_name, select_points)
    
            while True:
                # display the image and wait for a keypress
                cv2.imshow(win_name, clone)
                cv2.moveWindow(win_name, 1200,360);
                #plt.imshow(clone)
                #plt.show()
    
                key = cv2.waitKey(0) & 0xFF            
                if key == ord("r"): # Hit 'r' to replot the image
                    clone = image.copy()
                elif key == ord("c") and  zone_count < nzones: # Hit 'c' to confirm the selection
                    zone_count = zone_count + 1
                    zones_xy[zone_count-1] = np.append(np.asarray(rect_pts[0]),np.asarray(rect_pts[1]))                    
                    break
                elif key == ord("c") and  zone_count == nzones:
                    zones_xy[zone_count-1] = np.append(np.asarray(rect_pts[0]),np.asarray(rect_pts[1]))                    
                    break
    else:    
        
        # close the open windows
        cv2.destroyWindow(win_name)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)         
        
    zones_xy = np.squeeze(zones_xy)
    return rect_pts, zones_xy

# Prepare an image for testing
# A image array with RGB color channels
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert RGB to BGR

# Points of the target window
#rect_pts = define_roi(frame0)

#print("--- target window ---")
#print("Starting point is ", rect_pts[0])
#print("Ending   point is ", rect_pts[1])

