# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:21:33 2018

@author: administrador
"""
import cv2
import imutils

def frame_process(frame, roi, low_thre, high_thre):
    if (len(frame.shape) == 3):
       # BGR_TO_GRAY COLOR CONVERSION
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
       gray = frame

    # CROP frame to bahavior apparatus    
    gray_frame = gray[roi[2]:roi[3],roi[0]:roi[1]] 
    
    # SMOOTH GRAY frame
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)         

    # THRESHOLD the image        
    mask = cv2.threshold(gray_frame,low_thre,high_thre,cv2.THRESH_BINARY)[1]
    # mask = cv2.inRange(gray_frame,low_thre,high_thre)
    
    # Perform a series of EROSIONS + DILATIONS to remove any small regions of noise        
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)  
       
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(gray_frame,gray_frame, mask= mask)  
                
    # FIND CONTOURS in thresholded image, then grab the largest one  
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:2]
    c = cnts[0]  #max(cnts, key=cv2.contourArea)
#    contours.append(np.squeeze(c, axis=1))     
    print("Hello You")
    return res, mask, cnts, c
    
    
