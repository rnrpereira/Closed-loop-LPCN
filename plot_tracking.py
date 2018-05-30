#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 22:57:54 2018

@author: rodrigo
"""

def plot_tracking(frame, res, mask, c):
    import cv2
    from colorama import Fore, Back
    
    global extLeft, extRight, extTop, extBot, xy_positions, contour_dist
    global cX, cY, x_tail, y_tail, x_nose, y_nose
    
    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(frame,[c], -2, (0, 255, 255), 1)
    cv2.circle(frame, extLeft, 1, (0, 0, 255), -1)
    cv2.circle(frame, extRight, 1, (0, 255, 0), -1)
    cv2.circle(frame, extTop, 1, (255, 0, 0), -1)
    cv2.circle(frame, extBot, 1, (255, 255, 0), -1)
    
    cv2.drawContours(res, [c], -1, (255, 255, 255), 1)
    cv2.circle(res, extLeft, 1, (255, 255, 255), -1)
    cv2.circle(res, extRight, 1, (255, 255, 255), -1)
    cv2.circle(res, extTop, 1, (255, 255, 255), -1)
    cv2.circle(res, extBot, 1, (255, 255, 255), -1)
    
    cv2.drawContours(mask,c, -1, (255, 255, 255), 1)
#    cv2.circle(mask, extLeft, 1, (255, 255, 255), -1)
#    cv2.circle(mask, extRight, 1, (255, 255, 255), -1)
#    cv2.circle(mask, extTop, 1, (255, 255, 255), -1)
#    cv2.circle(mask, extBot, 1, (255, 255, 255), -1)
    
    cv2.circle(res,(x_tail,y_tail), 3, (255, 255, 255), -1)
    cv2.line(res,(cX,cY),(x_tail,y_tail),(255,255,255),1)
    #
    cv2.circle(frame,(cX,cY), 2, (255,255, 255), -1)
    cv2.circle(frame,(x_nose,y_nose), 3, (0, 0, 255), -1)
    cv2.line(frame,(cX,cY),(x_nose,y_nose),(0,255,0),1)
    cv2.circle(frame,(x_tail,y_tail), 3, (0, 255, 0), -1)
    cv2.line(frame,(cX,cY),(x_tail,y_tail),(0,255,0),1)

    # COMMENT UNTIL FIXED    
#    cv2.circle(frame,(round(x_neck),round(y_neck)), 2, (0, 255, 255), -1)
    
    # show the output image
    cv2.imshow("Frame", frame)
    #cv2.namedWindow("win1");
    cv2.moveWindow("Frame", 1200,20);
    
    cv2.imshow('Gray',res)
    #cv2.namedWindow("win2");
    cv2.moveWindow("Gray", 1200,460);
    
    cv2.imshow('Mask',mask)
    #cv2.namedWindow("win3");
    cv2.moveWindow("Mask", 1200,615);