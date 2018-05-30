# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:46:49 2018

@author: administrador
"""

def pose_calculations(frame_count, cnts, c):
    # GET POSE PARAMETERS FROM STATIC IMAGE CONTOUR
    # Based on Wang Z., Mirbozorgi A.A. & Ghonvanloo M. (2017)
    # An autmoated behavior analysis system for freely moving rodents using depth image
    
    import cv2
    import numpy as np
    import math
    
    global extLeft, extRight, extTop, extBot, xy_positions, contour_dist, roi
    global cX, cY, cX_frame,cY_frame, x_tail, y_tail, x_nose, y_nose, xc, yc
    global xy_nose_v, xy_tail_v,Sv, Rv, Ev, rhov
        
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])      
    #### CALCULATIONS ######
    cnts_pts,_,_ = cnts[0].shape
    contour_dist = np.zeros((cnts_pts,1),dtype=int)
    tail_nose_dist = np.zeros((cnts_pts,1),dtype=int)
    M = cv2.moments(c)
    if M["m00"] != 0:   
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))

        cX_frame = cX + roi[0]
        cY_frame = cY + roi[2]
         
        xy_positions[frame_count-1][0] = cX
        xy_positions[frame_count-1][1] = cY
    #    print('Centroid X = ',cX)
    #    print('Centroid Y = ',cY)
        # BODY AREA (S)
        S = M["m00"]
        # BODY RADIUS (R)
        for ii in range(cnts_pts):
            contour_dist[ii-1]  = math.hypot(cnts[0][ii-1,0][0] - cX,cnts[0][ii,0][1] - cY)            
        max_dist = contour_dist.max()
        imax_nose = np.argmax(contour_dist) 
        R = contour_dist[imax_nose]
        # NOSE POINT (x_nose, y_nose)
        x_nose = cnts[0][imax_nose,0][0]
        y_nose = cnts[0][imax_nose,0][1]
        # CIRCULARITY (E)
        E = S/(R**2)     
        # ELLIPTICITY (rho)
        ellipse = cv2.fitEllipse(cnts[0])
        xc = ellipse[0][0]
        yc = ellipse[0][1] 
        a =  ellipse[1][0]
        b =  ellipse[1][1]
        rho = b/a
        # TAIL BASE
        for ii in range(cnts_pts):
            tail_nose_dist[ii-1]  = math.hypot(cnts[0][ii-1,0][0] - x_nose,cnts[0][ii,0][1] - y_nose)
        max_dist_tail_nose = tail_nose_dist.max()
        imax_tail_nose = np.argmax(tail_nose_dist)
        x_tail = cnts[0][imax_tail_nose,0][0]
        y_tail = cnts[0][imax_tail_nose,0][1]
        xy_nose_v[frame_count-1][0] = x_nose
        xy_nose_v[frame_count-1][1] = y_nose
        xy_tail_v[frame_count-1][0] = x_tail
        xy_tail_v[frame_count-1][1] = y_tail
        Sv[frame_count-1][0] = S
        Rv[frame_count-1][0] = R
        Ev[frame_count-1][0] = E
        rhov[frame_count-1][0] = rho

        print('Ellipse Center = ', ellipse[0])
        print('MajAx,MinAx length = ', ellipse[1])    
        print('Rot_angle = ', ellipse[2])
    #    print('CenterBox = ', rect[0])    
    #    print('W,H = ', rect[1])    
    #    print('Box_angle = ', rect[2])
        print('Zone Box Vx1 = ', box_temp[0][0])
        print('Zone Box Vx2 = ', box_temp[0][3])
        print('Body_Center in-zone: ', Fore.RED + zoneB)
        print(Fore.RESET+ Back.RESET,'########################')

        
    else:
         pass