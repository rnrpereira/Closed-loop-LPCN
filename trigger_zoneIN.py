#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:33:01 2018

@author: administrador
"""
import numpy as np
import cv2
global frame_count
 
def trigger_zoneIN(frame, cX_frame,cY_frame,tg_zone_indx,box_temp,zone_IN,frame_count,videoWidth, videoHeight):
    
    contour_in = cv2.pointPolygonTest(box_temp[tg_zone_indx],(cX_frame,cY_frame),False)
    
    # ************** VISUAL OUTPUT WHEN ANIMAL GO INSIDE TARGET ZONE **************
    zone_IN[frame_count-1] = np.int(contour_in)
    if (contour_in == 1) or (contour_in == 0):
        zoneB = 'IN'
        cv2.circle(frame,(int(0.1*videoWidth),int(0.1*videoHeight)), 2, (0, 255, 0), 16)
        
        # ADD HERE OUTPUT TRIGGER TO CONTROL STIMULATION, LED etc...
        # ...
        
    else:
        zoneB = 'OUT'
        cv2.circle(frame,(int(0.1*videoWidth),int(0.1*videoHeight)), 2, (0, 0, 0), 16)
        
    # ************** VISUAL OUTPUT WHEN ANIMAL GO INSIDE TARGET ZONE **************
    
    return contour_in, zoneB