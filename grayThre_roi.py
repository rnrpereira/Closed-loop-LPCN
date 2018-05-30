# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:21:33 2018

@author: administrador
"""

def roi_thre(videoWidth=640,videoHeight=360):
    import cv2
    from roi_define import define_roi
    
    # ************** GRAY LEVEL SETUP ********************** # 
    def nothing(x):
        pass
    
    #VIDEO RESOLUTION
#    videoWidth = 640
#    videoHeight = 360
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
    ret0, frame0 = cap.read()
    img = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    img_thresh = img
    cv2.namedWindow('image')
    cv2.moveWindow("image", 800,20);
    
    # create trackbars for color change
    cv2.createTrackbar('LOW','image',0,255,nothing)
    cv2.createTrackbar('HIGH','image',0,255,nothing)
    #cv2.createTrackbar('B','image',0,255,nothing)
    
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,nothing)
    
    while(1):
        cv2.imshow('image',img_thresh)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
        # get current positions of four trackbars
        low_thre = cv2.getTrackbarPos('LOW','image')
        high_thre = cv2.getTrackbarPos('HIGH','image')
    #    b = cv2.getTrackbarPos('B','image')
        s = cv2.getTrackbarPos(switch,'image')
    
        if s == 0:
            img_thresh = img
        else:
            img_thresh = cv2.threshold(img,low_thre,high_thre,cv2.THRESH_BINARY)[1]
            #img[:] = [b,g,r]
    
    
    # ************** ROI SETUP ********************** # 
    pts = define_roi(frame0)
    
    # ************** END OF SETUP ********************** # 
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
#    cv2.waitKey(1)
    print("Hello You")
    return low_thre, high_thre, pts 
    