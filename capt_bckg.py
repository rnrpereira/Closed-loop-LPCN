# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:21:33 2018

@author: administrador
"""

def capture_bckg(videoWidth,videoHeight):
    import cv2
#    videoWidth = 640
#    videoHeight = 360    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
    ret0, frame_bckg = cap.read()
    
    while ret0:    # frame captured without any errors
        cv2.imshow("Frame",frame_bckg)
        cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
        cv2.moveWindow("Frame", 800,20);
        frame_bckg = cv2.cvtColor(frame_bckg, cv2.COLOR_BGR2GRAY)    
        cv2.imshow("Frame Backg",frame_bckg)
        cv2.namedWindow("Frame Backg",cv2.WINDOW_NORMAL)
        cv2.moveWindow("Frame Backg", 800,445);
        
        # ******* SAVE BACKGROUD FRAMES *******#   
    #    cv2.imwrite('/home/administrador/Programs/DAQ/frame_bckg.jpg',frame_bckg) #save image
    #    cv2.imwrite('/home/administrador/Programs/DAQ/frame_bckg_backup.jpg',frame_bckg) #save image        
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
             break    
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
    cv2.waitKey(1)    
    print("Hello You")
    
    