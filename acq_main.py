# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:21:33 2018

@author: administrador
"""

def acq_main(low_thre,high_thre,pts,params,ser):
    

    
    import cv2
    import time
    import numpy as np
    import scipy.io
    import frame_process
    import plot_tracking
    from pose_features import pose_calculations
    import trigger_zoneIN
    
    global frame_n, roi, xy_positions,xy_nose_v, xy_tail_v,Sv, Rv, Ev, rhov, cX_frame, cY_frame
        
    #### SET PARAMETERS ####
    # ********* BEHAVIOR + TARGET ZONE BOXES PARAMETERS *********************************************************
    # BEHAVIOR BOX # BOX ARENA
    box_arena = np.asarray([
            [[pts[0][0],pts[0][1]]],
            [[pts[0][0],pts[1][1]]],
            [[pts[1][0],pts[1][1]]],
            [[pts[1][0],pts[0][1]]]
            ])
    box_arena = np.int32(box_arena)

    # TARGET ZONES          
    tg_zone_indx = params['target_zone'] - 1
    zones_xy = params['zones_xy']
    zoneVertices = np.zeros((4,nzones,2),dtype=int)
    box_temp = []    
    if nzones == 1:
        box_temp = np.array([
                [[zones_xy[0,],zones_xy[1,]]],
                [[zones_xy[0,],zones_xy[3]]],
                [[zones_xy[2,],zones_xy[3,]]],
                [[zones_xy[2,],zones_xy[1,]]]])
    else:            
        for kk in range(nzones):
            box_tgzones = np.array([
                    [[zones_xy[kk][0],zones_xy[kk][1]]],
                    [[zones_xy[kk][0],zones_xy[kk][3]]],
                    [[zones_xy[kk][2],zones_xy[kk][3]]],
                    [[zones_xy[kk][2],zones_xy[kk][1]]]])                     
            box_temp.append(box_tgzones)

# ********* END BEHAVIOR + TARGET ZONE BOXES PARAMETERS *********************************************************

    # Write some Text
    font                    = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText     = (int(0.7 * videoWidth),int(0.05 * videoHeight))
    topRightCornerOfText    = (int(0.7 * videoWidth),int(0.1 * videoHeight))
    fontScale               = 0.5
    fontColor1              = (255,150,0)
    fontColor2              = (0,0,255)
    lineType                = 1
            
    # ********* VIDEO ACQ PARAMETERS *********************************************************
    videoWidth = params['videoWidth']
    videoHeight = params['videoHeight']
    # Time & FrameRate Settings
    frate_set = params['frate_set']     # FRAME RATE
    rec_dur = params['frate']           # Time of recording video (in seconds)
    nframes = int(rec_dur * frate_set)  # NUMBER OF FRAMES TO BE CAPTURED (N)
    # Define the codec and create VideoWriter object
    # works with MP4V (.MP4) & DIV4,3IVD,FMP4,DIVX,DX50,MP4V,XVID (ORDEM CRESENTE DE TAMANHO DO ARQUIVO DE VIDEO)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # Video output
    output_dir = params['output_dir']
    video_name = params['video_file']
    out = cv2.VideoWriter((output_dir + video_name),fourcc,frate_set,(videoWidth,videoHeight))
     
    frame_count = 0
    frame_n = 0
    roi = [pts[0][0],pts[1][0],pts[0][1],pts[1][1]]
    
    time_stamps_frames = np.zeros((nframes,1))
    time_stamps_frames_property = np.zeros((nframes,1))
        
    # ***********************************************************************************
    # ******* Behavioral tracking variables *********************************************
    xy_positions = np.zeros((nframes,2),dtype=int)
    xy_nose_v = np.zeros((nframes,2),dtype=int)
    xy_tail_v = np.zeros((nframes,2),dtype=int)
    Sv = np.zeros((nframes,1),dtype=float)
    Rv = np.zeros((nframes,1),dtype=float)
    Ev = np.zeros((nframes,1),dtype=float)
    rhov = np.zeros((nframes,1),dtype=float)
    contours = []
    
    zone_IN = 5 + np.zeros((nframes,1),dtype=int)
    
    # ********** Behavior Tracking ******************************************************
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
    cap.set(cv2.CAP_PROP_FPS,frate_set)
                
    # ******** VIDEO ACQ ONSET SIGNAL *********
    ser.write(b'0')      # ON OPEN-EPHYS (TTL 1)
    k = 0
    # ****************************

    # ******** RECORDING ACQ + VIDEO TRACKING ********************************************    
    while(cap.isOpened()):
    
        # Take each frame
        ret, frame = cap.read()
        if ret == False:    
           print('MISSING FRAME')
           break
        
        elif ret == True:   
            frame_count = frame_count + 1
            
            if (frame_count == 1 or frame_count == nframes - 1):
                ser.write(b'2')  
            elif (frame_count > 1 and frame_count < nframes - 1):
                ser.write(b'1')
        
            if frame_count <= nframes:
                frame_n = frame_count
                fr_index = frame_n - 1
          
                # write frame
                out.write(frame)
                
                # ****** TIME MEASUREMENTS **********
                time_stamps_frames[(fr_index),0] = time.perf_counter() # based on time. function (see https://www.webucator.com/blog/2015/08/python-clocks-explained/)
                time_stamps_frames_property[(fr_index),0] = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 # based on VideoCapture:Get property PRO_POS_MSEC
                # ***********************************
                            
                # BGR_TO_GRAY COLOR CONVERSION
                # CROP frame to bahavior apparatus
                # SMOOTH GRAY frame
                # THRESHOLD the image
                # Bitwise-AND mask and original image
                # FIND CONTOURS in thresholded image, then grab the largest one
                                                        
                res, mask, cnts, c = frame_process(frame, roi, low_thre, high_thre)
                #contours.append(np.squeeze(c, axis=1))
                contours.append(c)
                
                pose_calculations(frame_count, cnts, c)
                countour_in, zoneB = trigger_zoneIN(frame, cX_frame,cY_frame,tg_zone_indx,box_temp,zone_IN,frame_count,videoWidth, videoHeight)                
                plot_tracking(frame,res,mask,c)
                        
            else:
                print(frame_count)
    #            out.release()     # End of video recording
                #cv2.imshow('frame',frame)
                cv2.imshow("Frame", frame)
                #cv2.namedWindow("win1");
                cv2.moveWindow("Frame", 1200,20);            
                cv2.imshow('Gray',res)
                #cv2.namedWindow("win2");
                cv2.moveWindow("Gray", 1200,460);            
                cv2.imshow('Mask',mask)
                #cv2.namedWindow("win3");
                cv2.moveWindow("Mask", 1200,615);
                
                k = k + 1
    
            if k == 1 :
                out.release()
                ser.write(b'3')              # OFF OPEN-EPHYS (THERE IS NO TTL EVENT RECORDED !!)
                print(k)   
                                       
            esc = cv2.waitKey(5) & 0xFF
            if esc == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
    
    #cap.release()
    #cv2.destroyAllWindows()
    
    # ******************** SAVING OUTPUT FILE************************** #
    #import scipy.io    
    # ARRAY TO BE SAVED
    save_file = params['output_dir'] + params['tracking_file']
    out_track = {
        'centroid_pos': xy_positions, 
        'nose_pos': xy_nose_v, 
        'tail_pos': xy_tail_v, 
        'b_area': Sv,'b_radius': Rv,
        'circ': Ev,'ellipt': rhov,
        'contours': contours
    }
    scipy.io.savemat(save_file, out_track)
    
    # **************** CLEAN-UP OF WORKSPACE VARIABLES **************** #
    
    #del cnts_pts, fourcc, fr_index, frame0, img, img_thresh 
    
#    del xy_positions, c, Ev, Rv, Sv, rhov, xy_nose_v, xy_tail_v
#    del frame, mask, res
#    del k, ret, save_file        
    print("Hello You")
    return time_stamps_frames, time_stamps_frames_property, frame_n
    