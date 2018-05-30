#%% CAPTURE BACKGROUD IMAGE/FRAME (FRAME_BCKG)
import cv2
#VIDEO RESOLUTION
videoWidth = 640
videoHeight = 360

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
ret0, frame_bckg = cap.read()

while ret0:    # frame captured without any errors
    cv2.imshow("Frame",frame_bckg)
    cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
    cv2.moveWindow("Frame", 1200,20);

    frame_bckg = cv2.cvtColor(frame_bckg, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Frame Backg",frame_bckg)
    cv2.namedWindow("Frame Backg",cv2.WINDOW_NORMAL)
    cv2.moveWindow("Frame Backg", 1200,445);
    
    # ******* SAVE BACKGROUD FRAMES *******#   
#    cv2.imwrite('/home/administrador/Programs/DAQ/frame_bckg.jpg',frame_bckg) #save image
#    cv2.imwrite('/home/administrador/Programs/DAQ/frame_bckg_backup.jpg',frame_bckg) #save image
    
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
         break

cap.release()
cv2.destroyAllWindows()
del k

#%% GRAY LEVEL THRESHOLDING TO ANALYZE BEHAVIOR TRACKING ONLINE

import cv2
import numpy as np
from roi_define import define_roi

# ************** GRAY LEVEL SETUP ********************** # 
def nothing(x):
    pass

# Create a black image, a window
#img = np.zeros((360,640,3), np.uint8)
#img = cv2.imread('/home/administrador/Programs/DAQ/my_photo-5.jpg')
#img_thresh = img

#VIDEO RESOLUTION
videoWidth = 640
videoHeight = 360

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
ret0, frame0 = cap.read()
img = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
img_thresh = img
cv2.namedWindow('image')
cv2.moveWindow("image", 1200,20);

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

#del frame0, img, imthre
del k, s, switch, ret0

#%% IMPORT the necessary PACKAGES
import imutils
import cv2
import time
import math
import numpy as np
import scipy.io
from roi_define import define_roi

# ABRIR PORTA SERIAL DE CONTROLE DO ARDUINO
import serial
ser = serial.Serial('/dev/ttyUSB0', 9600)
if(ser.isOpen() == False):
    ser.Open()

#%% GET POSE PARAMETERS FROM STATIC IMAGE CONTOUR
# Based on Wang Z., Mirbozorgi A.A. & Ghonvanloo M. (2017)
# An autmoated behavior analysis system for freely moving rodents using depth image

#% reset -f

#VIDEO RESOLUTION
videoWidth = 640
videoHeight = 360

# ********* VIDEO ACQ *********************************************************
# Time & FrameRate Settings
frate_set = 5.4 # FRAME RATE
rec_dur = 60   # time of recording video (in seconds)
nframes = int(rec_dur * frate_set)  # NUMBER OF FRAMES TO BE CAPTURED (N)

# Define the codec and create VideoWriter object
# works with MP4V (.MP4) & DIV4,3IVD,FMP4,DIVX,DX50,MP4V,XVID (ORDEM CRESENTE DE TAMANHO DO ARQUIVO DE VIDEO)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

file_path = '/home/administrador/Programs/DAQ/Data/'
out = cv2.VideoWriter((file_path + 'RatoVV01_testAcq+Track.mp4'),fourcc, frate_set, (videoWidth,videoHeight))

#n_of_frames = 120  
frame_count = 0
frame_n = 0
#frame_gr = np.zeros((480,640,3,n_of_frames))
time_stamps_frames = np.zeros((nframes,1))
time_stamps_frames_property = np.zeros((nframes,1))


# ***********************************************************************************

# ********** Behavior Tracking ******************************************************
xy_positions = np.zeros((nframes,2),dtype=int)
xy_nose_v = np.zeros((nframes,2),dtype=int)
xy_tail_v = np.zeros((nframes,2),dtype=int)
Sv = np.zeros((nframes,1),dtype=float)
Rv = np.zeros((nframes,1),dtype=float)
Ev = np.zeros((nframes,1),dtype=float)
rhov = np.zeros((nframes,1),dtype=float)
contours = []

 
#frame_count = 0
roi = [pts[0][0],pts[1][0],pts[0][1],pts[1][1]]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
cap.set(cv2.CAP_PROP_FPS,frate_set)

# ******** VIDEO ACQ *********
ser.write(b'0')                       # ON OPEN-EPHYS (TTL 1)
k = 0
# ****************************

while(cap.isOpened()):

    # Take each frame
    ret, frame = cap.read()
    if ret == False:    
       print('MISSING FRAME')
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
#            print(frame_n)
#            print(time_stamps_frames[fr_index])
#            print(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
#            print(k)
            # ***********************************
            
            # BGR_TO_GRAY COLOR CONVERSION
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # CROP frame to bahavior apparatus    
            gray_frame = gray[roi[2]:roi[3],roi[0]:roi[1]] 
            # SMOOTH GRAY frame
            gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)         
            # THRESHOLD the image        
            mask = cv2.threshold(gray_frame,low_thre,high_thre,cv2.THRESH_BINARY)[1]
            #mask = cv2.inRange(gray_frame,low_thre,high_thre)
            # Perform a series of EROSIONS + DILATIONS to remove any small regions of noise        
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)         
            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(gray_frame,gray_frame, mask= mask)                  
            # FIND CONTOURS in thresholded image, then grab the largest one  
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
#            if len(cnts) == 0:
#                print('NO CONTOURS DETECTED !!')        
#                break
#            else:
#                print('CONTOUR DETECTED !!')
            
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:2]
            c = cnts[0]  #max(cnts, key=cv2.contourArea)
            contours.append(np.squeeze(c, axis=1))
            
            # determine the most extreme points along the contour
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
        
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
        
            #### CALCULATIONS ######
            cnts_pts,_,_ = cnts[0].shape
            contour_dist = np.zeros((cnts_pts,1),dtype=int)
            tail_nose_dist = np.zeros((cnts_pts,1),dtype=int)
        
            
            M = cv2.moments(c)
            if M["m00"] != 0:   
                  cX = int((M["m10"] / M["m00"]))
                  cY = int((M["m01"] / M["m00"]))
                  xy_positions[frame_count-1][0] = cX
                  xy_positions[frame_count-1][1] = cY
#                  print('Centroid X = ',cX)
#                  print('Centroid Y = ',cY)
            else:
                  continue 
                        
            # BODY AREA (S)
            S = M["m00"]
            
            # BODY RADIUS (R)
            cv2.circle(res,(cX,cY), 2, (255, 255, 255), -1)
            for ii in range(cnts_pts):
                contour_dist[ii-1]  = math.hypot(cnts[0][ii-1,0][0] - cX,cnts[0][ii,0][1] - cY)   
                 
            max_dist = contour_dist.max()
            imax_nose = np.argmax(contour_dist) 
            R = contour_dist[imax_nose]
        
            # NOSE POINT (x_nose, y_nose)
            x_nose = cnts[0][imax_nose,0][0]
            y_nose = cnts[0][imax_nose,0][1]
            cv2.circle(res,(x_nose,y_nose), 3, (255, 255, 255), -1)
            cv2.line(res,(cX,cY),(x_nose,y_nose),(255,255,255),1)
            
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
work_dir = '/home/administrador/Programs/DAQ/'
file_name = 'tracking_output.mat'

# ARRAY TO BE SAVED
#arr_mat = xy_positions
save_file = work_dir + file_name
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

del xy_positions, E, M, R, S, a, b, c, rho, contour_dist, Ev, Rv, Sv, rhov, xy_nose_v, xy_tail_v
del ellipse, extBot, extTop, extLeft, extRight
del frame, gray, gray_frame, mask, res
del ii, k, ret, save_file 
del tail_nose_dist, x_nose, y_nose, x_tail, y_tail, xc, yc, cX, cY
del imax_nose, imax_tail_nose, max_dist, max_dist_tail_nose

# ***************************************************************** #


## ********************************************************************* #
## BENCHMARK OF VIDEO_CAPTURE (WEBCAM)
## ********************************************************************* #
#
#import numpy as np

rel_timestamps = time_stamps_frames - time_stamps_frames[0]
diff_rel_ts = np.diff(rel_timestamps,axis=0)
             
ifd = np.diff(time_stamps_frames,axis=0) # inter frame duration
average_frame_rate = 1/np.mean(ifd[0:(frame_n-1)])

mean_time_jitter = np.mean(np.absolute(diff_rel_ts - 1/frate_set))

inst_frame_rate = 1/ifd
stdev_frame_rate = np.std(inst_frame_rate)

df = np.diff(time_stamps_frames,axis=0)


#print('Average frame rate = ',end=" ")
#print(average_frame_rate)
#
#print('Stdev frame rate = ',end=" ")
#print(stdev_frame_rate)


# Based on VideoCapture:Get property PRO_POS_MSEC
rel_timestamps_prop = (time_stamps_frames_property - time_stamps_frames_property[0])

diff_rel_ts_prop = np.diff(rel_timestamps_prop,axis=0)

ifd_prop = np.diff(time_stamps_frames_property,axis=0) # inter frame duration
average_frame_rate_prop = 1/np.mean(ifd_prop[0:(frame_n-1)])

mean_time_jitter_prop = np.mean(np.absolute(diff_rel_ts_prop - (1/frate_set)))

inst_frame_rate_prop = (1/ifd_prop)
stdev_frame_rate_prop = np.std(inst_frame_rate_prop)
df_prop = np.diff(time_stamps_frames_property,axis=0)

# ARRAY TO BE SAVED

file_name1 = 'vcap_params.mat'
file_name2 = 'vcap_params_prop.mat'
save_file1 = work_dir + file_name1
save_file2 = work_dir + file_name2

# OUTPUT => VIDEO CAPTURE PARAMETERS
video_cap_out = {'rel_timestamps': rel_timestamps,
    'diff_rel_ts': diff_rel_ts, 'ifd': ifd, 
    'average_frame_rate': average_frame_rate,
    'mean_time_jitter': mean_time_jitter,
    'stdev_frame_rate': stdev_frame_rate,'df': df
}
video_cap_out_opencv = {'rel_timestamps_prop': rel_timestamps_prop, 
    'diff_rel_ts_prop': diff_rel_ts_prop, 
    'ifd_prop': ifd_prop, 
    'average_frame_rate_prop': average_frame_rate_prop,
    'mean_time_jitter_prop': mean_time_jitter_prop,
    'stdev_frame_rate_prop': stdev_frame_rate_prop,
    'df_prop': df_prop
}

scipy.io.savemat(save_file1, video_cap_out)
scipy.io.savemat(save_file2, video_cap_out_opencv)

del rel_timestamps, diff_rel_ts, ifd, average_frame_rate, mean_time_jitter, stdev_frame_rate, df, inst_frame_rate
del rel_timestamps_prop, diff_rel_ts_prop, ifd_prop, average_frame_rate_prop, mean_time_jitter_prop, stdev_frame_rate_prop, df_prop, inst_frame_rate_prop
del time_stamps_frames, time_stamps_frames_property

del esc, file_name, file_name1, file_name2, fr_index
del img, img_thresh, frame0
# ********************************************************************* #

#%% IMPORTANT

#    def angle(a, b, c):

        # Create vectors from points
        ct = [x_tail - cX, y_tail - cY]
        cn = [x_nose - cX, y_nose - cY] 

        # Normalize vector
#        nba = sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
#        ba = [ba[0]/nba, ba[1]/nba, ba[2]/nba]
        v1_norm = math.sqrt(ct[0]**2 + ct[1]**2)
        v2_norm = math.sqrt(cn[0]**2 + cn[1]**2)
        
        # Calculate scalar product from vectors
        v1v2 = ct[0]*cn[0] + ct[1]*cn[1]
        
        # calculate the angle in radian
        body_angle = math.acos(v1v2/(v1_norm*v2_norm))
        body_angle_deg = math.degrees(body_angle)
        
        
        
        ## NECK ANGLE & HEAD ANGLE
        
        k = 0.3
        
        x_neck = cX + k * (cX - x_tail)
        y_neck = cY + k * (cY - y_tail)
        
        nt = [x_tail - x_neck,y_tail - y_neck]
        nn = [x_nose - x_neck, y_nose - y_neck]
        
        x_neck = int(x_neck)
        y_neck = int(y_neck)
        
        v1_neck_norm = math.sqrt(nt[0]**2 + nt[1]**2)
        v2_neck_norm = math.sqrt(nn[0]**2 + nn[1]**2)
        
        v1v2_neck = nt[0]*nn[0] + nt[1]*nn[1]
        
        neck_angle = math.acos(v1v2_neck/(v1_neck_norm*v2_neck_norm))
        neck_angle_deg = math.degrees(neck_angle)
        
        head_angle = math.pi - neck_angle
        head_angle_deg = math.degrees(head_angle)