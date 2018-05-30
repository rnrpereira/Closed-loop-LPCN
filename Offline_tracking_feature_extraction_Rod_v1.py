#%% CAPTURE BACKGROUD IMAGE/FRAME (FRAME_BCKG)
import cv2
#VIDEO RESOLUTION
videoWidth = 640
videoHeight = 360

cap = cv2.VideoCapture('/home/administrador/Programs/DAQ/Scripts PYTHON/my_video-5.mkv')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
ret0, frame_bckg = cap.read()

#frame_bckg = frame_bckg[172:260,7:638]

while ret0:    # frame captured without any errors    
    # CONVERT frames from BGR to GRAY
    if (len(frame_bckg.shape) == 3):
       frame_bckg_gray = cv2.cvtColor(frame_bckg, cv2.COLOR_BGR2GRAY)
    else:
       frame_bckg_gray = frame_bckg   

    cv2.imshow("Frame Backg",frame_bckg)
    cv2.namedWindow("Frame Backg",cv2.WINDOW_NORMAL)
    cv2.moveWindow("Frame Backg", 700,20);
    
    cv2.imshow("Frame Backg_Gray",frame_bckg_gray)
    cv2.namedWindow("Frame Backg_Gray",cv2.WINDOW_NORMAL)
    cv2.moveWindow("Frame Backg_Gray", 700,445);
    
    # ******* SAVE BACKGROUD FRAMES *******#   
#    cv2.imwrite('/home/administrador/Programs/DAQ/frame_bckg.jpg',frame_bckg) #save image
#    cv2.imwrite('/home/administrador/Programs/DAQ/frame_bckg_backup.jpg',frame_bckg) #save image
    
    k = cv2.waitKey(1) & 0xFF
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

del k

#%% SET GRAY LEVEL THRESHOLDING AND REGION-OF-INTEREST (ROI) TO ANALYZE BEHAVIOR TRACKING ONLINE

import cv2
import numpy as np
from roi_define import define_roi
from roi_define3 import define_roi3

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

filename = '/home/administrador/Programs/DAQ/Scripts PYTHON/my_video-5.mkv'
#filename ='/media/administrador/SAMSUNG/Projeto Autismo/DS3/PD 14/Bridge alone/guvcview_video-5.mkv'
cap = cv2.VideoCapture(filename)
ret0, frame0 = cap.read()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)

# CONVERT frames from BGR to GRAY
if (len(frame0.shape) == 3):
   img = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
else:
   img = frame0

img_thresh = img


# **** IF IMAGE IN DARKER THAN BACKGROUND *****
#img = 255 - img
# *********************************************

cv2.namedWindow('image')
cv2.moveWindow("image", 600,20);

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

nzones = 4
pts3, zones_xy = define_roi3(frame0,nzones)

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

#del frame0, img, imthre
del k, s, switch, ret0

# ************************************************************************************************* #


#%% REAL-TIME TRACKING WITH CENTROID USING MEANSHIFT ALGORITHM - FUNCIONA !! (# CELL 2)

#%reset -f

from colorama import Fore, Back
import cv2
#import numpy as np
import imutils
import math
from roi_define import define_roi

#cap = cv2.VideoCapture('/Users/rodrigo/Documents/Python Programming/video_test_RTT3.avi')
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,videoWidth)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,videoHeight)
#ret0, frame0 = cap.read()
#pts = define_roi(frame0)
#cap.release()

# Write some Text
font                    = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText     = (int(0.7 * videoWidth),int(0.05 * videoHeight))
topRightCornerOfText    = (int(0.7 * videoWidth),int(0.1 * videoHeight))
fontScale               = 0.5
fontColor1              = (255,150,0)
fontColor2              = (0,0,255)
lineType                = 1
# Transparency of target zone
alpha = 0.2  

# Zone
fontColor_zone          = [(0,0,255),(0,255,0),(0,255,255),(255,0,0)]

  
frame_count = 0

roi = [pts[0][0],pts[1][0],pts[0][1],pts[1][1]]
#roi_width = pts[1][0] - pts[0][1]
#roi_height = pts[1][1] - pts[0][1]

cap = cv2.VideoCapture(filename)

videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rec_dur = total_frames * fps   # recording duration (in SECONDS)

xy_positions = np.zeros((total_frames,2),dtype=int)
zone_IN = 5 + np.zeros((total_frames,1),dtype=int)

xy_nose_v = np.zeros((total_frames,2),dtype=int)
xy_tail_v = np.zeros((total_frames,2),dtype=int)
Sv = np.zeros((total_frames,1),dtype=float)
Rv = np.zeros((total_frames,1),dtype=float)
Ev = np.zeros((total_frames,1),dtype=float)
rhov = np.zeros((total_frames,1),dtype=float)

contours = []

# BOX ARENA
box_arena = np.asarray([
        [[pts[0][0],pts[0][1]]],
        [[pts[0][0],pts[1][1]]],
        [[pts[1][0],pts[1][1]]],
        [[pts[1][0],pts[0][1]]]
        ])
box_arena = np.int32(box_arena)

# TARGET ZONE     <<<<====== IMPORTANT !!
target_zone = 3
tg_zone_indx = target_zone - 1

# TARGET ZONES         

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

# ************  BACKGROUD SUBTRACTION
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorMOG2()

# ************  BACKGROUD SUBTRACTION

while(True):

    # Take each frame
    ret, frame = cap.read()

    if not(ret):
        break
        
    frame_count = frame_count + 1
    print('Frame = ', frame_count)

    
    # CONVERT frames from BGR to GRAY
    if (len(frame.shape) == 3):
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
       gray = frame
    
    # ************  APPLY BACKGROUD SUBTRACTION ************
#    frame = fgbg.apply(frame)
    
    # CROP the frame to the bahavior apparatus    
    gray_frame = gray[roi[2]:roi[3],roi[0]:roi[1]] 
    
    
    # **** IF IMAGE IN DARKER THAN BACKGROUND *****
#    gray_frame = 255 - gray_frame
    # *********************************************    
    
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # THRESHOLD the GRAY image to get only the RAT
    # (lower_white, upper_white)=> RED-light (10,250); BRIGHT-light (80, 250)
    mask = cv2.inRange(gray_frame,low_thre,high_thre)
    #ret,thresh = cv2.threshold(gray_frame,127,255,0)
    #    mask2 = mask[roi[0]:roi[1],roi[2]:roi[3]]
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(gray_frame,gray_frame, mask= mask)

   # Make sure to use a clone of the image in cv2.findContours (e.g. mask2.copy())
   # The cv2.findContours method is destructive (meaning it manipulates the image you pass in).
   # so if you plan on using that image again later, be sure to clone it.
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        print('NO CONTOURS DETECTED !!')        
        break
    else:
        print('CONTOUR DETECTED !!')

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:2]
    c = cnts[0]
    contours.append(np.uint16(c))
    
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    extLeft_frame = tuple(c[c[:, :, 0].argmin()][0] + [roi[0],roi[2]])
    extRight_frame = tuple(c[c[:, :, 0].argmax()][0] + [roi[0],roi[2]])
    extTop_frame = tuple(c[c[:, :, 1].argmin()][0] + [roi[0],roi[2]])
    extBot_frame = tuple(c[c[:, :, 1].argmax()][0] + [roi[0],roi[2]])

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
          print('Centroid X = ',cX)
          print('Centroid Y = ',cY)
    else:
          continue 
    
    contour_in = cv2.pointPolygonTest(box_temp[tg_zone_indx],(cX_frame,cY_frame),False)
    
    zone_IN[frame_count-1] = np.int(contour_in)
    if (contour_in == 1) or (contour_in == 0):
        zoneB = 'IN'
        cv2.circle(frame,(int(0.1*videoWidth),int(0.1*videoHeight)), 2, (0, 255, 0), 16)
    else:
        zoneB = 'OUT'
        cv2.circle(frame,(int(0.1*videoWidth),int(0.1*videoHeight)), 2, (0, 0, 0), 16)

    # BODY AREA (S)
    S = M["m00"]
    
    # BODY RADIUS (R)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    
    cX_frame = cX + roi[0]
    cY_frame = cY + roi[2]
    
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
    cv2.line(res,(cX,cY),(x_nose ,y_nose),(255,255,255),1)
    
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
    #    tail_nose_dist[ii-1]  = math.hypot(cnts[0][ii-1,0][0] - cX,cnts[0][ii,0][1] - cY) + math.hypot(x_nose-cX,y_nose - cY)
    
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

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(frame,[c+[[roi[0],roi[2]]]], -2, (0, 255, 255), 1)
    cv2.circle(frame, extLeft_frame, 1, (0, 0, 255), -1)
    cv2.circle(frame, extRight_frame, 1, (0, 255, 0), -1)
    cv2.circle(frame, extTop_frame, 1, (255, 0, 0), -1)
    cv2.circle(frame, extBot_frame, 1, (255, 255, 0), -1)
    
    cv2.drawContours(res, [c], -1, (255, 255, 255), 1)
    cv2.circle(res, extLeft, 1, (255, 255, 255), -1)
    cv2.circle(res, extRight, 1, (255, 255, 255), -1)
    cv2.circle(res, extTop, 1, (255, 255, 255), -1)
    cv2.circle(res, extBot, 1, (255, 255, 255), -1)
    
    cv2.drawContours(mask,c, -1, (255, 255, 255), 1)
    
    cv2.circle(res,(x_tail,y_tail), 3, (255, 255, 255), -1)
    cv2.line(res,(cX,cY),(x_tail,y_tail),(255,255,255),1)
    #
    cv2.circle(frame,(cX_frame,cY_frame), 2, (255,255, 255), -1)
    cv2.circle(frame,(x_nose+roi[0],y_nose+roi[2]), 3, (0, 0, 255), -1)
    cv2.line(frame,(cX_frame,cY_frame),(x_nose+roi[0],y_nose+roi[2]),(0,255,0),1)
    cv2.circle(frame,(x_tail+roi[0],y_tail+roi[2]), 3, (0, 255, 0), -1)
    cv2.line(frame,(cX_frame,cY_frame),(x_tail+roi[0],y_tail+roi[2]),(0,255,0),1)                   

    print('Ellipse Center = ', ellipse[0])
    print('MajAx,MinAx length = ', ellipse[1])    
    print('Rot_angle = ', ellipse[2])
    print('Zone Box Vx1 = ', box_temp[0][0])
    print('Zone Box Vx2 = ', box_temp[0][3])
    print('Body_Center in-zone: ', Fore.RED + zoneB)
    print(Fore.RESET+ Back.RESET,'########################')
        
    # draw the contour and center of the shape on the image
    cv2.drawContours(mask,[c], -1, (255, 255, 255), 1)
    cv2.drawContours(res,[c], -1, (255, 255, 255), 1)

    cv2.circle(mask, (cX, cY), 2, (255, 255, 255), 2)
    cv2.drawContours(mask,[box_temp[0]],0,(0,0,255),1)    
    cv2.circle(res, (cX, cY), 2, (255, 255, 255), 2)   

    overlay = frame.copy()
    cv2.circle(frame, (cX_frame, cY_frame), 2, (0, 255, 0), 2)
    cv2.circle(overlay, (cX_frame, cY_frame), 2, (0, 255, 0), 2)
    
#    cv2.drawContours(frame,[box_tgzone],0,fontColor2,-1) # Draw TARGET ZONE
    for ii in range(len(box_temp)):
      cv2.drawContours(frame,[box_temp[ii]],0,fontColor_zone[ii],-1) # Draw TARGET ZONES
   
#    cv2.ellipse(frame,ellipse_frame,(255,255,255),1)
    
    cv2.putText(overlay,'BEHAVIOR ARENA',topLeftCornerOfText, 
    font, fontScale, fontColor1,lineType,cv2.LINE_AA)
    
    cv2.putText(overlay,'TARGET ZONE',topRightCornerOfText, 
    font, fontScale, fontColor2,lineType,cv2.LINE_AA)

    cv2.addWeighted(frame, alpha,overlay,1 - alpha,0,overlay)
    
    cv2.drawContours(overlay,[box_arena],0,fontColor1,2) # Draw ARENA
       
    cv2.drawContours(res,[box_temp[0]],0,(0,0,255),1)
    cv2.ellipse(res,ellipse,(255,255,255),1)

#         cv2.line(im,ellipse) 

    cv2.imshow('Mask',mask)
    cv2.moveWindow("Mask", 350,20);
    cv2.imshow('FrameMask',res)
    cv2.moveWindow("FrameMask", 350,170);
    cv2.imshow('Frame',overlay)
    cv2.moveWindow("Frame", 350,320);    
   
    k = cv2.waitKey(5) & 0xFF
    if frame_count == total_frames or k == 27:
        break
    cv2.waitKey(10)
    
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

# ******************** SAVING OUTPUT FILE************************** #
import scipy.io
work_dir = '/home/administrador/Programs/DAQ/'
file_name = 'tracking_output.mat'

# ARRAY TO BE SAVED
#arr_mat = xy_positions
save_file = work_dir + file_name
out_track = {'centroid_pos': xy_positions, 'nose_pos': xy_nose_v, 'tail_pos': xy_tail_v, 'b_area': Sv,'b_radius': Rv,'circ': Ev,'ellipt': rhov,}
scipy.io.savemat(save_file, out_track)

# **************** CLEAN-UP OF WORKSPACE VARIABLES **************** #

del xy_positions, E, M, R, S, a, b, c, rho, contour_dist, Ev, Rv, Sv, rhov, xy_nose_v, xy_tail_v
del ellipse, extBot, extTop, extLeft, extRight
del frame, gray, gray_frame, mask, res
del ii, k, ret, save_file 
del tail_nose_dist, x_nose, y_nose, x_tail, y_tail, xc, yc, cX, cY
del imax_nose, imax_tail_nose, max_dist, max_dist_tail_nose

# ***************************************************************** #
