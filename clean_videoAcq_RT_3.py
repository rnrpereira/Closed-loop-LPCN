#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:29:53 2018

@author: rodrigo
"""
# *** TASK No.0 ************************************************************************************************* #
global params, frame_n
params = {
        'output_dir' : '/home/administrador/Programs/DAQ/',
        'tracking_file' : 'tracking_output.mat',
        'video_file' : 'RatoVV01_testAcq+Track.mp4',
        'vcap_bench_file1' : 'vcap_params.mat',
        'vcap_bench_file2' : 'vcap_params_prop.mat',
        'videoWidth': 640,
        'videoHeight': 360,
        'frate_set': 5.4,
        'rec_dur': 60,
        'nzones': 2,
        'target_zone': 1,
        'alpha': 0.2,
        'fontColor_zone':[(0,0,255),(0,255,0),(0,255,255),(255,0,0)]
}
# ************************************************************************************************************** #

# *** TASK No.1 - SET BACKGROUND IMAGE + INTENSITY TTHRESH + TRACKING ROI **************************************** #
from capt_bckg import capture_bckg
from grayThre_roi3 import roi_thre3
# CAPTURE BACKGROUD IMAGE/FRAME (FRAME_BCKG) - File capt_bckg.py
capture_bckg(params['videoWidth'],params['videoHeight'])
# GRAY LEVEL THRESHOLDING TO ANALYZE BEHAVIOR TRACKING ONLINE - File grayThre.py
low_thre, high_thre, pts, zones_xy = roi_thre3(nzones=params['nzones']) 
params.update({'zones_xy':zones_xy})

# *** TASK No. 2 - OPEN AARDUINO CONTROL PORT ******************************************************************** #
# ABRIR PORTA SERIAL DE CONTROLE DO ARDUINO
import serial
ser = serial.Serial('/dev/ttyUSB0', 9600)
if(ser.isOpen() == False):
    ser.Open()
    
# *** TASK No. 3 - VIDEO ACQUISITION + BEHAVIOR TRACKING ********************************************************* #
import acq_main
time_stamps_frames, time_stamps_frames_property, frame_n = acq_main(low_thre,high_thre,pts,params,ser)

# *** TASK No. 4 - BENCHMARK VIDEO ACQUISITION ******************************************************************* #
import bench_tests
video_cap_out, video_cap_out_opencv = bench_tests(frame_n)

# ********************************************************************* #
