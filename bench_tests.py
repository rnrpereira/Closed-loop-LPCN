#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 23:58:31 2018

@author: rodrigo
"""

def bench_tests(frame_n):
    import numpy as np
    import scipy
    
    global time_stamps_frames, time_stamps_frames_property, params
    ## ********************************************************************* #
    ## BENCHMARK OF VIDEO_CAPTURE (WEBCAM)
    ## ********************************************************************* #
    #
    #import numpy as np    
    frate_set = params['frate_set']
    
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
    save_file1 = params['output_dir'] + params['vcap_bench_file1']
    save_file2 = params['output_dir'] + params['vcap_bench_file2']
    
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
    
    return video_cap_out, video_cap_out_opencv