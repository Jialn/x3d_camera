# Copyright (c) 2020. All Rights Reserved.
# Created on 2021-04-01
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: xForward 3D Camera structure light algorithm pipeline

These file implement the core functions of xForward structured light 3D camera pipeline.
"""

import numpy as np
import numba
from numba import prange
numba.config.NUMBA_DEFAULT_NUM_THREADS=8

from numba.pycc import CC
cc = CC('structured_light_core')

@cc.export('gray_decode', (numba.uint8[:,:,:], numba.uint8[:,:,:], numba.uint8[:,:], numba.int64, numba.int64, numba.int64, numba.int16[:,:], numba.float64))
@numba.jit ((numba.uint8[:,:,:], numba.uint8[:,:,:], numba.uint8[:,:], numba.int64, numba.int64, numba.int64, numba.int16[:,:], numba.float64),nopython=True, parallel=True, cache=True)
def gray_decode(src, imgs_thresh, valid_map, image_num, height, width, img_index, unvalid_thres):
    for h in prange(height):
        for w in range(width):
            if valid_map[h,w] == 0:
                img_index[h,w] = -9999
                continue
            bin_code = 0
            current_bin_code_bit = 0
            for i in range(0, image_num):
                if src[i][h,w]>=imgs_thresh[i][h,w]+unvalid_thres:
                    current_bin_code_bit = current_bin_code_bit ^ 1
                elif src[i][h,w]<=imgs_thresh[i][h,w]-unvalid_thres:
                    current_bin_code_bit = current_bin_code_bit ^ 0
                else:
                    bin_code = -9999
                    break
                bin_code += (current_bin_code_bit <<  (image_num-1-i))
            img_index[h,w] = bin_code

""" Testing feature, sub pixel optimization for index
example:
   794   793   792   791   790   789   787   786   785   784   783   782
   ->
   794   793   792   791   790   789 - 787   786   785   784   783   782
   0     1     2     3     4     5     6     7     8     9     10    11
   -1.4  0.5   1.4   2.3   3.2   4.1 - 6.9   7.8   8.7   9.6   10.5  11.4
"""
@cc.export('index_sub_pix', (numba.float64[:,:], numba.int16[:,:], numba.int64, numba.int64, numba.float64))
@numba.jit ((numba.float64[:,:], numba.int16[:,:], numba.int64, numba.int64, numba.float64), nopython=True, parallel=True, nogil=True, cache=True)
def index_sub_pix(img_index_sub_pix, img_index, height, width, expected_pixel_per_index):
    # expected_pixel_per_index_int = round(expected_pixel_per_index)
    thres = 2  # expected_pixel_per_index_int + 1
    check_margin = 36
    for h in prange(height):
        line = img_index[h,:]
        w = check_margin
        while (w < width-check_margin):
            if line[w] == -9999:   # unvalid
                w += 1
                continue
            elif line[w-1] - line[w] == 2:  # possible un-consistant point
                i = 1
                # print("checkingl")
                unconsis_l = check_margin
                while(i<check_margin):
                    if(line[w-i-1] - line[w-i] == thres):
                        unconsis_l = w-i-1
                        break
                    elif(line[w-i-1] - line[w-i] > thres):
                        unconsis_l = -9999
                        break
                    i += 1
                i = 0
                unconsis_r = check_margin
                # print("checkingr")
                while(i<check_margin):
                    if(line[w+i] - line[w+i+1] == thres):
                        unconsis_r = w+i+1
                        break
                    elif(line[w+i] - line[w+i+1] > thres):
                        unconsis_r = -9999
                        break
                    i += 1
                # print("checking length")
                if unconsis_l != -9999 and unconsis_r != -9999:
                    #print("find unconsis point!")
                    #print((unconsis_l, w-1, w, unconsis_r, h))
                    unconsis_len = (unconsis_r - unconsis_l) // 2 + 1
                    left_dis = w - 1 - unconsis_l
                    right_dis = unconsis_r - w
                    if left_dis>=4 and right_dis>=4 and abs(left_dis - right_dis) <= 3:
                        unconsis_offset = unconsis_len / (unconsis_len + 1.0)
                        #print("unconsis_offset")
                        #print(unconsis_offset)
                        unconsis_offset *= 0.5
                        for i in range(unconsis_len//2):
                            img_index_sub_pix[h,w-1-i] = w-1-i - unconsis_offset/(i+1.0)
                        for i in range(unconsis_len//2):
                            img_index_sub_pix[h,w+i] = w+i + unconsis_offset/(i+1.0)
                        #print(range(unconsis_l,unconsis_r))
                        #print(img_index_sub_pix[h,unconsis_l:unconsis_r])
                        w = unconsis_r-2
            w += 1

@cc.export('get_dmap_from_index_map', (numba.float64[:,:],numba.float64[:,:], numba.int64,numba.int64, numba.int16[:,:],numba.int16[:,:], numba.float64,numba.float64,numba.float64, numba.float64[:,:],numba.float64[:,:]))
@numba.jit ((numba.float64[:,:],numba.float64[:,:], numba.int64,numba.int64, numba.int16[:,:],numba.int16[:,:], numba.float64,numba.float64,numba.float64, numba.float64[:,:],numba.float64[:,:] ), nopython=True, parallel=True, nogil=True, cache=True)
def get_dmap_from_index_map(dmap,depth_map, height,width, img_index_left,img_index_right, baseline,dmap_base,fx, img_index_left_sub_px,img_index_right_sub_px):
    max_allow_pixel_per_index = 1 + width // 600  # some typical condition: 1 for 640, 3 for 1280, 5 for 2560, 8 for 4200
    right_corres_point_offset_range = width // 10
    for h in prange(height):
        line_r = img_index_right[h,:]
        line_l = img_index_left[h,:]
        possible_points_r = np.zeros(width, dtype=np.int64)
        last_right_corres_point = -1
        for w in range(width):
            if line_l[w] == -9999:   # unvalid
                last_right_corres_point = -1
                continue
            ## find possible right indicator
            cnt = 0
            if last_right_corres_point > 0:
                checking_left_edge = last_right_corres_point - right_corres_point_offset_range
                checking_right_edge = last_right_corres_point + right_corres_point_offset_range
                if checking_left_edge <=0: checking_left_edge=0
                if checking_right_edge >=width: checking_left_edge=width
            else:
                checking_left_edge, checking_right_edge = 0, width
            for i in range(checking_left_edge, checking_right_edge):
                if line_r[i] == line_l[w]:
                    possible_points_r[cnt] = i
                    cnt += 1
            if cnt == 0:
                cnt_l, cnt_r = 0, 0
                for i in range(width):
                    if line_r[i] == line_l[w]-1:
                        possible_points_r[cnt_r+cnt_l] = i
                        cnt_l += 1
                    elif line_r[i] == line_l[w]+1:
                        possible_points_r[cnt_r+cnt_l] = i
                        cnt_r += 1
                if cnt_l > 0 and cnt_r > 0: cnt = cnt_l + cnt_r
                else: continue
            ## find right indicator w_r in possible_points
            w_r = 0.0
            for i in range(cnt): 
                p = possible_points_r[i]
                if img_index_right_sub_px[h, p] > 0.001: w_r += img_index_right_sub_px[h, p]
                else: w_r += p
            w_r /= cnt
            # check right outliers
            outlier_flag_r = False
            for i in range(cnt): 
                p = possible_points_r[i]
                if abs(p-w_r) >= max_allow_pixel_per_index: outlier_flag_r=True
            if outlier_flag_r: continue
            last_right_corres_point = round(w_r)
            ## refine left index around w
            w_l, w_l_cnt = 0.0, 0
            for i in range(w-max_allow_pixel_per_index, min(w+max_allow_pixel_per_index+1, width)):
                if img_index_left[h,i]==img_index_left[h,w]:
                    w_l_cnt += 1
                    if img_index_left_sub_px[h, i] > 0.001: w_l += img_index_left_sub_px[h, i]
                    else: w_l += i
            # check left outliers
            outlier_flag_l = False
            if w_l_cnt == 1:  # if only one near the checking range has the index, consider it could be an outliers  
                cnt = 0
                for i in range(width):
                    if line_l[i] == line_l[w]: cnt += 1
                    if cnt >= 2:
                        outlier_flag_l = True
                        break
            if outlier_flag_l: continue
            w_l = w_l / w_l_cnt
            ## stereo diff and depth
            stereo_diff = dmap_base + w_l - w_r
            dmap[h,w] =  (w_l - w_r)
            if stereo_diff > 0.000001:
                depth = fx * baseline / stereo_diff
                if 0.1 < depth < 2.0:
                    depth_map[h, w] = depth
                    # if subpix optimization is used
                    if img_index_left_sub_px[h, w] > 0.001 and img_index_left_sub_px[h, w-2] > 0.001:
                        if abs(depth_map[h, w] - depth_map[h, w-2]) < 0.01:
                            dis_r = img_index_left_sub_px[h, w] - (w-1)
                            dis_l = (w-1) - img_index_left_sub_px[h, w-2]
                            if dis_l > 0.0001 and dis_r > 0.0001:
                                diff = depth_map[h, w] - depth_map[h, w-2]
                                inter_depth_w_1 = depth_map[h, w-2] + diff * dis_l / (dis_l + dis_r)
                                # print(depth_map[h, w-1] - inter_depth_w_1)
                                if abs(depth_map[h, w-1] - inter_depth_w_1) < 0.002:
                                    depth_map[h, w-1] = inter_depth_w_1
                                    img_index_left_sub_px[h, w-1] = w-1

@cc.export('depth_filter', (numba.float64[:,:], numba.float64[:,:], numba.int64, numba.int64, numba.float64[:,:]))
@numba.jit ((numba.float64[:,:], numba.float64[:,:], numba.int64, numba.int64, numba.float64[:,:]), nopython=True, parallel=True, nogil=True, cache=True)
def depth_filter(depth_map, depth_map_raw, height, width, camera_kp):
    # a point could be considered as not flying when: points in checking range below max_distance > minmum num 
    max_distance = 0.005  # about 5-7 times of resolution per pxiel
    minmum_point_num_in_range = 2 + width // 400  # including the point itsself
    checking_range_in_meter = max_distance * 1.2
    checking_range_limit = width // 50
    fx, cx, fy, cy = camera_kp[0][0], camera_kp[0][2], camera_kp[1][1], camera_kp[1][2]
    for h in prange(height):
        w = 0
        while w < width:
            if depth_map_raw[h,w] != 0:
                point_x = depth_map_raw[h,w] * (w - cx) / fx
                point_y = depth_map_raw[h,w] * (h - cy) / fy
                checking_range_in_pix_x = (int)(checking_range_in_meter * fx / depth_map_raw[h,w])
                checking_range_in_pix_y = (int)(checking_range_in_meter * fy / depth_map_raw[h,w])
                checking_range_in_pix_x = min(checking_range_in_pix_x, checking_range_limit)
                checking_range_in_pix_y = min(checking_range_in_pix_y, checking_range_limit)
                is_not_flying_point_flag = 0
                for i in range(h-checking_range_in_pix_y, min(height, h+checking_range_in_pix_y+1)):
                    for j in range(w-checking_range_in_pix_x, min(width, w+checking_range_in_pix_x+1)):
                        curr_x = depth_map_raw[i,j] * (j - cx) / fx
                        curr_y = depth_map_raw[i,j] * (i - cy) / fy
                        distance = np.square(curr_x - point_x) + np.square(curr_y - point_y) + np.square(depth_map_raw[h,w] - depth_map_raw[i,j])
                        if distance < np.square(max_distance):
                            is_not_flying_point_flag += 1
                if is_not_flying_point_flag <= minmum_point_num_in_range: # unvalid the point
                    depth_map[h,w] = 0
                elif is_not_flying_point_flag >= checking_range_in_pix_x * checking_range_in_pix_y - 1:
                    w += checking_range_in_pix_x
                    continue
            w += 1

@cc.export('depth_avg_filter', (numba.float64[:,:], numba.float64[:,:], numba.int64, numba.int64, numba.float64[:,:]))
@numba.jit ((numba.float64[:,:], numba.float64[:,:], numba.int64, numba.int64, numba.float64[:,:]), nopython=True, parallel=True, nogil=True, cache=True)
def depth_avg_filter(depth_map, depth_map_raw, height, width, camera_kp):
    filter_max_length = 4  # from 0 - 5
    filter_weights = np.array([1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.3])
    filter_thres = 0.0025
    # run the horizontal filter
    for h in prange(height): 
        for w in range(width):
            if depth_map[h,w] != 0:
                left_weight, right_weight, depth_sum = 0.0, 0.0, depth_map[h,w]*filter_weights[0]
                for i in range(1, filter_max_length+1):
                    l_idx, r_idx = w-i, w+i
                    if(depth_map[h,l_idx] != 0 and depth_map[h,r_idx] != 0 and l_idx > 0 and r_idx < width and \
                        abs(depth_map[h,l_idx] - depth_map[h,w]) < filter_thres and abs(depth_map[h,r_idx] - depth_map[h,w]) < filter_thres):
                        left_weight += filter_weights[i]
                        right_weight += filter_weights[i]
                        depth_sum += (depth_map[h,r_idx] + depth_map[h,l_idx]) * filter_weights[i]
                    else: break
                depth_map[h,w] = depth_sum / (filter_weights[0] + left_weight + right_weight)
    # run the vertical filter
    for w in prange(width):
        for h in range(height): 
            if depth_map[h,w] != 0:
                left_weight, right_weight, depth_sum = 0.0, 0.0, depth_map[h,w]*filter_weights[0]
                for i in range(1, filter_max_length+1):
                    l_idx, r_idx = h-i, h+i
                    if(depth_map[l_idx,w] != 0 and depth_map[r_idx,w] != 0 and l_idx > 0 and r_idx < height and \
                        abs(depth_map[l_idx,w] - depth_map[h,w]) < filter_thres and abs(depth_map[r_idx,w] - depth_map[h,w]) < filter_thres):
                        left_weight += filter_weights[i]
                        right_weight += filter_weights[i]
                        depth_sum += (depth_map[r_idx,w] + depth_map[l_idx,w]) * filter_weights[i]
                    else: break
                depth_map[h,w] = depth_sum / (filter_weights[0] + left_weight + right_weight)



# compile ahead of time for faster running: 
#   python x3d_camera/structured_light_core.py
#   then you can import lib.structured_light_core as a pre-compiled library
#   compile ahead of time does not support parallel
if __name__ == "__main__":
    cc.output_dir=r'x3d_camera/lib/'
    cc.target_cpu=r'host'
    cc.verbose=True
    cc.compile()