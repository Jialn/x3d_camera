# Copyright (c) 2020. All Rights Reserved.
# used to process the dataset, like coping or updating the calib file
# usage: python x3d_camera/blender_ws/dataset_process_script.py '/media/ubuntu/pm951_jt/dataset_render/'

import os
import numpy as np
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    dataset_path = args.filepath
    dir_list = os.listdir(dataset_path)
    print(dir_list)
    for cur_file in dir_list:
        curr_path = dataset_path + cur_file
        if os.path.isfile(curr_path):
            pass
        if os.path.isdir(curr_path) and cur_file!="cali_imgs":
            cmd = "cp " + dataset_path + "calib.yml" + " " + curr_path + "/raw/"
            print("running cmd:" + cmd)
            os.system(cmd)
            cmd = "cp " + dataset_path + "calib.yml" + " " + curr_path + "/raw_no_light_bounces/"
            print("running cmd:" + cmd)
            os.system(cmd)

            # convert depth from 3 channle to 1 channel
            # img = cv2.imread(curr_path + '/left_depth_0001.png', cv2.IMREAD_UNCHANGED)
            # if len(img.shape) == 3:
            #     cv2.imwrite(curr_path + '/left_depth_0001.png', img[:,:,1], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # img = cv2.imread(curr_path + '/right_depth_0001.png', cv2.IMREAD_UNCHANGED)
            # if len(img.shape) == 3:
            #     cv2.imwrite(curr_path + '/right_depth_0001.png', img[:,:,1], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            
    exit()
