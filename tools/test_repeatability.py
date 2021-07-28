# Copyright (c) 2020. All Rights Reserved.
#
# usage: python x3d_camera/blender_ws/test_repeatability.py '/media/ubuntu/pm951_jt/dataset_render/'

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
    if dataset_path[-1] != '/': dataset_path = dataset_path + "/"
    print(dir_list)
    images = []
    for cur_file in dir_list:
        curr_path = dataset_path + cur_file
        if os.path.isfile(curr_path):
            print(curr_path)
            img = cv2.imread(curr_path, cv2.IMREAD_UNCHANGED)
            img = img * 30.0 # to mm
            images.append(img)
        if os.path.isdir(curr_path):
            pass 
    print(len(images))
    img_num = len(images)
    print(images)
    img_sum = images[0]
    for img in images[1:]:
        img_sum = img_sum + img

    for img in images:
        unvalid_points = np.where(img==0.0)
        print(unvalid_points)
        img_sum[unvalid_points] = 0
    
    img_avg = img_sum / img_num

    repeatabilities = []
    for img in images:
        error = img - img_avg
        valid_points = np.where(img_avg>0.00000001)
        error = error[valid_points]
        # print(error)
        # print(np.sum(error))
        rep = np.average(abs(error))
        print("repeatability:")
        print(rep)
        repeatabilities.append(rep)
    avg_rep = np.average(np.array(repeatabilities))
    print("avg repeatability:" + str(avg_rep))

    exit()
