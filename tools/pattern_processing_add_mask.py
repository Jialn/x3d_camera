# Copyright (c) 2020 XFORWARDAI. All Rights Reserved.
# used to process the pattern folder, add a mask to pattern
# usage: python temp/pattern_processing.py './temp/gray_code_pattern/' './temp/mask.bmp' 
# python temp/pattern_processing.py '/home/ubuntu/workplace/3dperceptionprototype/temp/gray_code_pattern/' '/home/ubuntu/workplace/3dperceptionprototype/temp/maskscene1/aluboard1.bmp'

import os
import numpy as np
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('srcpath', help='String Filepath')
    parser.add_argument('maskpath', help='String Filepath')
    # parser.add_argument('respath', help='String Filepath')
    args = parser.parse_args()
    
    srcpath = args.srcpath
    mask_path = args.maskpath
    respath = mask_path[:-4] # args.respath
    respath = respath + '/' # args.respath
    dir_list = os.listdir(srcpath)
    print(dir_list)
    img_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    cmd = "mkdir " + respath
    print("running cmd:" + cmd)
    os.system(cmd)

    for cur_file in dir_list:
        curr_path = srcpath + cur_file
        if os.path.isfile(curr_path):
            img = cv2.imread(curr_path, cv2.IMREAD_UNCHANGED)
            unvalid_points = np.where(img_mask == 0)
            img[unvalid_points] = 0
            img = cv2.imwrite(respath + cur_file, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        if os.path.isdir(curr_path) and cur_file!="cali_imgs":
            pass
            # cmd = "cp " + dataset_path + "calib.yml" + " " + curr_path + "/raw/"
            # print("running cmd:" + cmd)
            # os.system(cmd)
            
    exit()


