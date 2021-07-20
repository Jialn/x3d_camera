# Copyright (c) 2020 XFORWARDAI. All Rights Reserved.
# Created on 2021-04-01
# Autor: Jiangtao <jiangtao.li@xforwardai.com>
"""
Description: xForward 3D Camera structure light algorithm pipeline

These programs implement the xForward structured light 3D camera pipeline.
"""

import os
import cv2
import time
import numpy as np
import numba
from numba import prange
import transform_helper
from stereo_rectify import StereoRectify
import structured_light_core as sl_core  # import lib.structured_light_core as sl_core

### parameters
image_index_unvalid_thres = 1
image_index_using_positive_pattern_only = False
use_index_sub_pix_optimize=True
use_depth_avg_filter = True
use_depth_bi_filter = False

def get_image_index(image_path, appendix, rectifier):
    images_posi = []
    images_nega = []
    unvalid_thres = image_index_unvalid_thres
    for i in range(0, 2):
        fname = image_path + str(i) + appendix
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # rectify_image, accroding to left or right
        if appendix == '_l.bmp': img = rectifier.rectify_image(img.astype(np.uint8))
        else: img = rectifier.rectify_image(img.astype(np.uint8), left=False)
        # posi or negative
        if i % 2 == 0: prj_area_posi = img
        else: prj_area_nega = img
    prj_valid_map = prj_area_posi - prj_area_nega
    if image_index_using_positive_pattern_only:
        positive_pattern_only_avg_thres = (prj_area_posi//2 + prj_area_nega//2)
    thres, prj_valid_map_bin = cv2.threshold(prj_valid_map, unvalid_thres, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # prj_valid_map_bin = cv2.morphologyEx(prj_valid_map_bin, cv2.MORPH_CLOSE, kernel, iterations=1)  # erode then dilate
    # prj_valid_map_bin = cv2.morphologyEx(prj_valid_map_bin, cv2.MORPH_OPEN, kernel, iterations=1)  # erode then dilate
    # cv2.imwrite('prj_map_alg2.png', prj_valid_map.astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # cv2.imwrite('prj_map_alg2_bin.png', prj_valid_map_bin.astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    start_time = time.time()
    for i in range(2, 24):
        fname = image_path + str(i) + appendix
        if not os.path.exists(fname): break
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # rectify_image, accroding to left or right
        if appendix == '_l.bmp': img = rectifier.rectify_image(img.astype(np.uint8))
        else: img = rectifier.rectify_image(img.astype(np.uint8), left=False)
        # posi or negative
        if i % 2 == 0: images_posi.append(img)
        else:
            if image_index_using_positive_pattern_only: images_nega.append(positive_pattern_only_avg_thres)
            else: images_nega.append(img)
    # optional save binariz images
    # if appendix == '_l.bmp':
    #     cv2.imwrite("./bavg.jpg", (prj_area_posi//2 + prj_area_nega//2))
    #     for i in range(len(images_posi)):
    #         diff = (127 + images_posi[i]//2) - images_nega[i]//2
    #         thres, diff_bin = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
    #         cv2.imwrite("./b"+ str(i) + ".jpg", diff_bin)
    print("read and rectify images using %.3f s" % (time.time() - start_time))
    height, width = images_posi[0].shape[:2]
    img_index, src_imgs = np.zeros_like(images_posi[0], dtype=np.int16), np.array(images_posi)
    src_imgs_nega = np.array(images_nega)
    start_time = time.time()
    sl_core.gray_decode(src_imgs, src_imgs_nega, prj_valid_map_bin, len(images_posi), height,width, img_index, unvalid_thres)
    print("index decoding using %.3f s" % (time.time() - start_time))
    return img_index, rectifier.rectified_camera_kd

def run_stru_li_pipe(pattern_path, res_path, rectifier=None):
    if rectifier is None:
        rectifier = StereoRectify(scale=1.0, cali_file=pattern_path+'calib.yml')

    ### Rectify and Decode to index
    pipe_start_time = start_time = time.time()
    img_index_left, camera_kd_l = get_image_index(pattern_path, '_l.bmp', rectifier=rectifier)
    img_index_right, camera_kd_r = get_image_index(pattern_path, '_r.bmp', rectifier=rectifier)
    print("index decoding in total using %.3f s" % (time.time() - start_time))
    fx = camera_kd_l[0][0]
    cx, cx_r = camera_kd_l[0][2], camera_kd_r[0][2]
    dmap_base = cx_r - cx
    cam_transform = np.array(rectifier.T)[:,0]
    height, width = img_index_left.shape[:2]
    # TODO
    # print(rectifier.R1)
    # rot_angels = transform_helper.mat2euler(rectifier.R1)
    # print(rot_angels)
    # print(np.cos(rot_angels[0]))
    # print(np.cos(rot_angels[1]))
    # print(np.cos(rot_angels[2]))
    # print(np.cos(abs(rot_angels[0])+abs(rot_angels[1])+abs(rot_angels[2])))
    # print(( 0.8/(0.8+0.05*0.001) ))
    # baseline = np.linalg.norm(cam_transform) * np.cos(abs(rot_angels[0])+abs(rot_angels[1])+abs(rot_angels[2]))
    baseline = np.linalg.norm(cam_transform) * ( 0.8/(0.8+0.05*0.001) )  # = 0.9999375039060059

    ### Infer DiffMap from DecodeIndex
    dmap = np.zeros_like(img_index_left, dtype=np.float)
    depth_map = np.zeros_like(img_index_left, dtype=np.float)  # img_index_left.copy().astype(np.float)
    # subpixel for index
    expected_pixel_per_index = (width*0.75) / 1280.0
    img_index_left_sub_px = np.zeros_like(img_index_left, dtype=np.float)
    img_index_right_sub_px = np.zeros_like(img_index_left, dtype=np.float)
    if use_index_sub_pix_optimize and expected_pixel_per_index <= 1.1:
        start_time = time.time()
        sl_core.index_sub_pix(img_index_left_sub_px, img_index_left, height, width, expected_pixel_per_index)
        sl_core.index_sub_pix(img_index_right_sub_px, img_index_right, height, width, expected_pixel_per_index)
        print("index_sub_pix_optimize using %.3f s" % (time.time() - start_time))
    # gen depth and diff map
    start_time = time.time()
    sl_core.get_dmap_from_index_map(dmap, depth_map, height, width, img_index_left, img_index_right, baseline, dmap_base, fx, img_index_left_sub_px, img_index_right_sub_px)
    print("depth map generating from index %.3f s" % (time.time() - start_time))

    ### Run Depth Map Filter
    depth_map_raw = depth_map.copy()  # save raw depth map
    start_time = time.time()
    sl_core.depth_filter(depth_map, depth_map_raw, height, width, camera_kd_l)
    print("flying point filter %.3f s" % (time.time() - start_time))
    if use_depth_avg_filter:
        start_time = time.time()
        sl_core.depth_avg_filter(depth_map, depth_map_raw, height, width, camera_kd_l)
        print("depth avg filter %.3f s" % (time.time() - start_time))
    if use_depth_bi_filter:
        start_time = time.time()
        depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), d=width//10, sigmaColor=0.0025, sigmaSpace=0.0025, borderType=cv2.BORDER_DEFAULT)
        print("bilateralFilter %.3f s" % (time.time() - start_time))
    print("Total pipeline time: %.3f s" % (time.time() - pipe_start_time))
    ### Save Mid Results
    # img_correspondence_l = np.clip(img_index_left/width*255.0, 0, 255).astype(np.uint8)
    # img_correspondence_r = np.clip(img_index_right/width*255.0, 0, 255).astype(np.uint8)
    # cv2.imwrite(res_path + "/correspondence_l.png", img_correspondence_l)
    # cv2.imwrite(res_path + "/correspondence_r.png", img_correspondence_r)
    cv2.imwrite(res_path + '/diff_map_alg2.png', dmap.astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    depth_map_uint16 = depth_map * 30000
    cv2.imwrite(res_path + '/depth_alg2.png', depth_map_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    depth_map_raw_uint16 = depth_map_raw * 30000
    cv2.imwrite(res_path + '/depth_alg2_raw.png', depth_map_raw_uint16.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    
    ### Prepare results
    depth_map_mm = depth_map * 1000
    gray_img = cv2.imread(pattern_path + "0_l.bmp", cv2.IMREAD_UNCHANGED)
    gray_img = rectifier.rectify_image(gray_img)
    return gray_img, depth_map_mm, camera_kd_l

# test with existing pattern: 
#   python -m structured_light '/home/ubuntu/workplace/3dperceptionprototype/temp/dataset_render/0004/raw'
if __name__ == "__main__":
    import sys
    import glob
    import shutil
    import open3d as o3d
    import matplotlib.pyplot as plt

    is_render = False
    scale_image = None  # None or 1.0 by default. if calibra file is generated with high res and pattern is downscaled, using this option to upscale back
    
    if len(sys.argv) <= 1:
        print("run with args 'pattern_path'")
    image_path = sys.argv[1]
    if image_path[-1] == "/": image_path = image_path[:-1]

    ### convert to gray and scale the image if needed
    if not os.path.exists(image_path + "_gray"):
        shutil.copytree(image_path + "/",image_path + "_gray/")
        os.system("mkdir " + image_path + "_gray")
        os.system("cp -r " + image_path + "/* " + image_path + "_gray/")
    image_path = image_path + '_gray/'
    images = glob.glob(image_path + '*.bmp')
    for i, fname in enumerate(images):
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(fname, img)
        if scale_image is not None and scale_image != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (round(w*scale_image), round(h*scale_image)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(fname, img)

    ### build up runing parameters and run the pipeline
    res_path = image_path + "../x3d_cam_res"
    if not os.path.exists(res_path): os.system("mkdir " + res_path)
    gray, depth_map_mm, camera_kp = run_stru_li_pipe(image_path, res_path)

    ### save to point cloud
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(gray.astype(np.uint8)),
        o3d.geometry.Image(depth_map_mm.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=6000.0)
    h, w = gray.shape[:2]
    fx, fy, cx, cy = camera_kp[0][0], camera_kp[1][1], camera_kp[0][2], camera_kp[1][2]
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy))
    # flip it if needed
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    o3d.io.write_point_cloud(res_path + "/points.ply", pcd, write_ascii=False, compressed=False)
    print("res saved to:" + res_path)

    ### visualize
    # plt.subplot(1, 2, 1)
    # plt.title('grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    # o3d.visualization.draw_geometries([pcd])