# Copyright (c) 2020. All Rights Reserved.
# Created on 2021-09-05
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
A calibrating helper for stereo structure-light x3d camera

To run on real camera, set "capture = True" then:
python -m stereo_calib "images/stereocali/"
This will open camera, grabing the image interactively and do the calibration.

To run on saved data, set "capture = False" then:
python -m stereo_calib "images/stereocali/", change the path accrodingly

This will call the capture images first to the specified path, and the do the calibrating and save calib.yml to the same path.
copy it to where you needed.

If you don't need to capture image again, like the case trying calib for the second time. comment the line "capture=True"
If you don't need to save result to where the depth cam uses, cali_cfg_path = "images/raw"
"""
import os
import numpy as np
import cv2
import glob
import argparse
from struli_camera_main import X3DCamera
from config import Config
if Config.use_high_speed_projector:
    from projector_lcp4500 import PyLCR4500
else:
    from projector_pdc03 import PyPDC


capture=False # if capture is not needed
remove_old_files = True # remove_old_files when capture
# write to cali_cfg_path or not
cali_cfg_path = None
# cali_cfg_path = "confs/"  # comment this line if you dont want to update the camera cfg file
corners_vertical = 8
corners_horizontal = 11
corner_distance = 0.014302  # in meters
light_up_scale = 1.2  # e.g, 1.2, to make the images brighter, incase can not find threshold


pattern_size = (corners_horizontal, corners_vertical)
pdc_03_port = "/dev/ttyUSB0"

def blur_image(image):
  res = cv2.blur(image, (3, 3))
  return res

class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 20, 0.002)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        w = pattern_size[1]
        h = pattern_size[0]
        self.objp = np.zeros((corners_vertical*corners_horizontal, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2) * corner_distance

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        if capture: self.capture_images(self.cal_path)
        self.read_images(self.cal_path)

    def capture_images(self, img_path):
        save_path = img_path
        # remove old images
        if remove_old_files:
            cmd = "rm " + save_path + "*.png"
            print("running cmd:" + cmd)
            os.system(cmd)
        i = 0

        if Config.use_high_speed_projector:
            projector = PyLCR4500()
            # projector.scan_one_pattern(index=8)
        else:
            projector = PyPDC(port=pdc_03_port, logging=False)
            projector.shake_hands()
            projector.scan_one_pattern(interval_time=0, index=18, display_time=2500)

        depth_camera = X3DCamera(hw_trigger=False, scale=None, logging=True)
        cv2.namedWindow('left', 0)
        cv2.namedWindow('right', 0)

        print("check left and right, make sure they are not inversed \
            after camera is proper focused, press 'e' to enable full light pattern and improve exposure, then press 's' to save images")
        while True:
            left = depth_camera.get_one_frame(camera_id=depth_camera.camera_ids[0])
            h, w = left.shape[:2]
            cv2.resizeWindow('left', w // 2, h // 2)
            cv2.resizeWindow('right', w // 2, h // 2)
            cv2.imshow("left", left)
            right = depth_camera.get_one_frame(camera_id=depth_camera.camera_ids[1])
            cv2.imshow("right", right)
            key = cv2.waitKey(50)
            if key & 0xFF == ord('e'):
                if Config.use_high_speed_projector: projector.ProjectSolidWhitePattern()
                else: projector.scan_one_pattern(interval_time=0, index=0, display_time=2500)
                depth_camera._camera.update_exp_time(depth_camera._camera.exp_time*2.0)
            if key & 0xFF == ord('s'):
                cv2.imwrite(save_path + "left_" + str(i) + ".png", left)
                cv2.imwrite(save_path + "right_" + str(i) + ".png", right)
                print("saved in:" + save_path + str(i) + ".png")
                i += 1
                print("framecnt" + str(i))
            if key == 27: break

        depth_camera.close()
        cv2.destroyAllWindows()

    def read_images(self, cal_path):
        images_left = glob.glob(cal_path + '/left_*.png')
        images_right = glob.glob(cal_path + '/right_*.png')
        # images_left = glob.glob(cal_path + '/left/*.png')
        # images_right = glob.glob(cal_path + '/right/*.png')
        images_left.sort()
        images_right.sort()
        print(images_left)
        print(images_right)

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i], cv2.IMREAD_UNCHANGED)
            img_r = cv2.imread(images_right[i], cv2.IMREAD_UNCHANGED)
            # img_l = blur_image(img_l)
            # img_r = blur_image(img_r)
            gray_l = light_up_scale*cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = light_up_scale*cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            gray_l = gray_l.astype(np.uint8)
            gray_r = gray_r.astype(np.uint8)

            # Find the chess board corners
            print("find cor...")
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret_l and ret_r:
                print("find subpix...")
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                if corners_l[0][0][0]+corners_l[0][0][1] > corners_l[-1][0][0]+corners_l[-1][0][1]:
                    corners_l = corners_l[::-1]
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, pattern_size,
                                                  corners_l, ret_l)
                cv2.imshow("curr_left", img_l)
                cv2.waitKey(100)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                if corners_r[0][0][0]+corners_r[0][0][1] > corners_r[-1][0][0]+corners_r[-1][0][1]:
                    corners_r = corners_r[::-1]
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, pattern_size,
                                                  corners_r, ret_r)
                cv2.imshow("curr_right", img_r)
                cv2.waitKey(100)
                cv2.imwrite(cal_path+ str(i) + "_right"  + ".jpg", img_r)
                cv2.imwrite(cal_path+ str(i) + "_left" + ".jpg", img_l)

            img_shape = gray_l.shape[::-1]

        flags = 0 # cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None, flags=flags)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None, flags=flags)

        tot_error = 0
        for i in range(len(self.objpoints)):
            imgpoints_l_rp, _ = cv2.projectPoints(self.objpoints[i], self.r1[i], self.t1[i], self.M1, self.d1)
            error = cv2.norm(self.imgpoints_l[i],imgpoints_l_rp, cv2.NORM_L2)/len(imgpoints_l_rp)
            tot_error += error
        print ("cam intrsic and distort:" + \
            "\n left K: " + str(self.M1) + \
            "\n left avg reprj error: " + str(tot_error/len(self.objpoints)))

        tot_error = 0
        for i in range(len(self.objpoints)):
            imgpoints_r_rp, _ = cv2.projectPoints(self.objpoints[i], self.r2[i], self.t2[i], self.M2, self.d2)
            error = cv2.norm(self.imgpoints_r[i],imgpoints_r_rp, cv2.NORM_L2)/len(imgpoints_r_rp)
            tot_error += error
        print (" right K: " + str(self.M2) + \
            "\n right avg reprj error: " + str(tot_error/len(self.objpoints)))

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # # flags |= cv2.CALIB_RATIONAL_MODEL
        # # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # # flags |= cv2.CALIB_FIX_K3
        # # flags |= cv2.CALIB_FIX_K4
        # # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        print("rms:", ret)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()

        outfile = cv2.FileStorage(self.cal_path + 'calib.yml', cv2.FileStorage_WRITE)
        outfile.write('K1', M1)
        outfile.write('K2', M2)
        outfile.write('D1', d1)
        outfile.write('D2', d2)
        outfile.write('R', R)
        outfile.write('T', T)
        outfile.write('E', E)
        outfile.write('F', F)
        outfile.release()
        print("result saved to:" + self.cal_path + 'calib.yml')

        return camera_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    path = args.filepath
    if path[-1] != '/': path = path + '/'
    cal_data = StereoCalibration(path)
    print(cal_data)

    from stereo_rectify import StereoRectify
    rectifier = StereoRectify(scale=None, cali_file=path + 'calib.yml')
    img = cv2.imread(path + 'left_0.png', cv2.IMREAD_UNCHANGED)
    img_r = cv2.imread(path + 'right_0.png', cv2.IMREAD_UNCHANGED)
    img_rectified = rectifier.rectify_image(img, left=True)
    img_rectified_r = rectifier.rectify_image(img_r, left=False)

    cv2.imwrite(path + "left_0_rectfy.jpg", img_rectified)
    cv2.imwrite(path + "right_0_rectfy.jpg", img_rectified_r)

    cimg = np.hstack((img_rectified, img_rectified_r))
    cimg = cv2.resize(cimg,(0,0),fx=0.25,fy=0.25)
    imh, imw, _ = cimg.shape
    for i in range(0,imh,20):
        pt1 = (0,i)
        pt2 = (imw,i)
        cv2.line(cimg,pt1, pt2, (0,0,255),1)
    cv2.imshow('show',cimg)
    cv2.waitKey()

    # copy cali files
    if cali_cfg_path is not None:
        # cmd = "cp " + path + "calib.yml" + " " + cali_cfg_path
        # print("running cmd:" + cmd)
        # os.system(cmd)
        import shutil
        shutil.copy(path + "calib.yml", cali_cfg_path+"calib.yml")

    exit()
