# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-11-13
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: xForward 3D Camera

This class implement the xForward structure light 3D camera.

Test with real camera: "python py"
Record dataset command example: "python py dataset ./dataset_3dpattern 0"

The typical procedure of a 3D capture is:
1. trigger the projector, scan all patterns saved in projector;
2. the hardware-line synced camera will capture all the images and saved to a specified path;
3. run the "_caculate_depth" function, generate 2D image, depth iamge, and camera intrsic matrix;
4. generate point cloud.

About interface saving path if saving flag is set:
    files in image_path: 0_l.bmp, 0_r.bmp, 1_l.bmp, 1_r.bmp, .... 23_l.bmp, 23_r.bmp, calib.yml
    files in res_path: depth.png(uint16, *2000/60000 -> mm) depth_raw.png gray.png gray_unlight.png pc_cl_bi.ply camera_kd.yml
"""
import os
import sys
import numpy as np
import cv2
import time
import open3d as o3d
from config import Config
from HuaraySDKWrapper import HuarayCamera
from stereo_rectify import StereoRectify
if Config.use_phsft:
    lib_path = os.path.abspath(os.path.join('../struc_light'))
    print(lib_path)
    sys.path.append(lib_path)
    import structured_light_cuda as struli
    struli.phase_decoding_unvalid_thres = Config.phsft_thres
else:
    from x3d_camera import structured_light as struli

def adjust_gamma(imgs, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    new_imgs = cv2.LUT(np.array(imgs, dtype=np.uint8), table)
    return new_imgs

def convert_depth_to_color(depth_map_mm, scale=None):
    h, w = depth_map_mm.shape[:2]
    depth_image_color_vis = depth_map_mm.copy()
    valid_points = np.where(depth_image_color_vis>=0.1)
    # depth_near_cutoff, depth_far_cutoff = np.min(depth_image_color_vis[valid_points]), np.max(depth_image_color_vis[valid_points])
    depth_near_cutoff, depth_far_cutoff = np.percentile(depth_image_color_vis[valid_points], 3), np.percentile(depth_image_color_vis[valid_points], 97)
    depth_far_cutoff = depth_near_cutoff + (depth_far_cutoff-depth_near_cutoff) * 1.2
    depth_range = depth_far_cutoff-depth_near_cutoff
    # print((depth_near_cutoff, depth_far_cutoff))
    depth_image_color_vis[valid_points] = depth_far_cutoff - depth_image_color_vis[valid_points]  # - depth_near_cutoff
    depth_image_color_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_color_vis, alpha=255/(depth_range)), cv2.COLORMAP_JET)  #COLORMAP_JET HOT
    if scale is not None:
        depth_image_color_vis = cv2.resize(depth_image_color_vis, ((int)(w*scale), (int)(h*scale)))
    return depth_image_color_vis

def gen_point_clouds_from_images(depth, camera_kp, image, save_path=None):
    """Generate PointCloud from images and camera kp
    """
    convert_rgb_to_intensity = True
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        convert_rgb_to_intensity = False
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(image),
        o3d.geometry.Image(depth.astype(np.float32)),
        convert_rgb_to_intensity=convert_rgb_to_intensity,
        depth_scale=1.0,
        depth_trunc=6000.0)
    h, w = image.shape[:2]
    fx, fy, cx, cy = camera_kp[0][0], camera_kp[1][1], camera_kp[0][2], camera_kp[1][2]
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy))
    if save_path is not None:
        import copy
        if save_path[-4:] != '.ply': save_path = save_path + "/points.ply"
        pcd_to_write = copy.deepcopy(pcd)
        pcd_to_write.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=10))
        pcd_to_write.orient_normals_towards_camera_location()
        o3d.io.write_point_cloud(save_path, pcd_to_write, write_ascii=False, compressed=False)
        print("res saved to:" + save_path)
    return pcd


class X3DCamera():
    """ Class for xForward 3D camera
    """

    def __init__(self,
                 hw_trigger=True,
                 scale=0.5,
                 logging=False):
        """
        Init

        Args:
            logging (bool): log or not.
            scale(float): the scale for image, used for downsampling.
        """
        self._logging = logging
        self._correct_blue_channel = True
        self._camera = HuarayCamera(hw_trigger=hw_trigger, scale=scale)
        self.camera_ids = self._camera.camera_ids
        self._camera_kd = None
        self._depth_img = None
        self._depth_point_cloud = None
        self._rectified_rgb_img = None
        if scale is None: scale = 1.0
        self.scale = scale
        dir_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file
        self._res_path = dir_path + "/images"
        self._pattern_path = self._res_path + "/raw/"
        self._rectifier = StereoRectify(scale=scale, cali_file=self._pattern_path+'calib.yml')
        # run capture for once to init internal flags
        # self._camera.get_one_frame(self.camera_ids[0], force_gray=False, use_env_exp=True)  # left
        # self.close_projector()
        self.re_capture_image_using_env_exposure = Config.re_capture_color_image_using_env_exposure

    def get_one_frame(self, camera_id):
        """ Grab one frame by sw trigger.
        Note: using get_one_frame and get_projected_raw_images mixed could make some hw trigger fails.
            Have not found the solution for this problem.

        Args:
            camera_id: The id of camera. For Dual cam, it could be 0, 1 or 2.
                Note that 0 could be the usb webcam if a laptop is used. 
        """
        return self._camera.get_one_frame(camera_id)

    def get_images(self, undistort=True):
        """Get the images: RGB, gray/IR and depth image
        The RGB here is a simulated color, gray + blue channle image for now.
        """
        if self._camera.is_color_camera() and self.re_capture_image_using_env_exposure:
            if not Config.use_high_speed_projector:
                self.close_projector()
                time.sleep(0.033)
            bgr = self._camera.get_one_frame(self.camera_ids[0], force_gray=False, use_env_exp=True)  # left
        start_time = time.time()
        self._camera.get_projected_raw_images(path=self._pattern_path)
        gray, depth_mili_meter, camera_kd = self._caculate_depth()
        print("Total time including capture: %.3f s" % (time.time() - start_time))
        self._camera_kd = camera_kd
        self._depth_img = depth_mili_meter
        # simulate color for mono camor capture color image
        if self._camera.is_color_camera():
            if not self.re_capture_image_using_env_exposure:
                if Config.do_demosac_for_color_camera:
                    bgr = self._camera.captured_images_left_color[Config.pattern_start_index]
                else:
                    bgr = self._camera.captured_images_left[Config.pattern_start_index]
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BAYER_BG2BGR)
            bgr = self._rectifier.rectify_image(bgr)
            if Config.save_pattern_to_disk: cv2.imwrite(self._res_path + "/color.bmp", bgr)
            self._rgb_image, self._depth_image = bgr, depth_mili_meter
        else:
            if Config.fake_color_using_blue_light:
                # dual channle fake rgb/bgr
                target_brightness_level = 90.0
                target_brightness_level_blue = 72.0
                mean = np.mean(gray)
                # auto_gamma_val = math.log10(0.5)/math.log10(mean/255)
                if self._logging: print("gray image mean:" + str(mean))
                gray = gray * target_brightness_level / mean
                gray_blue_light = self._camera.captured_images_left[Config.pattern_start_index]
                blue_part = adjust_gamma(gray_blue_light, gamma=0.7)
                mean = np.mean(blue_part)
                if self._correct_blue_channel:
                    blue_part = blue_part.astype(np.uint16)
                    max_h = blue_part.shape[0]
                    for h in range(max_h):
                        adj = (max_h/2 - h) / max_h
                        blue_part[h] = blue_part[h] * (1+ 0.75 * adj)
                    blue_part = np.clip(blue_part, 0, 255).astype(np.uint8)
                if self._logging: print("blue part mean:" + str(mean))
                blue_part = blue_part * target_brightness_level_blue / mean
                bgr = np.tile(gray[:, :, None], (1, 1, 3))  # actually is "bgr"
                bgr[:, :, 0] = blue_part
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)
            else:
                bgr = None
            self._rgb_image, self._depth_image = gray, depth_mili_meter
        return bgr, gray, depth_mili_meter

    def get_point_cloud(self, save=False, recaptue_rgb=False):
        """ Get point cloud

        This should be called after get_images.
        """
        rgb, depth = self._rgb_image.copy(), self._depth_image.copy()
        if recaptue_rgb:
            if not Config.use_high_speed_projector:
                self.close_projector()
                time.sleep(0.033)
            rgb = self._camera.get_one_frame(self.camera_ids[0], force_gray=False, use_env_exp=True)
            rgb = self._rectifier.rectify_image(rgb)
        save_path = self._res_path if save else None
        color_cloud = gen_point_clouds_from_images(depth, self._camera_kd, rgb, save_path=save_path)
        return color_cloud

    def get_camera_kp(self):
        """ Get camera intrinsic parameters
        """
        return self._camera_kd

    def _caculate_depth(self):
        """ Caculate depth from projected images
        """
        images = [self._camera.captured_images_left, self._camera.captured_images_right]
        is_bayer_color_image = self._camera.is_color_camera() and (not Config.do_demosac_for_color_camera)
        use_blue_chan_only = is_bayer_color_image and Config.use_blue_channel_for_color_camera
        gray, depth, camera_kd = struli.run_stru_li_pipe(
            pattern_path=self._pattern_path, res_path=self._res_path, rectifier=self._rectifier, images=images,
            is_bayer_color_image=is_bayer_color_image, use_blue_chan_only=use_blue_chan_only)
        return gray, depth, camera_kd

    def close(self):
        self._camera.close()

    def close_projector(self):
        self._camera._projector.turn_led(0)


# test with real camera: "python py"
# record dataset command example: "python py dataset ./dataset_3dpattern 0"
if __name__ == "__main__":

    if len(sys.argv) <= 1:  # run with no args, use the real camera
        depth_camera = X3DCamera(logging=True, scale=Config.default_testing_scale)
        while True:
            _, gray_img, depth_img_mm = depth_camera.get_images()  # rgb_img is None
            cv2.imshow("depth", convert_depth_to_color(depth_img_mm, scale=0.5))
            ## cv2.imshow("depth", (depth_img_mm*1000).astype(np.uint8)) # to um
            # depth_camera._camera.captured_images_left[36] = depth_camera._camera.captured_images_left[36] / 4
            cv2.imshow("pattern_example", depth_camera._camera.captured_images_left[36].astype(np.uint8))
            # cv2.imshow("pattern_example_r", depth_camera._camera.captured_images_right[36])
            print("Press space on the image window to continue, or press ESC to end the script")
            while True:
                key = cv2.waitKey(10)
                if key & 0xFF == ord(' '): break
                if key == 27: break  # ESC
            if key == 27: break  # ESC pressed
        if Config.use_depth_as_cloud_color:
            depth_vis = convert_depth_to_color(depth_img_mm)
            points = gen_point_clouds_from_images(depth_img_mm, depth_camera._camera_kd, depth_vis, save_path="./")
        else:
            points = depth_camera.get_point_cloud(save=Config.save_point_cloud, recaptue_rgb=True)
        import open3d as o3d
        points.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw(geometry=points, width=1600, height=900, point_size=1,
            bg_color=(0.5, 0.5, 0.5, 0.5), show_ui=True)
        print("res saved in:" + depth_camera._res_path)
        depth_camera.close()
    else:  # run with args
        if sys.argv[1] not in ['dataset']:
            print("run with args 'None', or 'dataset'")
            exit()
        
        if sys.argv[1] == 'dataset':  # record dataset
            scale = 1.0 # 0.5
            cali_file_path = "./x3d_camera/images/raw/calib.yml"
            saving_path = sys.argv[2] + '/'
            start_id = (int)(sys.argv[3])
            depth_camera = X3DCamera(logging=True, scale=scale)

            while True:
                current_id_str = "0"* (4-len(str(start_id))) + str(start_id)
                current_path = saving_path + current_id_str
                print("current saving path: " + current_path)
                if not os.path.exists(current_path):
                    os.makedirs(current_path)
                    os.makedirs(current_path + "/raw")
                cmd = "cp " + cali_file_path + " " + current_path + "/raw/"
                print("running cmd:" + cmd)
                os.system(cmd)
                # update build-in path
                depth_camera._res_path = current_path
                depth_camera._pattern_path = current_path + "/raw/"
                depth_camera._rectifier = StereoRectify(scale=scale, cali_file=depth_camera._pattern_path+'calib.yml')
                # get images
                rgb_img, gray_img, depth_img = depth_camera.get_images()
                points = depth_camera.get_point_cloud(save=True)
                # save images and post process
                cv2.imwrite(current_path+'/rgb_image.png', rgb_img)
                h, w = rgb_img.shape[:2]
                cv2.imshow("img", rgb_img)
                # write a preview in saving folder
                cv2.imwrite(saving_path + current_id_str + '.png', rgb_img)
                print("Press space on the image window to continue capture frame:" + str(start_id+1) + ", or press ESC to end the script")
                while True:
                    key = cv2.waitKey(10)
                    if key & 0xFF == ord(' '): break
                    if key == 27: break  # ESC
                if key == 27: break  # ESC pressed
                start_id += 1

            depth_camera.close()
            cv2.destroyAllWindows()
            exit()

