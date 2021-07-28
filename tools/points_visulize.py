# Copyright (c) 2020. All Rights Reserved.
# Visulize depth using color map by fited ref plane
# usage: 
#   1. spefiy path to variable "image_path"
#   2. the path should include "gray.bmp  depth.exr  camera_kd.txt"
#   3. python tools\points_visulize.py

import numpy as np
import cv2
import argparse
import open3d as o3d

image_path = "./images/temp/"

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

def convert_depth_to_color(depth_map_mm, scale=None, depth_plane=None, percentile=(3, 97)):
    h, w = depth_map_mm.shape[:2]
    depth_image_color_vis = depth_map_mm.copy()
    valid_points = np.where(depth_image_color_vis>=0.1)
    if depth_plane is not None: depth_image_color_vis[valid_points] = depth_image_color_vis[valid_points] - depth_plane[valid_points]
    # depth_near_cutoff, depth_far_cutoff = np.min(depth_image_color_vis[valid_points]), np.max(depth_image_color_vis[valid_points])
    depth_near_cutoff, depth_far_cutoff = np.percentile(depth_image_color_vis[valid_points], percentile[0]), np.percentile(depth_image_color_vis[valid_points], percentile[1])
    depth_far_cutoff = depth_near_cutoff + (depth_far_cutoff-depth_near_cutoff) * 1.0
    depth_range = depth_far_cutoff-depth_near_cutoff
    # print((depth_near_cutoff, depth_far_cutoff))
    depth_image_color_vis[valid_points] = depth_far_cutoff - depth_image_color_vis[valid_points]  # - depth_near_cutoff
    depth_image_color_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_color_vis, alpha=255/(depth_range)), cv2.COLORMAP_JET)  #COLORMAP_JET HOT
    if scale is not None:
        depth_image_color_vis = cv2.resize(depth_image_color_vis, ((int)(w*scale), (int)(h*scale)))
    return depth_image_color_vis

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filepath', help='String Filepath')
    # args = parser.parse_args()
    # dir = args.filepath
    dir = image_path

    gray_img = cv2.imread(dir+"gray.bmp", cv2.IMREAD_UNCHANGED)
    depth_img_mm = cv2.imread(dir+"depth.exr", cv2.IMREAD_UNCHANGED)
    camera_kd = np.loadtxt(dir+"camera_kd.txt")

    vis_scale = 2
    height, width = gray_img.shape[:2]
    image_vis_size = (width//vis_scale, height//vis_scale)
    gray_img_vis = cv2.resize(gray_img, image_vis_size)
    coord_list = []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            # 画圈（图像:img，坐标位置:xy，半径:1(就是一个点)，颜色:蓝，厚度：-1(就是实心)
            cv2.circle(gray_img_vis, (x, y), 6, (255, 0, 0), thickness=1)
            cv2.putText(gray_img_vis, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", gray_img_vis)
            coord_list.append([(int)(x*vis_scale),(int)(y*vis_scale)])
            #写入txt
            x_str = str(x)
            y_str = str(y)
            f = open(dir + "coordinate.txt", "a+")
            f.writelines(x_str + ' ' + y_str + '\n')

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", gray_img_vis)
    while (True):
        try:
            key = cv2.waitKey(100)
        except Exception:
            cv2.destroyAllWindows()
            break
        if key == 27: break  # ESC
    cv2.destroyAllWindows()

    # fit plane
    tmp_A, tmp_b = [], []
    for point in coord_list:
        if depth_img_mm[point[1], point[0]] > 0.1:
            tmp_A.append([point[0], point[1], 1])  # [x, y, 1].T
            tmp_b.append(depth_img_mm[point[1], point[0]])  # z
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    print("solution: z = %f x + %f y + %f" % (fit[0], fit[1], fit[2]))
    print("errors:" + str(errors))
    print("residual:" + str(residual))

    # gen color map results
    a, b, c = float(fit[0]), float(fit[1]), float(fit[2])
    depth_plane=np.fromfunction(lambda i,j: a*j+b*i+c, (height,width), dtype=float)
    img_res = convert_depth_to_color(depth_img_mm, scale=None, depth_plane=depth_plane, percentile=(10, 90))
    cv2.imwrite(dir+"depth_vis_res.jpg", img_res)
    # gen point clouds
    gray_vis = (0.5 + gray_img / 512.0)
    depth_vis = (img_res * np.tile(gray_vis[:, :, None], (1, 1, 3))).astype(np.uint8)
    points = gen_point_clouds_from_images(depth_img_mm, camera_kd, depth_vis, save_path=dir)
    # show results
    cv2.imshow("res", img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
