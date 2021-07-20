# Copyright (c) 2020 XFORWARDAI. All Rights Reserved.
#
# Register all the ply file in the path
# Install open-3d: "pip3 install open3d", do not use "open3d-python"
# Usage example: python tools/icp_registration.py temp/icptest/runing_path/

import open3d as o3d
import numpy as np
import time
import os
import argparse
import copy

# down-sample voxel_size
voxel_size_down_sample = 0.001 # 0.002
voxel_size = 0.002  # voxel_size for alg paramters
normal_radius = voxel_size * 10 # voxel_size * 2
fast_global_registration_feature_radius = voxel_size * 12.5 # voxel_size * 5
fast_global_registration_distance_threshold = voxel_size * 2500 # voxel_size * 2 # voxel_size * 0.5
icp_max_correspondence_distance_coarse = voxel_size * 25
icp_max_correspondence_distance_fine = voxel_size * 2.5

def execute_fast_global_registration(source, target):
    radius_feature = fast_global_registration_feature_radius
    print("Get FPFH feature using search radius {}".format(radius_feature))
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    distance_threshold = fast_global_registration_distance_threshold
    print("fast_global_registration, distance_threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source, target,
        source_fpfh, target_fpfh, o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    return result


def pairwise_registration(source, target, init_trans):
    print("apply point - point or plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(source, target, icp_max_correspondence_distance_coarse,
                                                             init_trans,
                                                             o3d.pipelines.registration.TransformationEstimationPointToPlane())  # TransformationEstimationPointToPlane
    print(icp_coarse.transformation)
    icp_fine = o3d.pipelines.registration.registration_icp(source, target, icp_max_correspondence_distance_fine,
                                                           icp_coarse.transformation,
                                                           o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(icp_fine.transformation)
    # transformation_icp = icp_fine.transformation
    # information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
    #     source, target, icp_max_correspondence_distance_fine, icp_fine.transformation)
    # return transformation_icp, information_icp
    return icp_fine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    dataset_path = args.filepath 
    dir_list = os.listdir(dataset_path)
    dir_list.sort()
    if dataset_path[-1] != '/': dataset_path = dataset_path + "/"

    print("read ply file")
    clouds = []
    for cur_file in dir_list:
        curr_path = dataset_path + cur_file
        if os.path.isdir(curr_path): curr_path = curr_path + "/points.ply"   # for dataset format
        if os.path.isfile(curr_path) and curr_path[-4:]=='.ply':
            print("loading ply file:" + curr_path)
            ply_cld = o3d.io.read_point_cloud(curr_path)
            ply_cld.translate(np.zeros(3), relative=False)
            # print(ply_cld.get_center())
            ply_cld.scale(0.001, center=np.zeros(3))
            clouds.append(ply_cld)

    print("outlier removal, down sample and estimate normals")
    processed_clouds = []
    for i in range(len(clouds)):
        processed_cloud = copy.deepcopy(clouds[i])
        processed_clouds.append(processed_cloud)
        # if i >= 4: processed_clouds[i].rotate(processed_clouds[i].get_rotation_matrix_from_xyz((np.pi,0,0)), center=(0,0,0))
        # cloud.paint_uniform_color([1, 0.706, 0])    # yellow
        # cloud, outlier_index = o3d.geometry.radius_outlier_removal(cloud, nb_points=16, radius=0.5)
        processed_clouds[i] = processed_clouds[i].voxel_down_sample(voxel_size=voxel_size_down_sample)
        processed_clouds[i].estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

    print("run registration_icp")
    transforms = []
    for i in range(len(processed_clouds)-1):
        start = time.time()
        reg_p2p = execute_fast_global_registration(processed_clouds[i+1], processed_clouds[i])
        # reg_p2p_refine = o3d.pipelines.registration.registration_icp(processed_source, processed_target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        reg_p2p_refine = pairwise_registration(processed_clouds[i+1], processed_clouds[i], init_trans=reg_p2p.transformation)
        print("registration using %.3f s.\n" % (time.time() - start))
        for i in range(i+1,len(processed_clouds)):
            processed_clouds[i].transform(reg_p2p_refine.transformation)
            clouds[i].transform(reg_p2p_refine.transformation)

    print("rendering")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for cloud in processed_clouds: 
        vis.add_geometry(cloud)
    vis.update_renderer()
    vis.run()

    print("writing res")
    res_points = copy.deepcopy(clouds[0])
    for i in range(1, len(clouds)):
        res_points = res_points + clouds[i]
    save_path = dataset_path[:-1]
    o3d.io.write_point_cloud(save_path + "_registered_points.ply", res_points, write_ascii=False, compressed=False)
