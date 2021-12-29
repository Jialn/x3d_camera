
"""
This file is not for general python env!
Should be runing inside blender.

For the first time, make sure camera name is correct.

For usage with dataset gen:
    set saving_path
    set data_set_start_id, data_set_length, then wait for done
"""
# To install packages for blender builtin python:
# win:
# (in blender's python path, like C:\tools\blender-2.91.2-windows64\2.91\python\bin)
# ./python -m pip install --upgrade pip
# ./python -m pip  config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# ./python -m pip install opencv-python # imageio pypng not needed
# linux:
# (in blender's python path, like: cd /home/ubuntu/workplace/tools/blender-2.91.2-linux64/2.91/python/bin/)
# ./python3.7m -m ensurepip
# ./pip3  config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# ./pip3 install --upgrade pip
# ./pip3 install opencv-python
import os
import bpy
import numpy as np
import math
import cv2
import json
import random
from bpy import context
from mathutils import Matrix, Vector
import numpy as np
import math


#######################
### Confs
gen_data_set = True
main_scene_name = "_mainScene"
left_camera_name_in_scene = 'Camera'
right_camera_name_in_scene = 'Camera.right'

### path
saving_path = "/home/jiangtao.li/Pictures/"
cali_img_path = saving_path + 'cali_imgs/'

### for gen dataset
data_set_start_id = 0
data_set_length = 1
frames_range = range(0, 1)  # (0, 24) for dataset gen, smaller range for script testing
gen_camera_para = True

### set left camera parameters, paras about right cameras are auto caculated
left_camera_pos = [1.5, -4.4, 1.09]  # [x, y, z] of left camera, in meters
left_camera_rot = [90, 0, 0] # rotate degree around [x, y, z]-axis
fov_angle = (3.1415926/180.0) * 85
baseline = 0.25
resolution_x = 1280
resolution_y = 800
cycles_samples = 256
cycles_max_bounces = 8 # 0 or 8
use_denoising = True


"""
Description: 
Helper for 3D transformation
About "axes_para" parameter in this file:
    axes_para is an optional axis-specification. The format is (firstaxis, parity, repetition, frame).
    A fast look up table for axes_para:
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0), 'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0), 'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1), 'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1), 'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)
    First char 's/r' stands for static(extrinsic) or rotating(intrinsic) coordinate axes
        extrinsic: rotation is about the static fixed axis
        intrinsic: rotation is about the rotated new axis
    The following 3 characters 'x/y/z' give the order in which the rotations will be performed.
    The defalut is sxyz, i.e., static RollPitchYaw. 
"""
def to_rad(deg):
    """Return radian form degree.
    """
    return np.array(deg) * np.pi / 180.0


def to_deg(rad):
    """Return degree from radian.
    """
    return np.array(rad) * 180.0 / np.pi


def xyzrpy2mat(pose_xyzrpy):
    """Return homogenous rotation matrix from the robot arm [x,y,z,r,p,y] representation.
    """
    pos = pose_xyzrpy[:3]
    rpy = to_rad(pose_xyzrpy[3:6])
    R = euler2mat(rpy[0], rpy[1], rpy[2])
    R_with_pos = np.insert(R, 3, values=pos, axis=1)
    # rotation matrix with homogenous coordinates
    trans_matrix = np.insert(R_with_pos, 3, values=np.array([0, 0, 0, 1]), axis=0)
    # print(R)
    # print(trans_matrix)
    return trans_matrix


def euler2mat(ai, aj, ak, axes_para=(0, 0, 0, 0)):
    """
    Return rotation matrix from Euler angles and axis sequence.
    Args:
        ai : First rotation angle (according to `axes_para`).
        aj : Second rotation angle (according to `axes_para`).
        ak : Third rotation angle (according to `axes_para`).
        axes_para : optional Axis specification. The format is (firstaxis, parity, repetition, frame).
            The defalut is sxyz, i.e., static RollPitchYaw, for details refer to description of this file
    Return:
        numpy.array in shape (3,3), the rotation matrix, 3D non-homogenous coordinates
    """
    _NEXT_AXIS = [1, 2, 0, 1]
    i, parity, repetition, frame = axes_para
    j, k = _NEXT_AXIS[i + parity], _NEXT_AXIS[i - parity + 1]
    if frame: ai, ak = ak, ai
    if parity: ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def mat2euler(mat, axes_para=(0, 0, 0, 0)):
    """
    Give Euler angles from rotation matrix. Note that many Euler angle triplets can describe one matrix.
    Args:
        mat : array-like shape (3, 3). homogenous (4, 4) will be trunked to (3, 3)
        axes_para : optional Axis specification. The format is (firstaxis, parity, repetition, frame).
    Return:
        a list, rotation angle for specified axis sequence.
    """
    _EPS4 = np.finfo(float).eps * 4.0  # For testing whether a number is close to zero
    _NEXT_AXIS = [1, 2, 0, 1]
    i, parity, repetition, frame = axes_para
    j, k = _NEXT_AXIS[i + parity], _NEXT_AXIS[i - parity + 1]
    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS4:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS4:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return [ax, ay, az]

# paras about right cameras, are auto caculated
left_camera_pose_mat = xyzrpy2mat(np.array(left_camera_pos+left_camera_rot))
right_camera_pos = np.array(left_camera_pos) + np.dot(left_camera_pose_mat, np.array([baseline, 0, 0, 0]).T).T[:3]
print(np.dot(left_camera_pose_mat, np.array([baseline, 0, 0, 0]).T).T[:3])
right_camera_rot = left_camera_rot  # rot to left, in degree

def set_camera_pose():
    pos, rot = left_camera_pos, left_camera_rot
    set_obj_pose(left_camera_name_in_scene, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
    pos, rot = right_camera_pos, right_camera_rot
    set_obj_pose(right_camera_name_in_scene, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

def gen_camera_paras(savepath=None):
    # return a dict
    res = {}
    P, K, RT = get_3x4_P_matrix_from_blender(bpy.data.objects[left_camera_name_in_scene])
    # if savepath is not None: np.savetxt(savepath + '_left_cameraK.txt', K)
    res['left_camera_k'] = np.array(K).tolist()
    res['left_camera_rt'] = np.array(RT).tolist()
    res['left_camera_p'] = np.array(P).tolist()
    P, K, RT = get_3x4_P_matrix_from_blender(bpy.data.objects[right_camera_name_in_scene])
    res['right_camera_k'] = np.array(K).tolist()
    res['right_camera_rt'] = np.array(RT).tolist()
    res['right_camera_p'] = np.array(P).tolist()
    return res
    
def gen_califile_from_render_camera_para(camera_para, path):
    K1, K2 = np.array(camera_para['left_camera_k']), np.array(camera_para['right_camera_k'])
    D1, D2 = np.zeros(5), np.zeros(5)
    RT1, RT2 = np.array(camera_para['left_camera_rt']), np.array(camera_para['right_camera_rt'])
    RT1 = np.insert(RT1, 3, values=np.array([0., 0., 0., 1.]), axis=0)
    RT2 = np.insert(RT2, 3, values=np.array([0., 0., 0., 1.]), axis=0)
    RT = np.dot(RT2, np.linalg.inv(RT1))
    R, T = RT[:3,:3], RT[:3, 3]
    outfile = cv2.FileStorage(path, cv2.FileStorage_WRITE)
    outfile.write('K1', K1)
    outfile.write('K2', K2)
    outfile.write('D1', D1)
    outfile.write('D2', D2)
    outfile.write('R', R)
    outfile.write('T', T)
    outfile.release()
    print("opencv stereo cali result saved to:" + path)

def set_obj_pose(name, x=None, y=None, z=None, rot_x=None, rot_y=None, rot_z=None):
    active_obj = bpy.data.objects[name] # caliboard
    active_obj.select_set(True)
    viewlayer.objects.active = active_obj
    if x is not None:
        bpy.context.object.location[0] = x
    if y is not None:
        bpy.context.object.location[1] = y
    if z is not None:
        bpy.context.object.location[2] = z
    if rot_x is not None:
        bpy.context.object.rotation_euler[0] = rot_x * np.pi / 180.0
    if rot_y is not None:
        bpy.context.object.rotation_euler[1] = rot_y * np.pi / 180.0
    if rot_z is not None:
        bpy.context.object.rotation_euler[2] = rot_z * np.pi / 180.0
        
        
def saveNormalizedExr(filepath, savepath, minVal=0, maxVal=2.0, use_16bit=False):
    exrMap = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # trunc
    exrMap[exrMap > maxVal] = maxVal
    # covert to uint
    if use_16bit: exrMap = ((exrMap-minVal)*(pow(2,16)-1)/(maxVal-minVal)).astype(np.uint16)
    else: exrMap = ((exrMap-minVal)*(pow(2,8)-1)/(maxVal-minVal)).astype(np.uint8)
    # save
    cv2.imwrite(savepath, exrMap, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    print("saved converted exr to:" + savepath)

def convert_depth_to_color(depth_map_meter):
    depth_image_color_vis = depth_map_meter.copy()
    # depth_image_color_vis[np.isnan(depth_image_color_vis)] = 0  # not working
    valid_points = np.where(depth_image_color_vis <= 10000000.000)
    depth_near_cutoff, depth_far_cutoff = np.percentile(depth_image_color_vis[valid_points], 5), np.percentile(depth_image_color_vis[valid_points], 95)
    if np.isnan(depth_far_cutoff):
        print("depth_far_cutoff is nan!")
        depth_far_cutoff = depth_near_cutoff + 10.0
    depth_far_cutoff = depth_near_cutoff + (depth_far_cutoff-depth_near_cutoff) * 1.2
    depth_range = depth_far_cutoff-depth_near_cutoff
    print((depth_near_cutoff, depth_far_cutoff))
    depth_image_color_vis[valid_points] = depth_far_cutoff - depth_image_color_vis[valid_points]  # - depth_near_cutoff
    depth_image_color_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_color_vis, alpha=255/(depth_range)), cv2.COLORMAP_JET)  #COLORMAP_JET HOT
    return depth_image_color_vis

def saveX3dDepth(filepath, savepath):
    depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    depth = depth[:,:,1]
    depth = convert_depth_to_color(depth)
    cv2.imwrite(savepath, depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])
    print("saved converted exr to:" + savepath)

scene = bpy.context.scene
viewlayer = bpy.context.view_layer
set_camera_pose()
scene.objects[left_camera_name_in_scene].data.angle = fov_angle
scene.objects[right_camera_name_in_scene].data.angle = fov_angle
scene.render.resolution_x = resolution_x
scene.render.resolution_y = resolution_y

scene.cycles.caustics_reflective = True
scene.cycles.caustics_refractive = False
scene.cycles.filter_width = 1
scene.cycles.samples = cycles_samples  # 128
scene.cycles.use_denoising = use_denoising  # True
scene.cycles.max_bounces = cycles_max_bounces  # 12, 8 or 0

def add_img_output_node():
    # clear all nodes
    tree = scene.node_tree
    for node in tree.nodes:
        # if n.label == 'Image'
        tree.nodes.remove(node)
    links = tree.links
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    # add image output node
    image_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    image_file_output.label = 'Image'
    links.new(render_layers.outputs['Image'], image_file_output.inputs[0])
    image_file_output.base_path = ''

# add_img_output_node()

def generate_pattern(path, max_bounces=8, use_hdr=False):
    scene.render.engine = 'CYCLES'
    scene.cycles.max_bounces = max_bounces
    scene.render.image_settings.color_mode = 'RGB'
    if use_hdr:
        scene.render.image_settings.file_format = "OPEN_EXR"
    else:
        scene.render.image_settings.file_format = 'BMP'
    scene.render.use_file_extension = True
    scene.view_settings.exposure = -0.3

    scene.camera = bpy.data.objects[left_camera_name_in_scene]
    for frame_nr in frames_range:
        # set current frame
        scene.frame_set(frame_nr)
        print("current frame:" + str(frame_nr))
        bpy.data.scenes[main_scene_name].render.filepath = path + str(frame_nr) + "_l"
        bpy.ops.render.render(write_still=True)
        
    scene.camera = bpy.data.objects[right_camera_name_in_scene]
    for frame_nr in frames_range:
        scene.frame_set(frame_nr)
        print("current frame:" + str(frame_nr))
        bpy.data.scenes[main_scene_name].render.filepath = path + str(frame_nr) + "_r"
        bpy.ops.render.render(animation=False, write_still=True)

    print("generate patterns end!")

def gen_gt(path):
    # generate depth and rendered patterns without light bounces
    scene.render.engine = 'CYCLES'
    scene.cycles.max_bounces = 0  # 12, 8 or 0
    scene.cycles.samples = 2
    scene.render.image_settings.color_mode = 'RGB'
    # 必须设置OPEN_EXR，否则无法输出深度
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = '16'
    scene.render.use_file_extension = True
    bpy.context.scene.render.use_compositing = True
    # set nodes
    scene.use_nodes = True
    
    tree = scene.node_tree
    links = tree.links
    # clear unused nodes
    for node in tree.nodes:
        if node.label == 'Depth GT Output':
            tree.nodes.remove(node)
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    # add depth node
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth GT Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    # set empty base path
    depth_file_output.base_path = ''

    # render the frames
    # set current frame, we only need 1 frame for gt gen
    frame_nr = 0
    scene.frame_set(frame_nr)
    frame_str = str(frame_nr)
    frame_append = "0"* (4-len(frame_str)) + str(frame_str)
    print("current frame:" + frame_append)
    
    # render left
    scene.camera = bpy.data.objects[left_camera_name_in_scene]
    bpy.data.scenes[main_scene_name].render.filepath = path + "left"
    depth_file_output.file_slots[0].path = scene.render.filepath + '_depth_'
    bpy.data.scenes[main_scene_name].render.filepath = path + str(frame_nr) + "_l"
    bpy.ops.render.render(animation=False, write_still=True)
    print(depth_file_output.file_slots[0].path + frame_append + ".exr")
    print(depth_file_output.file_slots[0].path + frame_append + ".png")
    # bpy.data.scenes[main_scene_name].node_tree.nodes['File Output'].format.use_zbuffer = True
    # convert exr file to png
    saveX3dDepth(depth_file_output.file_slots[0].path + frame_append + ".exr", depth_file_output.file_slots[0].path + frame_append + ".png")
    # saveNormalizedExr(image_file_output.file_slots[0].path + frame_append + ".exr", image_file_output.file_slots[0].path + frame_append + ".png")

    # render right
    scene.camera = bpy.data.objects[right_camera_name_in_scene]
    bpy.data.scenes[main_scene_name].render.filepath = path + "right"
    depth_file_output.file_slots[0].path = scene.render.filepath + '_depth_'
    bpy.data.scenes[main_scene_name].render.filepath = path + str(frame_nr) + "_r"
    bpy.ops.render.render(animation=False, write_still=True)
    # convert exr file to png
    print(depth_file_output.file_slots[0].path + frame_append + ".exr")
    print(depth_file_output.file_slots[0].path + frame_append + ".png")
    saveX3dDepth(depth_file_output.file_slots[0].path + frame_append + ".exr", depth_file_output.file_slots[0].path + frame_append + ".png")
    
    # record camera para and objects' pose
    camera_para_dict = gen_camera_paras(path + str(frame_nr))
    object_poses_dict = {}
    for obj in scene.objects:
        # print(obj.name)
        # print(obj.location.x)
        # print(obj.matrix_world)
        object_poses_dict[obj.name] =  np.array(obj.matrix_world).tolist()

    with open(path+"camera_para_dict.json", "w") as json_file:
        json.dump(camera_para_dict, json_file)
    # with open(path+"object_poses_dict.json", "w") as json_file:
    #     json.dump(object_poses_dict, json_file)  
    gen_califile_from_render_camera_para(camera_para_dict, path+"calibra.xml")
    # rm unsed files
    cmd = "rm " + path + "0_l.exr " + path + "0_r.exr"
    print("run cmd: " + cmd)
    os.system(cmd)
    scene.cycles.max_bounces = cycles_max_bounces  # 12, 8 or 0
    scene.cycles.samples = cycles_samples  # 128
    print("generate gt script end! the path is: " + path)

if gen_data_set:
    import time
    start_time = time.time()
    for data_set_idx in range(data_set_start_id, data_set_start_id + data_set_length):
        sample_start_time = time.time()
        # generate path
        current_id_str = "0"* (4-len(str(data_set_idx))) + str(data_set_idx)
        current_path = saving_path + current_id_str + "/"
        pattern_path = current_path
        gt_path = current_path

        print("current saving path: " + current_path)
        generate_pattern(pattern_path)
        print("generate patterns end, index:" + str(data_set_idx))
        # generate groundtruth
        gen_gt(gt_path)
        print("generate gt end, index:" + str(data_set_idx))
        # copy a preview
        # img = cv2.imread(pattern_path + "0_l.bmp", cv2.IMREAD_UNCHANGED)
        # cv2.imwrite(saving_path+current_id_str+".jpg", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        sample_time = (time.time() - sample_start_time) / 60.0
        print("sample time (in min):" + str(sample_time))
        # 拼接的四张图像
        image1 = cv2.imread(current_path + '0_l.bmp')
        image2 = cv2.imread(current_path + '0_r.bmp')
        image3 = cv2.imread(current_path + 'left_depth_0000.png')
        image4 = cv2.imread(current_path + 'right_depth_0000.png')
        h, w = image1.shape[:2]
        image = np.zeros((h*2, w*2, 3), np.uint8)
        image[0:h,0:w] = image1
        image[0:h,w:] = image2
        image[h:,0:w] = image3
        image[h:,w:] = image4
        cv2.imwrite(current_path + 'preview.jpg',image)
    runing_time = (time.time() - start_time) / 3600.0
    print("script end! total time: " + str(runing_time) + " hours")

# set back
scene.view_settings.exposure = -0.3
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.file_format = 'PNG'
