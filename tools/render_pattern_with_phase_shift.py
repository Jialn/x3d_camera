
"""
This file is not for general python env!
Should be runing inside.

For usage with dataset gen:
    set saving_path
    set gen_cali_images=True, generate cali_images to saving_path
    set gen_cali_images=False and data_set_start_id, data_set_length, then wait for done
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
import cv2
import json
import random
from bpy import context
from mathutils import Matrix, Vector

#######################
### Confs
gen_cali_images = False  # only for gen cali image, blender will exit after generated
gen_data_set = True

### for gen dataset
data_set_start_id = 0
data_set_length = 1
frames_range = range(0, 36)  # (0, 24) for dataset gen, smaller range for script testing
gen_gt_gen_pattern_without_bounces = True
gen_camera_para = True
### random ranges, in meters or degree
range_x = (-0.28, 0.28)
range_y = (-0.20, 0.20)
range_z = (0.0, 0.1)
range_rot_x = (-20.0, 20.0)
range_rot_y = (-20.0, 20.0)
range_rot_z = (-180.0, 180.0)
fixed_objs_in_scene = [] # reserved, TODO. fixed obj in scene, example: find obj by name in ["CupSet", "McLaren 720S (2017)"]
range_obj_number_in_one_scene = (3, 10)  # include left and right

### path
# saving_path = "/home/ubuntu/workplace/3dperceptionprototype/temp/dataset_render/"
saving_path = "/media/ubuntu/pm951_jt/dataset_render_with_phsft_pattern/"
cali_img_path = saving_path + 'cali_imgs/'
### path for 3d_cam
# cali_img_path = '/home/ubuntu/workplace/3dperceptionprototype/temp/stereocali_render/'
# pattern_path = '/home/ubuntu/workplace/3dperceptionprototype/x3d_camera/images_render/raw/'
# gt_path = '/home/ubuntu/workplace/3dperceptionprototype/x3d_camera/images_render/'

### set pos and rot
# depth_camera_pos = [0, 0, 1.0]  # [x, y, z], in meters
# depth_camera_rot_x = 0 # rotate degree around x-axis
depth_camera_pos = [0, -0.4, 0.8]  # [x, y, z] of left camera, in meters
projector_offset_x = 0.1
projector_rot_y = 4  # in degree
right_camera_offset_x = 0.25
right_camera_rot_y = 12  # in degree
depth_camera_rot_x = - 0.95 * np.arctan2(depth_camera_pos[1], depth_camera_pos[2])  * (180.0/3.1415926) # rotate degree around x-axis
### set fov
fov_angle = 45 * (3.1415926/180.0)
normalized_light_height = 0.75
bpy.data.lights["Light"].node_tree.nodes["映射.001"].inputs[3].default_value[0] = 1.0 / normalized_light_height
bpy.data.lights["Light"].node_tree.nodes["映射.001"].inputs[3].default_value[1] = 1.6 / normalized_light_height

resolution_x = 1296
resolution_y = 1024

cycles_samples = 64
cycles_max_bounces = 8 # 0 or 8


# resolution_x = 2592
# resolution_y = 2048

################
### random pose generating
def get_random_xyzrpy():
    x = random.uniform(range_x[0], range_x[1])
    y = random.uniform(range_y[0], range_y[1])
    z = random.uniform(range_z[0], range_z[1])
    rot_x = random.uniform(range_rot_x[0], range_rot_x[1]) * abs(z) / range_z[1]
    rot_y = random.uniform(range_rot_y[0], range_rot_y[1]) * abs(z) / range_z[1]
    rot_z = random.uniform(range_rot_z[0], range_rot_z[1])
    return x, y, z, rot_x, rot_y, rot_z

def random_set_pos(obj_list):
    obj_number_in_one_scene = random.randint(range_obj_number_in_one_scene[0], range_obj_number_in_one_scene[1])
    in_scene_list = random.sample(obj_list, k=obj_number_in_one_scene)
    # move all things away firstly
    for obj in obj_list:
        obj.location.x = 2.5
    for obj in in_scene_list:
        print(obj.name)
        x, y, z, rot_x, rot_y, rot_z = get_random_xyzrpy()
        print((x, y, z, rot_x, rot_y, rot_z))
        obj.location.x = x
        obj.location.y = y
        obj.location.z = z
        obj.rotation_euler[0] = rot_x * np.pi / 180.0
        obj.rotation_euler[1] = rot_y * np.pi / 180.0
        obj.rotation_euler[2] = rot_z * np.pi / 180.0

def random_set_light():  # default pos, xyz: (+-4, 0, 2)
    x, y, z = random.uniform(2, 10), random.uniform(-2, 2), random.uniform(0, 6)
    set_obj_pose('PlaneLight1', x=x, y=y, z=z)
    print("set light1: " + str((x,y,z)))
    x, y, z = -random.uniform(2, 10), random.uniform(-2, 2), random.uniform(0, 6)
    set_obj_pose('PlaneLight2', x=x, y=y, z=z)
    print("set light2: " + str((x,y,z)))

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
    P, K, RT = get_3x4_P_matrix_from_blender(bpy.data.objects["Camera"])
    # if savepath is not None: np.savetxt(savepath + '_left_cameraK.txt', K)
    res['left_camera_k'] = np.array(K).tolist()
    res['left_camera_rt'] = np.array(RT).tolist()
    res['left_camera_p'] = np.array(P).tolist()
    P, K, RT = get_3x4_P_matrix_from_blender(bpy.data.objects["Camera.right"])
    res['right_camera_k'] = np.array(K).tolist()
    res['right_camera_rt'] = np.array(RT).tolist()
    res['right_camera_p'] = np.array(P).tolist()
    return res
    

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

def saveX3dDepth(filepath, savepath):
    depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    depth = depth[:,:,1]
    depth = depth * 30000.0 # * 60000.0 / 2.0
    cv2.imwrite(savepath, depth.astype(np.uint16), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    print("saved converted exr to:" + savepath)


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


def set_pos_projector_and_cameras(x=None, y=None, z=None, rot_x=0):
    set_obj_pose("Camera", x, y, z, rot_x, 0, 0)
    if x is None:
        set_obj_pose("Camera.right", None, y, z, rot_x, right_camera_rot_y, 0)
    else:
        set_obj_pose("Camera.right", x+right_camera_offset_x, y, z, rot_x, right_camera_rot_y, 0)
    set_obj_pose("Light", x+projector_offset_x, y, z, 0, 0, 0)
    bpy.data.lights["Light"].node_tree.nodes["映射"].inputs[2].default_value[0] = - rot_x * np.pi / 180.0
    bpy.data.lights["Light"].node_tree.nodes["映射"].inputs[2].default_value[1] = - projector_rot_y * np.pi / 180.0


scene = bpy.context.scene
viewlayer = bpy.context.view_layer
set_pos_projector_and_cameras(x=depth_camera_pos[0], y=depth_camera_pos[1], z=depth_camera_pos[2], rot_x=depth_camera_rot_x)
scene.objects['Camera'].data.angle = fov_angle
scene.objects['Camera.right'].data.angle = fov_angle
scene.render.resolution_x = resolution_x
scene.render.resolution_y = resolution_y

scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.cycles.filter_width = 1
scene.cycles.samples = cycles_samples  # 128
scene.cycles.use_denoising = False  # True
scene.cycles.max_bounces = cycles_max_bounces  # 12, 8 or 0

# get all the objects
object_list = []
for sub_collection in scene.collection.children:
    print(sub_collection.name)
    if sub_collection.name in ["General", "Caliboard"]:
        continue
    for obj in sub_collection.all_objects:  # objects
        print("add object:" + obj.name)
        object_list.append(obj)

# save objects' pose before random moving
pose_list_backup = []
for obj in object_list:
    pose_list_backup.append(obj.matrix_world)

def add_img_output_node():
    # clear all nodes
    tree = scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    links = tree.links
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    # add image output node
    image_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    image_file_output.label = 'Image'
    links.new(render_layers.outputs['Image'], image_file_output.inputs[0])
    image_file_output.base_path = ''

add_img_output_node()

if gen_cali_images:
    # generate caliboard images
    path = cali_img_path
    scene.render.engine = 'BLENDER_EEVEE' # BLENDER_EEVEE for faster render when calibrating
    scene.eevee.use_volumetric_lights = False
    scene.eevee.use_soft_shadows = False
    scene.eevee.gi_irradiance_smoothing = 0
    scene.view_settings.gamma = 1.0
    scene.eevee.taa_render_samples = 64
    scene.view_settings.exposure = 8.0  # over exposed for stabler corner detection
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    scene.render.use_file_extension = True
    
    # a little bit higher to avoid caliboard out of view
    set_pos_projector_and_cameras(x=depth_camera_pos[0]*1.2, y=depth_camera_pos[1]*1.2, z=depth_camera_pos[2]*1.2, rot_x=depth_camera_rot_x)
        
    caliboard=bpy.data.objects["Cube.001"] # caliboard
    caliboard.select_set(True)
    viewlayer.objects.active = caliboard

    cali_pose_center = np.array([0.0, -0.1, 0.24, 20, 0, 0])
    cali_poses_diff = [
        # [3 pos, 3 rotation_euler ]
        [0.16,  -0.1, 0.05, 20, 0, 0],
        [-0.12, 0.02, 0.10, 0, 0, 0],
        [0.15,  0.05, -0.01, 0, -20, 0],
        [0.01,  0.09, 0.10, 0, 0, 0],
        [-0.13, 0.02, 0.08, 0, 20, 0],
        [0.16,  -0.15, 0.12, 0, 0, 0],
        [-0.15, 0.12, 0.01, 20, 0, 0],
        [0.01,  -0.2, 0.03, 0, 20, 0],
        [0.11,  0.10, 0.06, 0, 0, 20]
    ]

    for i in range(len(cali_poses_diff)):
        cali_pose = cali_pose_center + np.array(cali_poses_diff[i])
        # rotate and move caliboard
        bpy.context.object.location[0]      = cali_pose[0] * 1.2
        bpy.context.object.location[1]      = cali_pose[1]
        bpy.context.object.location[2]      = cali_pose[2]
        bpy.context.object.rotation_euler[0]= cali_pose[3] * np.pi / 180.0
        bpy.context.object.rotation_euler[1]= cali_pose[4] * np.pi / 180.0
        bpy.context.object.rotation_euler[2]= cali_pose[5] * np.pi / 180.0
        bpy.context.object.scale=(0.1,0.1,0.1)
        # render the image
        bpy.data.scenes["Scene"].render.filepath = path + "left_" + str(i)
        scene.camera = bpy.data.objects["Camera"]
        bpy.ops.render.render(write_still=True)
        bpy.data.scenes["Scene"].render.filepath = path + "right_" + str(i)
        scene.camera = bpy.data.objects["Camera.right"]
        bpy.ops.render.render(write_still=True)

    # move caliboard away
    bpy.context.object.location[0] = 10
    
    # set back
    scene.eevee.taa_render_samples = 64 # 64
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.exposure = -0.3
    set_pos_projector_and_cameras(x=depth_camera_pos[0], y=depth_camera_pos[1], z=depth_camera_pos[2], rot_x=depth_camera_rot_x)

    print("generate calibating script end! now you can run: \
        'python -m x3d_camera.stereo_calib temp/stereocali_render/' to calibrate \
         make sure use_render_image=True flag in the repo's globe config file.")
    exit()


def generate_pattern(path, max_bounces=8, use_hdr=False):
    scene.render.engine = 'CYCLES'
    scene.cycles.max_bounces = max_bounces
    scene.render.image_settings.color_mode = 'RGB'
    if use_hdr:
        scene.render.image_settings.file_format = "OPEN_EXR"
    else:
        scene.render.image_settings.file_format = 'BMP'
    scene.render.use_file_extension = True
    scene.view_settings.exposure = -0.7

    scene.camera = bpy.data.objects["Camera"]
    for frame_nr in frames_range:
        # set current frame
        scene.frame_set(frame_nr)
        print("current frame:" + str(frame_nr))
        bpy.data.scenes["Scene"].render.filepath = path + str(frame_nr) + "_l"
        bpy.ops.render.render(write_still=True)
        
    scene.camera = bpy.data.objects["Camera.right"]
    for frame_nr in frames_range:
        scene.frame_set(frame_nr)
        print("current frame:" + str(frame_nr))
        bpy.data.scenes["Scene"].render.filepath = path + str(frame_nr) + "_r"
        bpy.ops.render.render(animation=False, write_still=True)

    print("generate patterns end!")

def gen_one_pattern_for_3d_cam():
    # generate rendered patterns
    path = pattern_path
    generate_pattern(path)
    print("generate patterns end! If for 3D camera testing, now you can run: 'python -m x3d_camera.x3d_camera")

def gen_gt(path):
    # generate depth and rendered patterns without light bounces
    scene.cycles.max_bounces = 0  # 12, 8 or 0
    if gen_gt_gen_pattern_without_bounces:
        generate_pattern(path+"/raw_no_light_bounces/", max_bounces=0, use_hdr=False)
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.color_mode = 'RGB'
    # 必须设置OPEN_EXR，否则无法输出深度
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = '16'
    scene.render.use_file_extension = True
    # set nodes
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    # clear all nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    # add depth node
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    # set empty base path
    depth_file_output.base_path = ''

    # render the frames
    # set current frame, we only need 1 frame for gt gen
    frame_nr = 1
    scene.frame_set(frame_nr)
    frame_str = str(frame_nr)
    frame_append = "0"* (4-len(frame_str)) + str(frame_str)
    print("current frame:" + frame_append)
    
    # render left
    scene.camera = bpy.data.objects["Camera"]
    bpy.data.scenes["Scene"].render.filepath = path + "left"
    depth_file_output.file_slots[0].path = scene.render.filepath + '_depth_'
    bpy.data.scenes["Scene"].render.filepath = path + str(frame_nr) + "_l"
    bpy.ops.render.render(animation=False, write_still=True)
    # convert exr file to png
    saveX3dDepth(depth_file_output.file_slots[0].path + frame_append + ".exr", depth_file_output.file_slots[0].path + frame_append + ".png")
    # saveNormalizedExr(image_file_output.file_slots[0].path + frame_append + ".exr", image_file_output.file_slots[0].path + frame_append + ".png")

    # render right
    scene.camera = bpy.data.objects["Camera.right"]
    bpy.data.scenes["Scene"].render.filepath = path + "right"
    depth_file_output.file_slots[0].path = scene.render.filepath + '_depth_'
    bpy.data.scenes["Scene"].render.filepath = path + str(frame_nr) + "_r"
    bpy.ops.render.render(animation=False, write_still=True)
    # convert exr file to png
    saveX3dDepth(depth_file_output.file_slots[0].path + frame_append + ".exr", depth_file_output.file_slots[0].path + frame_append + ".png")
    
    # record camera para and objects' pose
    camera_para_dict = gen_camera_paras(path + str(frame_nr))
    object_poses_dict = {}
    for obj in scene.objects:
        print(obj.name)
        # print(obj.location.x)
        # print(obj.matrix_world)
        object_poses_dict[obj.name] =  np.array(obj.matrix_world).tolist()

    with open(path+"camera_para_dict.json", "w") as json_file:
        json.dump(camera_para_dict, json_file)
    with open(path+"object_poses_dict.json", "w") as json_file:
        json.dump(object_poses_dict, json_file)
        
    # rm unsed files
    cmd = "rm " + path + "1_l.exr " + path + "1_r.exr"
    print("run cmd: " + cmd)
    os.system(cmd)

    # clear all added nodes and add default node back
    for n in tree.nodes:
        tree.nodes.remove(n)
    add_img_output_node()
    # set back max_bounces
    scene.cycles.max_bounces = cycles_max_bounces  # 12, 8 or 0
    
    print("generate gt script end! the path is: " + path)

if gen_data_set:
    import time
    start_time = time.time()
    for data_set_idx in range(data_set_start_id, data_set_start_id + data_set_length):
        sample_start_time = time.time()
        # randomly move things
        # random_set_pos(object_list)
        random_set_light()
        # generate path
        current_id_str = "0"* (4-len(str(data_set_idx))) + str(data_set_idx)
        current_path = saving_path + current_id_str + "/"
        pattern_path = current_path + "raw/"
        gt_path = current_path

        print("current saving path: " + current_path)
        generate_pattern(pattern_path)
        print("generate patterns end, index:" + str(data_set_idx))
        # generate groundtruth
        gen_gt(gt_path)
        print("generate gt end, index:" + str(data_set_idx))
        # copy a preview
        img = cv2.imread(pattern_path + "0_l.bmp", cv2.IMREAD_UNCHANGED)
        cv2.imwrite(saving_path+current_id_str+".jpg", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        sample_time = (time.time() - sample_start_time) / 60.0
        print("sample time (in min):" + str(sample_time))

    runing_time = (time.time() - start_time) / 3600.0
    print("script end! total time: " + str(runing_time) + " hours")

# set back
scene.view_settings.exposure = -0.3
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.file_format = 'PNG'

# set back objects's pose
for i in range(len(object_list)):
    object_list[i].matrix_world = pose_list_backup[i]
