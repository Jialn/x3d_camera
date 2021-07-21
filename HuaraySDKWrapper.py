# Copyright (c) 2020. All Rights Reserved.
# Created on 2020-10-12
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
A Wrapper of Huaray Camera SDK with PDC 03 projector HW trigger support. 
Note that add some extra delays when using the new USB3.0 Camera.
"""
from MVSDK import *
from ImageConvert import *
import struct
import time
import datetime
import numpy as np
import cv2
import gc
from config import Config

g_cameraStatusUserInfo = b"statusInfo"
g_saving_flag = Config.save_pattern_to_disk  # save image or not
g_captured_frame_cnt_left = 0
g_captured_frame_cnt_right = 0  # for the second camera
g_captured_images_left = [None]*Config.pattern_end_index  # For capture one image, the image will be copied to g_last_captured_image
g_captured_images_right = [None]*Config.pattern_end_index  # For capture one image, the image will be copied to g_last_captured_image
g_captured_images_left_color = [None]*Config.pattern_end_index
g_captured_images_right_color = [None]*Config.pattern_end_index
g_scale = None
g_is_color_image = Config.is_color_camera
g_use_blue_channel = Config.use_blue_channel_for_color_camera
g_last_captured_image = None
g_image_paras = None


# Get frame callback function
def onGetFrameEx(frame, userInfo):
    # start_time = time.time()
    global g_captured_frame_cnt_left, g_last_captured_image, g_image_paras

    nRet = frame.contents.valid(frame)
    if (nRet != 0):
        print("frame is invalid!")
        frame.contents.release(frame)
        return
    userInfo = c_char_p(userInfo).value.decode()
    # print("BlockId = %d userInfo = %s"  %(frame.contents.getBlockId(frame), userInfo))

    imageParams = IMGCNV_SOpenParam()
    imageParams.dataSize    = frame.contents.getImageSize(frame)
    imageParams.height      = frame.contents.getImageHeight(frame)
    imageParams.width       = frame.contents.getImageWidth(frame)
    imageParams.paddingX    = frame.contents.getImagePaddingX(frame)
    imageParams.paddingY    = frame.contents.getImagePaddingY(frame)
    imageParams.pixelForamt = frame.contents.getImagePixelFormat(frame)
    g_image_paras = imageParams

    imageBuff = frame.contents.getImage(frame)
    userBuff = c_buffer(b'\0', imageParams.dataSize)
    memmove(userBuff, c_char_p(imageBuff), imageParams.dataSize)
    # 释放驱动图像缓存资源
    frame.contents.release(frame)
    # if image is Mono8: if imageParams.pixelForamt == EPixelType.gvspPixelMono8
    cvImageBuff = userBuff
    if Config.do_imagearray_reshape_in_camera_call_back:
        h, w = imageParams.height, imageParams.width
        cvImageBuff = np.array(bytearray(cvImageBuff)).reshape(h, w)
    g_last_captured_image = cvImageBuff
    if g_captured_frame_cnt_left < Config.pattern_end_index:
        g_captured_images_left[g_captured_frame_cnt_left] = cvImageBuff
    print(str(g_captured_frame_cnt_left) + '_l', end=" ", flush=True)
    g_captured_frame_cnt_left += 1
    if g_captured_frame_cnt_left == Config.pattern_end_index: gc.collect()
    # print("callback using %.5f s" % (time.time() - start_time))


# for the second camera
def onGetFrameExRight(frame, userInfo):
    global g_captured_frame_cnt_right, g_last_captured_image, g_image_paras

    nRet = frame.contents.valid(frame)
    if (nRet != 0):
        print("frame is invalid!")
        frame.contents.release(frame)
        return
    userInfo = c_char_p(userInfo).value.decode()
    imageParams = IMGCNV_SOpenParam()
    imageParams.dataSize    = frame.contents.getImageSize(frame)
    imageParams.height      = frame.contents.getImageHeight(frame)
    imageParams.width       = frame.contents.getImageWidth(frame)
    imageParams.paddingX    = frame.contents.getImagePaddingX(frame)
    imageParams.paddingY    = frame.contents.getImagePaddingY(frame)
    imageParams.pixelForamt = frame.contents.getImagePixelFormat(frame)
    g_image_paras = imageParams

    imageBuff = frame.contents.getImage(frame)
    userBuff = c_buffer(b'\0', imageParams.dataSize)
    memmove(userBuff, c_char_p(imageBuff), imageParams.dataSize)
    frame.contents.release(frame)
    cvImageBuff = userBuff
    if Config.do_imagearray_reshape_in_camera_call_back:
        h, w = imageParams.height, imageParams.width
        cvImageBuff = np.array(bytearray(cvImageBuff)).reshape(h, w)
    g_last_captured_image = cvImageBuff
    if g_captured_frame_cnt_right < Config.pattern_end_index:
        g_captured_images_right[g_captured_frame_cnt_right] = cvImageBuff
    print(str(g_captured_frame_cnt_right) + '_r', end=" ", flush=True)
    g_captured_frame_cnt_right += 1
    if g_captured_frame_cnt_left == Config.pattern_end_index: gc.collect()


# 相机连接状态回调函数
def deviceLinkNotify(connectArg, linkInfo):
    if (EVType.offLine == connectArg.contents.m_event):
        print("camera has off line, userInfo [%s]" % (c_char_p(linkInfo).value))
    elif (EVType.onLine == connectArg.contents.m_event):
        print("camera has on line, userInfo [%s]" % (c_char_p(linkInfo).value))


connectCallBackFuncEx = connectCallBackEx(deviceLinkNotify)
frameCallbackFuncEx = callbackFuncEx(onGetFrameEx)
frameCallbackFuncEx_1 = callbackFuncEx(onGetFrameExRight)


# 注册相机连接状态回调
def subscribeCameraStatus(camera):
    eventSubscribe = pointer(GENICAM_EventSubscribe())
    eventSubscribeInfo = GENICAM_EventSubscribeInfo()
    eventSubscribeInfo.pCamera = pointer(camera)
    nRet = GENICAM_createEventSubscribe(byref(eventSubscribeInfo), byref(eventSubscribe))
    if (nRet != 0):
        print("create eventSubscribe fail!")
        return -1

    nRet = eventSubscribe.contents.subscribeConnectArgsEx(eventSubscribe, connectCallBackFuncEx, g_cameraStatusUserInfo)
    if (nRet != 0):
        print("subscribeConnectArgsEx fail!")
        eventSubscribe.contents.release(eventSubscribe)
        return -1

    eventSubscribe.contents.release(eventSubscribe)
    return 0


# 反注册相机连接状态回调
def unsubscribeCameraStatus(camera):
    # 反注册上下线通知
    eventSubscribe = pointer(GENICAM_EventSubscribe())
    eventSubscribeInfo = GENICAM_EventSubscribeInfo()
    eventSubscribeInfo.pCamera = pointer(camera)
    nRet = GENICAM_createEventSubscribe(byref(eventSubscribeInfo), byref(eventSubscribe))
    if (nRet != 0):
        print("create eventSubscribe fail!")
        return -1
    nRet = eventSubscribe.contents.unsubscribeConnectArgsEx(eventSubscribe, connectCallBackFuncEx, g_cameraStatusUserInfo)
    if (nRet != 0):
        print("unsubscribeConnectArgsEx fail!")
        eventSubscribe.contents.release(eventSubscribe)
        return -1
    eventSubscribe.contents.release(eventSubscribe)
    return 0


# 设置软触发
def setSoftTriggerConf(camera):
    # 创建control节点
    acqCtrlInfo = GENICAM_AcquisitionControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AcquisitionControl())
    nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if (nRet != 0):
        print("create AcquisitionControl fail!")
        return -1

    # 设置触发源为软触发
    trigSourceEnumNode = acqCtrl.contents.triggerSource(acqCtrl)
    nRet = trigSourceEnumNode.setValueBySymbol(byref(trigSourceEnumNode), b"Software")
    if (nRet != 0):
        print("set TriggerSource value [Software] fail!")
        # 释放相关资源
        trigSourceEnumNode.release(byref(trigSourceEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # 需要释放Node资源
    trigSourceEnumNode.release(byref(trigSourceEnumNode))

    # 设置触发方式
    trigSelectorEnumNode = acqCtrl.contents.triggerSelector(acqCtrl)
    nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"FrameStart")
    if (nRet != 0):
        print("set TriggerSelector value [FrameStart] fail!")
        # 释放相关资源
        trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # 需要释放Node资源
    trigSelectorEnumNode.release(byref(trigSelectorEnumNode))

    # 打开触发模式
    trigModeEnumNode = acqCtrl.contents.triggerMode(acqCtrl)
    nRet = trigModeEnumNode.setValueBySymbol(byref(trigModeEnumNode), b"On")
    if (nRet != 0):
        print("set TriggerMode value [On] fail!")
        # 释放相关资源
        trigModeEnumNode.release(byref(trigModeEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # 需要释放相关资源
    trigModeEnumNode.release(byref(trigModeEnumNode))
    acqCtrl.contents.release(acqCtrl)

    return 0


# 设置外触发
def setLineTriggerConf(camera, hw_triger_delay):
    # 创建control节点
    acqCtrlInfo = GENICAM_AcquisitionControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AcquisitionControl())
    nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if (nRet != 0):
        print("create AcquisitionControl fail!")
        return -1

    # 设置触发源为软触发
    trigSourceEnumNode = acqCtrl.contents.triggerSource(acqCtrl)
    nRet = trigSourceEnumNode.setValueBySymbol(byref(trigSourceEnumNode), b"Line1")
    if (nRet != 0):
        print("set TriggerSource value [Line1] fail!")
        # 释放相关资源
        trigSourceEnumNode.release(byref(trigSourceEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # 需要释放Node资源
    trigSourceEnumNode.release(byref(trigSourceEnumNode))

    # 设置触发方式
    trigSelectorEnumNode = acqCtrl.contents.triggerSelector(acqCtrl)
    nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"FrameStart")
    if (nRet != 0):
        print("set TriggerSelector value [FrameStart] fail!")
        # 释放相关资源
        trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # 需要释放Node资源
    trigSelectorEnumNode.release(byref(trigSelectorEnumNode))

    # 打开触发模式
    trigModeEnumNode = acqCtrl.contents.triggerMode(acqCtrl)
    nRet = trigModeEnumNode.setValueBySymbol(byref(trigModeEnumNode), b"On")
    if (nRet != 0):
        print("set TriggerMode value [On] fail!")
        # 释放相关资源
        trigModeEnumNode.release(byref(trigModeEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # 需要释放Node资源
    trigModeEnumNode.release(byref(trigModeEnumNode))

    # 设置delay, for new USB3.0 camera
    if hw_triger_delay != 0:
        triggerDelayDoubleNode = acqCtrl.contents.triggerDelay(acqCtrl)
        nRet = triggerDelayDoubleNode.setValue(byref(triggerDelayDoubleNode), hw_triger_delay * 1000.0)
        if (nRet != 0):
            print("set TriggerActivation value [RisingEdge] fail!")
            # 释放相关资源
            triggerDelayDoubleNode.release(byref(triggerDelayDoubleNode))
            acqCtrl.contents.release(acqCtrl)
            return -1

        # 需要释放Node资源
        triggerDelayDoubleNode.release(byref(triggerDelayDoubleNode))

    # 设置触发沿
    trigActivationEnumNode = acqCtrl.contents.triggerActivation(acqCtrl)
    nRet = trigActivationEnumNode.setValueBySymbol(byref(trigActivationEnumNode), b"RisingEdge")
    if (nRet != 0):
        print("set TriggerActivation value [RisingEdge] fail!")
        # 释放相关资源
        trigActivationEnumNode.release(byref(trigActivationEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1

    # 需要释放Node资源
    trigActivationEnumNode.release(byref(trigActivationEnumNode))
    acqCtrl.contents.release(acqCtrl)
    return 0


# 打开相机
def openCamera(camera):
    # 连接相机
    nRet = camera.connect(camera, c_int(GENICAM_ECameraAccessPermission.accessPermissionControl))
    if (nRet != 0):
        print("camera connect fail!")
        return -1
    else:
        print("camera connect success.")
    # 注册相机连接状态回调
    nRet = subscribeCameraStatus(camera)
    if (nRet != 0):
        print("subscribeCameraStatus fail!")
        return -1

    return 0


# 关闭相机
def closeCamera(camera):
    # 反注册相机连接状态回调
    nRet = unsubscribeCameraStatus(camera)
    if (nRet != 0):
        print("unsubscribeCameraStatus fail!")
        return -1

    # 断开相机
    nRet = camera.disConnect(byref(camera))
    if (nRet != 0):
        print("disConnect camera fail!")
        return -1

    return 0


def set_double_node_value(camera, attrName, dVal):
    # 通用属性设置:构造doubleNode节点
    attrValueNode = pointer(GENICAM_DoubleNode())
    attrValueNodeInfo = GENICAM_DoubleNodeInfo()
    attrValueNodeInfo.pCamera = pointer(camera)
    attrValueNodeInfo.attrName = attrName
    nRet = GENICAM_createDoubleNode(byref(attrValueNodeInfo), byref(attrValueNode))
    attrName = c_char_p(attrName).value.decode()
    if (nRet != 0):
        print(attrName + ": create Node fail!")
        return -1

    nRet = attrValueNode.contents.setValue(attrValueNode, c_double(dVal))
    if (nRet != 0):
        print(attrName + ": set value [%f] fail!" % (dVal))
        # 释放相关资源
        attrValueNode.contents.release(attrValueNode)
        return -1
    else:
        print(attrName + ": set value [%f] success." % (dVal))

    # 释放节点资源
    attrValueNode.contents.release(attrValueNode)


def setWhiteBalanceAuto(camera, enable=True):
    # note this does not works well in low light, use setCorrectedWhiteBalance instead
    acqCtrlInfo = GENICAM_AnalogControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AnalogControl())
    nRet = GENICAM_createAnalogControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if (nRet != 0):
        print("create AnalogControl fail!")
        return -1
    # AnalogControl GENICAM_AnalogControl_balanceWhiteAuto
    trigSelectorEnumNode = acqCtrl.contents.balanceWhiteAuto(acqCtrl)
    if enable:
        nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"Once")  # Once Continuous Off
    else:    
        nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"Off")  # Once Continuous Off
    if (nRet != 0):
        print("set setWhiteBalance fail!")
        # 释放相关资源
        trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return -1
    else:
        print("set white balance suc!")
    # 需要释放Node资源
    trigSelectorEnumNode.release(byref(trigSelectorEnumNode))


# setWhiteBalance
def setCorrectedWhiteBalance(camera, red_ratio, blue_ratio):
    setWhiteBalanceAuto(camera, enable=False)
    # 创建control节点
    acqCtrlInfo = GENICAM_AnalogControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AnalogControl())
    nRet = GENICAM_createAnalogControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if (nRet != 0):
        print("create AnalogControl fail!")
        return -1
    # AnalogControl GENICAM_AnalogControl_balanceWhiteAuto GENICAM_AnalogControl_balanceRatioSelector
    trigSelectorEnumNode = acqCtrl.contents.balanceRatioSelector(acqCtrl)
    nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"Red")  # Once Continuous Off
    if (nRet != 0):
        print("set setWhiteBalance fail!")
        trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return
    else:
        print("set white balance suc!")
    set_double_node_value(camera, b"BalanceRatio", red_ratio)
    nRet = trigSelectorEnumNode.setValueBySymbol(byref(trigSelectorEnumNode), b"Blue")  # Once Continuous Off
    if (nRet != 0):
        print("set setWhiteBalance fail!")
        trigSelectorEnumNode.release(byref(trigSelectorEnumNode))
        acqCtrl.contents.release(acqCtrl)
        return
    else:
        print("set white balance suc!")
    set_double_node_value(camera, b"BalanceRatio", blue_ratio)
    # 需要释放Node资源
    trigSelectorEnumNode.release(byref(trigSelectorEnumNode))


# 设置Gain, from 1 to 32
def setGain(camera, dVal):
    attrName = b"GainRaw"
    set_double_node_value(camera, attrName, dVal)

# Gamma, from 0.1 to 2.0
def setGamma(camera, dVal):
    attrName = b"Gamma"
    set_double_node_value(camera, attrName, dVal)


# 设置曝光
def setExposureTime(camera, dVal):
    # if expo time > 90ms, the camera could met internal error. using gain instead
    if dVal > 90000:
        setGain(camera, dVal/90000)
        dVal = 90000
    else:
        setGain(camera, 1.0)
        dVal = dVal
    # 通用属性设置:设置曝光 --根据属性类型，直接构造属性节点。如曝光是 double类型，构造doubleNode节点
    exposureTimeNode = pointer(GENICAM_DoubleNode())
    exposureTimeNodeInfo = GENICAM_DoubleNodeInfo()
    exposureTimeNodeInfo.pCamera = pointer(camera)
    exposureTimeNodeInfo.attrName = b"ExposureTime"
    nRet = GENICAM_createDoubleNode(byref(exposureTimeNodeInfo), byref(exposureTimeNode))
    if (nRet != 0):
        print("create ExposureTime Node fail!")
        return -1
    # 设置曝光时间
    nRet = exposureTimeNode.contents.setValue(exposureTimeNode, c_double(dVal))
    if (nRet != 0):
        print("set ExposureTime value [%f]us fail!" % (dVal))
        exposureTimeNode.contents.release(exposureTimeNode)
        return -1
    else:
        print("set ExposureTime value [%f]us success." % (dVal))
    # 释放节点资源
    exposureTimeNode.contents.release(exposureTimeNode)
    return 0


# 枚举相机
def enumCameras():
    system = pointer(GENICAM_System())
    nRet = GENICAM_getSystemInstance(byref(system))
    if (nRet != 0):
        print("getSystemInstance fail!")
        return None, None
    # 发现相机
    cameraList = pointer(GENICAM_Camera())
    cameraCnt = c_uint()
    nRet = system.contents.discovery(system, byref(cameraList), byref(cameraCnt), c_int(GENICAM_EProtocolType.typeAll))
    if (nRet != 0):
        print("discovery fail!")
        return None, None
    elif cameraCnt.value < 1:
        print("discovery no camera!")
        return None, None
    else:
        print("cameraCnt: " + str(cameraCnt.value))
        return cameraCnt.value, cameraList


def grabOne(camera):
    global g_captured_frame_cnt_left, g_captured_frame_cnt_right, g_last_captured_image
    g_captured_frame_cnt_left, g_captured_frame_cnt_right = 0, 0
    g_last_captured_image = None
    # 创建流对象
    streamSourceInfo = GENICAM_StreamSourceInfo()
    streamSourceInfo.channelId = 0
    streamSourceInfo.pCamera = pointer(camera)

    streamSource = pointer(GENICAM_StreamSource())
    nRet = GENICAM_createStreamSource(pointer(streamSourceInfo), byref(streamSource))
    if (nRet != 0):
        print("create StreamSource fail!")
        return None
    # 创建control节点
    acqCtrlInfo = GENICAM_AcquisitionControlInfo()
    acqCtrlInfo.pCamera = pointer(camera)
    acqCtrl = pointer(GENICAM_AcquisitionControl())
    nRet = GENICAM_createAcquisitionControl(pointer(acqCtrlInfo), byref(acqCtrl))
    if (nRet != 0):
        print("create AcquisitionControl fail!")
        # 释放相关资源
        streamSource.contents.release(streamSource)
        return None
    # 执行一次软触发
    g_last_captured_image = None
    trigSoftwareCmdNode = acqCtrl.contents.triggerSoftware(acqCtrl)
    nRet = trigSoftwareCmdNode.execute(byref(trigSoftwareCmdNode))
    if (nRet != 0):
        print("Execute triggerSoftware fail!")
        # 释放相关资源
        trigSoftwareCmdNode.release(byref(trigSoftwareCmdNode))
        acqCtrl.contents.release(acqCtrl)
        streamSource.contents.release(streamSource)
        return None
    # 释放相关资源
    trigSoftwareCmdNode.release(byref(trigSoftwareCmdNode))
    acqCtrl.contents.release(acqCtrl)
    streamSource.contents.release(streamSource)
    # wait callback done
    while True:
        time.sleep(0.002)
        if g_last_captured_image is not None: break
    return g_last_captured_image


def init_camera(cam_list, cam_index, hw_triger, hw_triger_delay, exp_time, call_back_fun):
    """
    init camera with hw trigger or not

        exp_time: in us
    """
    camera = cam_list[cam_index]
    # Open camera
    nRet = openCamera(camera)
    if (nRet != 0):
        print("openCamera fail.")
        return -1
    # 创建流对象
    streamSourceInfo = GENICAM_StreamSourceInfo()
    streamSourceInfo.channelId = 0
    streamSourceInfo.pCamera = pointer(camera)
    streamSource = pointer(GENICAM_StreamSource())
    nRet = GENICAM_createStreamSource(pointer(streamSourceInfo), byref(streamSource))
    if (nRet != 0):
        print("create StreamSource fail!")
        return -1
    # Set external HW line trigger
    if hw_triger:
        setLineTriggerConf(camera, hw_triger_delay)
    else:
        setSoftTriggerConf(camera)
    # set expo and white balance
    setExposureTime(camera, exp_time)
    if Config.is_color_camera:
        setCorrectedWhiteBalance(camera=camera, red_ratio=Config.white_balance_red_ratio, blue_ratio=Config.white_balance_blue_ratio)
    setGamma(camera, Config.gamma)
    # 注册拉流回调函数
    userInfo = str(cam_index).encode()  # to bytes #b'test'
    nRet = streamSource.contents.attachGrabbingEx(streamSource, call_back_fun, userInfo)
    if (nRet != 0):
        print("attachGrabbingEx fail!")
        streamSource.contents.release(streamSource)
        return -1
    # 开始拉流
    nRet = streamSource.contents.startGrabbing(streamSource, c_ulonglong(0), c_int(GENICAM_EGrabStrategy.grabStrartegySequential))
    if (nRet != 0):
        print("startGrabbing fail!")
        streamSource.contents.release(streamSource)
        return -1
    return camera, streamSource


def close_camera(camera, streamSource):
    # 反注册回调函数
    userInfo = b"test"
    nRet = streamSource.contents.detachGrabbingEx(streamSource, frameCallbackFuncEx, userInfo)
    if (nRet != 0):
        print("detachGrabbingEx fail!")
        streamSource.contents.release(streamSource)
        return -1
    # 停止拉流
    nRet = streamSource.contents.stopGrabbing(streamSource)
    if (nRet != 0):
        print("stopGrabbing fail!")
        streamSource.contents.release(streamSource)
        return -1
    # 关闭相机
    nRet = closeCamera(camera)
    if (nRet != 0):
        print("closeCamera fail")
        streamSource.contents.release(streamSource)
        return -1
    # 释放资源
    streamSource.contents.release(streamSource)


def find_and_open_cameras(camera_ids, hw_triger, hw_triger_delay, exp_time):
    # Find cameras
    cameraCnt, cameraList = enumCameras()
    if cameraCnt is None:
        return -1
    # Show Camera info
    for index in range(0, cameraCnt):
        camera = cameraList[index]
        print("\nCamera Id = " + str(index))
        print("Key           = " + str(camera.getKey(camera)))
        print("vendor name   = " + str(camera.getVendorName(camera)))
        print("Model  name   = " + str(camera.getModelName(camera)))
        print("Serial number = " + str(camera.getSerialNumber(camera)))
    # Init Cameras
    cameras, stream_sources = [], []
    if cameraCnt == 3: 
        print("find 3 camera!")
        camera_id_offset = 1
    else:  camera_id_offset = 0
    camera, stream_source = init_camera(
        cam_list=cameraList, cam_index=camera_ids[0] + camera_id_offset,
        hw_triger=hw_triger, hw_triger_delay=hw_triger_delay,
        exp_time=exp_time,
        call_back_fun=frameCallbackFuncEx)
    cameras.append(camera)
    stream_sources.append(stream_source)
    if len(camera_ids) > 1:
        camera_1, stream_source_1 = init_camera(
            cam_list=cameraList, cam_index=camera_ids[1] + camera_id_offset,
            hw_triger=hw_triger, hw_triger_delay=hw_triger_delay,
            exp_time=exp_time,
            call_back_fun=frameCallbackFuncEx_1)
        cameras.append(camera_1)
        stream_sources.append(stream_source_1)
    return cameras, stream_sources


def is_color_image():
    global g_is_color_image
    return g_is_color_image

def close_cameras(cameras, stream_sources):
    for i in range(len(cameras)):
        close_camera(cameras[i], stream_sources[i])

def set_hw_trigger(camera, hw_triger_delay):
    setLineTriggerConf(camera, hw_triger_delay)

def set_sw_trigger(camera):
    setSoftTriggerConf(camera)


def try_scan_all_patterns(synced_projector, camera_ids, frame_end, scan_time):
    global g_captured_frame_cnt_left, g_captured_frame_cnt_right
    g_captured_frame_cnt_left, g_captured_frame_cnt_right = Config.pattern_start_index, Config.pattern_start_index 

    # Trigger projector patterns    
    if Config.use_high_speed_projector:
        synced_projector.ProjectPatternOnce()
    else:
        synced_projector.scan_all_pattern(scan_time[0], scan_time[1], scan_time[2])
    # Wait to get the images
    time_out = 3  # s
    cnt = 0
    while (g_captured_frame_cnt_left < frame_end):
        time.sleep(0.002)
        cnt += 1
        if cnt > time_out / 0.002: return False

    if len(camera_ids) > 1:
        while (g_captured_frame_cnt_right < frame_end):
            time.sleep(0.002)
            cnt += 1
            if cnt > time_out / 0.002: return False

    return True


def capture_all_patterns(synced_projector, camera_ids, path, scan_time):
    global g_saving_flag, g_scale, g_is_color_image, g_use_blue_channel, g_image_paras
    frame_start, frame_end = Config.pattern_start_index, Config.pattern_end_index
    start_time = time.time()
    while(True):
        flag = try_scan_all_patterns(synced_projector, camera_ids, frame_end, scan_time)
        if flag: break
    print(" Pure capture time: %.3f s" % (time.time() - start_time))
    start_time = time.time()
    h, w = g_image_paras.height, g_image_paras.width
    if not Config.do_imagearray_reshape_in_camera_call_back:
        for i in range(frame_start, frame_end):
                g_captured_images_left[i] = np.array(bytearray(g_captured_images_left[i])).reshape(h, w)
                g_captured_images_right[i] = np.array(bytearray(g_captured_images_right[i])).reshape(h, w)

    if g_is_color_image and Config.do_demosac_for_color_camera:
        for i in range(frame_start, frame_end):
            g_captured_images_left_color[i] = cv2.cvtColor(g_captured_images_left[i], cv2.COLOR_BAYER_BG2BGR)
            g_captured_images_right_color[i] = cv2.cvtColor(g_captured_images_right[i], cv2.COLOR_BAYER_BG2BGR) # cv2.COLOR_BAYER_BG2GRAY
            if g_use_blue_channel:
                g_captured_images_left[i] = g_captured_images_left_color[i][:,:,0]  # spilt channle bule in BGR
                g_captured_images_right[i] = g_captured_images_right_color[i][:,:,0]  # spilt channle bule in BGR
            else:
                g_captured_images_left[i] = cv2.cvtColor(g_captured_images_left_color[i], cv2.COLOR_RGB2GRAY)
                g_captured_images_right[i] = cv2.cvtColor(g_captured_images_right_color[i], cv2.COLOR_RGB2GRAY)

    for i in range(frame_start, frame_end):
        if g_scale != 1.0 or (g_is_color_image and Config.do_demosac_for_color_camera):
            if g_scale is not None: # and g_scale != 1.0:
                g_captured_images_left[i] = cv2.resize(g_captured_images_left[i], (round(w * g_scale), round(h * g_scale)), interpolation=cv2.INTER_LINEAR)
                g_captured_images_right[i] = cv2.resize(g_captured_images_right[i], (round(w * g_scale), round(h * g_scale)), interpolation=cv2.INTER_LINEAR)
        if g_saving_flag:
            cv2.imwrite(path + str(i) + '_l' + '.bmp', g_captured_images_left[i])
            cv2.imwrite(path + str(i) + '_r' + '.bmp', g_captured_images_right[i])
    print("Pre-process time(reshape bytearray, optional demosac, scale, save image, etc): %.3f s" % (time.time() - start_time))


def hdr_amplify_diff_of_inv(img_list, img_list_high_exp, save_path=None):
    # hdr2: add (highexp_image - highexp_image_inv) to low exp image
    # for 7 + 1 + 4 phase shift only
    gray_code_range = (0, 10)
    # gray code
    for cnt in range(Config.pattern_start_index+gray_code_range[0], Config.pattern_start_index+gray_code_range[1]):
        img_list[cnt] =  img_list[cnt] // 2 + img_list_high_exp[cnt] // 2
        if Config.save_pattern_to_disk: cv2.imwrite(save_path + str(cnt) + "hdr.jpg", img_list[cnt])
    # phsft
    for cnt in range(Config.pattern_start_index+gray_code_range[1], Config.pattern_start_index+gray_code_range[1]+2, 1):
        high = img_list_high_exp[cnt].astype(np.int16)
        high_inv =  img_list_high_exp[cnt+2].astype(np.int16)
        high_diff = high - high_inv
        # high_diff = (high_diff * high_pattern_weight).astype(np.int16)
        high_diff_inv = - high_diff
        low = img_list[cnt].astype(np.int16)
        low_inv =  img_list[cnt+2].astype(np.int16)
        hdr = low + high_diff
        img_list[cnt] = np.clip(hdr, 0, 255).astype(np.uint8)
        hdr_inv = low_inv + high_diff_inv
        img_list[cnt+2] = np.clip(hdr_inv, 0, 255).astype(np.uint8)
        if Config.save_pattern_to_disk: cv2.imwrite(save_path + str(cnt) + "hdr.jpg", img_list[cnt])
        if Config.save_pattern_to_disk: cv2.imwrite(save_path + str(cnt+2) + "hdr.jpg", img_list[cnt+2])

def hdr_using_reflective_ratio(img_list, img_list_high_exp, save_path=None):
    # for 7 + 1 + 4 phase shift only
    ref = img_list[Config.pattern_start_index+0] # 0 light on, 1 light off
    ref_mean = np.mean(ref)
    ref = ref / ref_mean  # normalize mean to 1.0
    print(ref_mean)
    ref_max = 255.0/ref_mean
    high_weight = ref_max - ref
    low_weight =  ref
    high_weight = high_weight * 0.5 / np.mean(high_weight)
    low_weight = low_weight * 0.5 / np.mean(low_weight)
    # gray code
    image_range = range(Config.pattern_start_index+0, Config.pattern_start_index+10)
    for cnt in image_range:
        img_list[cnt] =  img_list[cnt] // 2 + img_list_high_exp[cnt] // 2
        if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])
    # phsft
    image_range = range(Config.pattern_start_index+10, Config.pattern_start_index+14)
    if Config.use_high_speed_projector: # high spd projector needs fixed expo time, should use diff to elimate env light
        for cnt in image_range:
            diff_of_higher_prj = img_list_high_exp[cnt].astype(np.int16) - img_list[cnt]
            hdr = diff_of_higher_prj * high_weight + img_list[cnt] #  * low_weight
            img_list[cnt] = np.clip(hdr, 0, 255).astype(np.uint8)
            if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])
    else:
        for cnt in image_range:
            hdr = img_list_high_exp[cnt] * high_weight + img_list[cnt] * low_weight
            img_list[cnt] = np.clip(hdr, 0, 255).astype(np.uint8)
            if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])

def hdr_16bit(img_list, img_list_high_exp, save_path=None):
    # return 16 bit hdr images
    # only for 7+1+4 phsft pattern
    # the following procedures should be able to handle 16bit unsigned short images
    image_range = range(Config.pattern_start_index+0, Config.pattern_start_index+14)
    for cnt in image_range:
        # low_expo_image = img_list[cnt].astype(np.uint16)
        # gamma_corrected_low_expo_image = (low_expo_image + 255) / 255 * low_expo_image
        gamma_corrected_low_expo_image = img_list[cnt].astype(np.uint16)
        hdr = (Config.hdr_high_exp_rate * gamma_corrected_low_expo_image).astype(np.uint16)
        low_expo_pts = np.where(img_list[cnt] < 32)
        hdr[low_expo_pts] = img_list_high_exp[cnt][low_expo_pts]
        img_list[cnt] = hdr
        if Config.save_pattern_to_disk: cv2.imwrite(save_path[:-2] + str(cnt) + save_path[-2:] + ".bmp", img_list[cnt])

hdr_preprocess = hdr_using_reflective_ratio

class HuarayCamera():
    """ Class to Wrap Huaray Industrial Camera API.
    Support HW trigger, not support SW trigger.
    """

    def __init__(self, hw_trigger=True, scale=None, logging=False):
        """Init

        Args:
            hw_trigger: use hardware trigger or not
            scale: scale the image
        """
        self._logging = logging
        # camera parameters
        camera_ids = Config.camera_ids  # [0, 1]  # The id of camera. For Dual cam, it could be [0,1] or [1,2]. Note that 0 could be the usb webcam if a laptop is used. 
        self.switch_left_right = Config.switch_left_right
        self.enable_auto_exposure = Config.enable_auto_exposure  # True  # open this to find a better exp time
        self.enable_hdr = Config.enable_hdr  # False
        self.exp_time = Config.exp_time  # 7500.0 # exposure time for 3D projector, us
        self.exp_time_env = Config.exp_time_env  # 50000.0 # exposure time for environmet images without projector openning
        self._hdr_high_exp_rate = Config.hdr_high_exp_rate  # 4.0  # high exposure time = hdr_high_exp_rate * exp_time
        self.hw_triger_delay = Config.hw_triger_delay  # 20  # 10
        self.scan_time = Config.scan_time  # projector pattern scan time: interval_time, start_index, display_time
        # init PDC projector
        if hw_trigger:
            if Config.use_high_speed_projector:
                from projector_lcp4500 import PyLCR4500
                self._projector = PyLCR4500()
            else:
                from projector_pdc03 import PyPDC
                self._projector = PyPDC(port=Config.projector_port, logging=False)
                self._projector.shake_hands()
        # init cameras
        self.hw_triger = False  # for possible initial auto_exposure
        if len(camera_ids) > 1 and self.switch_left_right:
            print("switch left and right camera")
            camera_ids[0], camera_ids[1] = camera_ids[1], camera_ids[0]
        self.cameras, self.stream_sources = find_and_open_cameras(camera_ids, self.hw_triger, self.hw_triger_delay, self.exp_time)
        self.camera_ids = camera_ids
        if self.enable_auto_exposure:  # auto expo for env and pattern
            self.auto_exposure_env()
            self.auto_exposure_for_pattern()
        if hw_trigger:
            for camera in self.cameras:
                set_hw_trigger(camera, self.hw_triger_delay)
            self.hw_triger = True
        global g_scale
        g_scale = scale
        global g_captured_images_left, g_captured_images_right
        self.captured_images_left = g_captured_images_left
        self.captured_images_right = g_captured_images_right
        self.captured_images_left_color = g_captured_images_left_color
        self.captured_images_right_color = g_captured_images_right_color

    def auto_exposure_for_pattern(self):
        # setexposure, make mean to proper_range
        max_try = 20
        proper_range = Config.auto_expo_range_for_pattern  # [40, 60] # should have low exp for patterns
        if not Config.use_high_speed_projector:
            ref_pattern_idx = 18
            self._projector.scan_one_pattern(interval_time=0, index=ref_pattern_idx, display_time=50)
            time.sleep(0.2)
        # else:
        #     ref_pattern_idx = 7
        #     time_us = 50 * 1000
        #     self._projector.UpdateProjectorSetting(pat_exp_time=time_us, frame_period=time_us, led_current=Config.led_current)
        #     self._projector.scan_one_pattern(index=ref_pattern_idx)
        #     time.sleep(0.2)

        for try_cnt in range(max_try):
            for cam in self.cameras:
                setExposureTime(cam, self.exp_time)
            image = self.get_one_frame(self.camera_ids[0])
            h, w = image.shape[:2]
            h_qut, w_qut = h // 4, w // 4
            mean = np.mean(image[h_qut:h_qut*3, w_qut:w_qut*3])
            print(mean)
            if mean < proper_range[0]:
                self.exp_time = self.exp_time * 1.2
            elif mean > proper_range[1]:
                self.exp_time = self.exp_time * 0.8
            else:
                print("auto exposure time:" + str(self.exp_time) + "\t average brightness:" + str(mean))
                break
            if try_cnt >= max_try-1: print("failed to find a proper expo time")
            if self.exp_time > 80000.0:
                self.exp_time = 80000.0
                break
        print("auto exposure time:" + str(self.exp_time) + "\t average brightness:" + str(mean))
        max_expo_time = self.exp_time * self._hdr_high_exp_rate if self.enable_hdr else self.exp_time
        if self.scan_time[-1]*1000 < max_expo_time:
            self.scan_time[-1] = (int)(max_expo_time/1000.0) + 1
            print("set projector display time: "+str(self.scan_time[-1]))
        # if Config.use_high_speed_projector:
        #     self._projector.UpdateProjectorSetting(pat_exp_time=Config.pat_exp_time, frame_period=Config.frame_period, led_current=Config.led_current)

    def auto_exposure_env(self):
        # auto exposure for projector off, make mean to proper_range
        max_try = 20
        proper_range = Config.auto_expo_range_for_env_light  # [80, 120]
        if not Config.use_high_speed_projector:
            ref_pattern_idx = 1  # 1 for all off
            self._projector.scan_one_pattern(interval_time=0, index=ref_pattern_idx, display_time=50)
            time.sleep(0.2)
        for try_cnt in range(max_try):
            for cam in self.cameras:
                setExposureTime(cam, self.exp_time_env)
            image = self.get_one_frame(self.camera_ids[0])
            h, w = image.shape[:2]
            h_qut, w_qut = h // 4, w // 4
            mean = np.mean(image[h_qut:h_qut*3, w_qut:w_qut*3])
            if mean < proper_range[0]:
                self.exp_time_env = self.exp_time_env * 1.2
            elif mean > proper_range[1]:
                self.exp_time_env = self.exp_time_env * 0.8
            else: break
            if try_cnt >= max_try-1: print("failed to find a proper expo time")
        print("auto exposure time:" + str(self.exp_time_env) + "\t average brightness:" + str(mean))

    def update_exp_time(self, exp_time=None):
        if exp_time is None:
            exp_time = self.exp_time
        for cam in self.cameras:
            setExposureTime(cam, exp_time)

    def get_one_frame(self, camera_id, force_gray=True, use_env_exp=False):
        """ Grab one frame by sw trigger.
        Note: using get_one_frame and get_projected_raw_images mixed could make some hw trigger fails.
            Have not found the solution for this problem.

        Args:
            camera_id: The id of camera. For Dual cam, it could be 0, 1 or 2.
                Note that 0 could be the usb webcam if a laptop is used. 
            force_gray: the image could be gray or color depends on camera. Force to convert to gray if it is a color camera. 
        """
        global g_scale, g_image_paras
        if camera_id not in self.camera_ids:
            print("no such camera!")
            return
        idx = self.camera_ids.index(camera_id)
        camera = self.cameras[idx]
        if use_env_exp:
            self.update_exp_time(self.exp_time_env)
        if self.hw_triger:
            set_sw_trigger(camera)
        ret = grabOne(camera)
        if not Config.do_imagearray_reshape_in_camera_call_back:
            ret = np.array(bytearray(ret)).reshape(g_image_paras.height, g_image_paras.width)
        h, w = ret.shape[:2]
        if self.is_color_camera():
            ret = cv2.cvtColor(ret, cv2.COLOR_BAYER_BG2BGR)
        if force_gray and self.is_color_camera():
            # print("convert color to gray!")
            ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        if g_scale is not None:
            ret = cv2.resize(ret, (round(w * g_scale), round(h * g_scale)), interpolation=cv2.INTER_LINEAR)
        # set back trigger and exp time
        if self.hw_triger:
            set_hw_trigger(camera, self.hw_triger_delay)
        if use_env_exp:
            self.update_exp_time()
        return ret

    def is_color_camera(self):
        """Get the images type.
        This function is only updated when get_one_frame or get_projected_raw_images is called.

        Return: True or False
        """
        return is_color_image()

    def get_projected_raw_images(self, path):
        """Get the images

        Args:
            path: the saving path
        """
        if not self.hw_triger:
            print("not init with hw trigger, can not call this funtion")
            return
        if self.enable_hdr:
            self.get_projected_raw_images_hdr(path)
        else:
            self.get_projected_raw_images_normal(path)

    def get_projected_raw_images_normal(self, path):
        """Get the images

        Args:
            path: the saving path
        """
        capture_all_patterns(synced_projector=self._projector, camera_ids=self.camera_ids, path=path, scan_time=self.scan_time)

    def get_projected_raw_images_hdr(self, path):
        """Get the images

        Args:
            path: the saving path
        """
        if Config.use_high_speed_projector:
            self._projector.SetLEDCurrent((int)(Config.led_current * self._hdr_high_exp_rate))
        else:
            for cam in self.cameras:
                setExposureTime(cam, self.exp_time * self._hdr_high_exp_rate)
        capture_all_patterns(synced_projector=self._projector, camera_ids=self.camera_ids, path=path+"highexp_", scan_time=self.scan_time)
        import copy
        captured_images_left_high_exp = copy.deepcopy(self.captured_images_left)
        captured_images_right_high_exp = copy.deepcopy(self.captured_images_right)

        if Config.use_high_speed_projector:
            self._projector.SetLEDCurrent(Config.led_current)
        else:
            for cam in self.cameras:
                setExposureTime(cam, self.exp_time)
        capture_all_patterns(synced_projector=self._projector, camera_ids=self.camera_ids, path=path+"lowexp_", scan_time=self.scan_time)

        hdr_preprocess(self.captured_images_left, captured_images_left_high_exp, save_path=path+"_l")
        hdr_preprocess(self.captured_images_right, captured_images_right_high_exp, save_path=path+"_r")

    def close(self):
        """ Release camera
        """
        close_cameras(self.cameras, self.stream_sources)
        if self.hw_triger and (not Config.use_high_speed_projector):  # high_speed_projector turn off automaticly
            self._projector.turn_led(0)
            self._projector.close()


# test with "python -m HuaraySDKWrapper"
if __name__ == "__main__":
    hw_trigger = True
    # test for hw trigger
    camera = HuarayCamera(hw_trigger=hw_trigger, scale=0.5)
    if hw_trigger:
        camera.get_projected_raw_images(path="./temp/x3d_imgs/")
    else:
        img_1 = camera.get_one_frame(camera_id=camera.camera_ids[0])
        h, w = img_1.shape[:2]
        cv2.namedWindow('img1', 0)
        cv2.resizeWindow('img1', w // 2, h // 2)
        cv2.namedWindow('img2', 0)
        cv2.resizeWindow('img2', w // 2, h // 2)
        while (True):
            img_1 = camera.get_one_frame(camera_id=camera.camera_ids[0])
            cv2.imshow('img1', img_1)
            img_2 = camera.get_one_frame(camera_id=camera.camera_ids[1])
            cv2.imshow('img2', img_2)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    print("--------- capture_frames end ---------")
    camera.close()
