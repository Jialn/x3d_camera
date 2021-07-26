# Copyright (c) 2020. All Rights Reserved.
# Created on 2021-2-21
# Autor: Jiangtao <jiangtao.li@gmail.com>
"""
Description: 
Provide the global configuration for this repo
"""

class Config():
    """ Helper class for configuration
    """
    # General Parameters
    save_pattern_to_disk = False
    re_capture_color_image_using_env_exposure = False or save_pattern_to_disk
    save_point_cloud = True
    use_depth_as_cloud_color = False
    cali_file = "confs/cali-robotarm-2021-07-22.yml"

    # Alg Parameters
    pose_detector_type = "mask2d"  # "pvn3d", "mask2d", or "ir_marker"
    use_phsft = True  # must be true, only support gray + phase shift for now
    save_all_patterns = False and (save_pattern_to_disk)
    # use_phsft = False
    pattern_start_index = 0
    pattern_end_index = 38  # the pattern number in projector, used for images capture
    if not save_all_patterns:
        if use_phsft:
            pattern_start_index = 24
            phsft_thres = 2
            pattern_end_index = 38  # the pattern number in projector, used for images capture
        else:
            pattern_start_index = 0
            pattern_end_index = 24

    # Camera Parameters
    # hardware related
    camera_ids = [0, 1]  # The id of camera. For Dual cam, it could be [0,1] or [1,2]. Note that 0 could be the usb webcam if a laptop is used. 
    switch_left_right = False
    is_color_camera = True
    use_high_speed_projector = False  # note high_speed prj only support phase shift patterns
    is_white_projector = True  # blue projector or white projector
    default_testing_scale = 1.0  # for x3d_camera test program
    # only for color camera
    do_imagearray_reshape_in_camera_call_back = True  # build imagearray in camera callback or after all images are captured
    do_demosac_for_color_camera = False  # if not, demosac will be done in the later progress by GPU
    use_blue_channel_for_color_camera = is_color_camera and (not is_white_projector)  # color convert to gray or use blue channel only. If use white projector, set to False
    fake_color_using_blue_light = False and (not is_color_camera) and (not is_white_projector)
    # for low speed 30fps projector
    if not use_high_speed_projector:
        hw_triger_delay = 20  # 10
        scan_time = [0, pattern_start_index, 10] # [interval_time, start_index, display_time]
        import sys
        if sys.platform == 'win32': projector_port = "COM3"
        else: projector_port = "/dev/ttyUSB0"
    # for hi-speed projector
    if use_high_speed_projector:
        hw_triger_delay = 5  # 10
        scan_time = [5, pattern_start_index, 15] # [interval_time, start_index, display_time]
        pat_exp_time = scan_time[2] * 1000  # used to setup pattern exposure time for uniform exp pattern mode.
        frame_period = (scan_time[0] + scan_time[2]) * 1000  # used to setup frame period of projector. If frame pattern count is 1, it represent time duration of adjacent pattern.
        led_current = 51  # PWM duty cycle strength. 0->no current, 255->max current
    # expo and white balance
    enable_auto_exposure = False  # automaticly find a proper exp time
    auto_expo_range_for_env_light = [90, 130]  # the expected mean range for env light when auto_exposing
    if use_phsft:
        auto_expo_range_for_pattern = [30, 50]  # should have low exp for patterns
    else:
        auto_expo_range_for_pattern = [30, 50]  # could be a litlle bit higher for for graycode pattern
    exp_time = 20000.0  # exposure time for 3D projector, us. ignore this if enable_auto_exposure
    exp_time_env = 220000.0  # exposure time for environmet images without projector openning. ignore this if enable_auto_exposure
    gamma = 0.75  # 1.0, 0.8; if < 1.0, level the black zone up, better for dark zone. but bring lots of flying points
    white_balance_red_ratio = 1.9
    white_balance_blue_ratio = 1.9
    # hdr
    enable_hdr = False
    hdr_high_exp_rate = 2.0 # led_current will be set to led_current*hdr_high_exp_rate for high speed projector
                            # high exposure time = hdr_high_exp_rate * exp_time for low spd projector
    if enable_hdr:
        if led_current * hdr_high_exp_rate > 255:
            led_current = 255 // hdr_high_exp_rate
        high_led_current = 50 + (round)((led_current - 50) * hdr_high_exp_rate)
        phsft_thres = 2

    def __init__(self):
        """Init.
        """
        pass


if __name__ == "__main__":
    pass
