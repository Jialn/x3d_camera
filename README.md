# X3D Camera
The main repository for X3D Camera.

## Overview
This repository includes

- python wrapper for structure light depth camera

    1. drivers and python interface of huarry industrial camera and projector
    2. structure light depth generation interface


## Installation
Tested on ubuntu18.04 and python3.6

### Huarry Camera SDK
- Install huarry camera SDK if our own structure light depth camera is used

  http://download.huaraytech.com/pub/sdk/Ver2.2.3/ or
  http://download.huaraytech.com/pub/sdk/Ver2.2.5/

### Others

  ```
  pip install opencv-python pyserial hidapi open3d

  ```
  Also need to put struc_light side by side with the folder of this repo.

## Test

- Test 3D camera
  ```
  python -m x3d_camera.x3d_camera
  ```

## Progress of Calibration

1. 开启光机，开启光机打光开关
2. 开启相机预览 ``python stereo_calib.py "temp/stereocali/" ``会弹出left和right窗后预览，用手挡一下左相机，检查左右是否插反，如果反了就调换一下USB口
3. 调整光机投影视野范围，工作区间尽可能居中；调整左右相机视野尽可能全覆盖光机视野范围。对广角相机，左右视差不可太大，远景相同的物体在左右图中像素位置应尽量靠近
4. 放置标定板工作区间中心位置，调整摄像头对焦使得标定板尽可能清晰。调整光机镜头对焦使条纹中部和远部尽可能清晰
5. 抓图. 按e键切换抓图模式，相机将会自动增加曝光。标定时曝光要向右曝光，如需修改，相机相关参数设置在config.py，曝光参数exp_time = ***. 标定抓图会自动改为这个值的二倍。
按S保存图片，存够9张后按ESC退出抓图，执行标定
6. 检查标定结果：
``python x3d_camera.py`` 抓取3D图，然后打开结果文件夹：
``nautilus ./images``
 用 meshlab 打开 pc_cl_bi.ply，点云应该横平竖直，现实中的平面在点云成像中各个角度观察都无明显弯曲


## Contributing Workflow
1. Install code style tools
```bash
pip install pre-commit cpplint pydocstyle
sudo apt install clang-format
```

2. Make local changes
```bash
git co -b PR_change_name origin/master
```

  Make change to your code

3. Run pre-commit before commit

```bash
pre-commit run --files ./*
```
  This will format the code and do checking. Then commit if passed.

4. Make pull request:
```bash
git push origin PR_change_name
```
