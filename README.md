# X3D Camera
The main repository for X3D Camera.


## Installation

### 已经验证过的平台和环境
- 操作系统：Windows10, Ubuntu18.04, Ubuntu20.04, Nvidia NX的自带操作系统; 注意更换操作系统需要重新编译pyd文件，默认的pyd文件为Windows10下编译的版本
- Python版本：Python3.8.6, Python 3.8.10;
- CUDA版本：CUDA 11.2, CUDA11.4; 注意不同架构的显卡需要重新编译cubin文件，默认的文件为RTX30系显卡编译的版本


### Instrcutions
- 安装Python3.8, 建议 Python3.8.6 or Python3.8.10，安装时注意选择把python添加进path
- 下载安装Huarry Camera SDK：http://download.huaraytech.com/pub/sdk/Ver2.2.5/
- 下载安装VS2019: https://visualstudio.microsoft.com/zh-hans/vs/community/
    安装VS2019完毕后需要添加"cl.exe"的环境变量，示例: `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64`

- 下载安装CUDA11.4：https://developer.nvidia.com/cuda-downloads
- 通过Pip安装packages
  ```
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  pip install numpy opencv-python pyserial hidapi open3d pycuda
  pip install pycuda open3d
  ```

    



## Test

- Test 3D camera
  ```
  python -m struli_camera_main.struli_camera_main
  ```

## Progress of Calibration

1. 开启光机，开启光机打光开关
2. 开启相机预览 ``python stereo_calib.py "temp/stereocali/" ``会弹出left和right窗后预览，用手挡一下左相机，检查左右是否插反，如果反了就调换一下USB口
3. 调整光机投影视野范围，工作区间尽可能居中；调整左右相机视野尽可能全覆盖光机视野范围。对广角相机，左右视差不可太大，远景相同的物体在左右图中像素位置应尽量靠近
4. 放置标定板工作区间中心位置，调整摄像头对焦使得标定板尽可能清晰。调整光机镜头对焦使条纹中部和远部尽可能清晰
5. 抓图. 按e键切换抓图模式，相机将会自动增加曝光。标定时曝光要向右曝光，如需修改，相机相关参数设置在config.py，曝光参数exp_time = ***. 标定抓图会自动改为这个值的二倍。
按S保存图片，存够9张后按ESC退出抓图，执行标定
6. 检查标定结果：
``python struli_camera_main.py`` 抓取3D图，然后打开结果文件夹：
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
