# X3D Camera
The main repository for X3D Camera.


## Installation

### 已经验证过的平台和环境
- 操作系统：Windows10, Ubuntu18.04, Ubuntu20.04, Nvidia Xavier NX的自带操作系统;

- Python版本：Python3.8.6, Python 3.8.10;

  注意更换操作系统和python版本需要重新编译*.pyd文件，默认的pyd文件为Windows10下Python 3.8.6，Cython 0.29.24编译的版本。

- CUDA版本：CUDA 11.2, CUDA11.4; 

   注意不同架构的显卡需要重新编译cuda_core.cubin文件，默认的文件为RTX30系显卡，CUDA11.4编译的版本


### Instrcutions
- 安装Python3.8, 建议 Python3.8.6，安装时注意选择把python添加进path
- 下载安装Huarry Camera SDK：http://download.huaraytech.com/pub/sdk/Ver2.2.5/
- 下载安装VS2019 (Linux跳过此步): https://visualstudio.microsoft.com/zh-hans/vs/community/
    安装时需要确认“使用C++的桌面开发”是勾选状态
    安装VS2019完毕后需要添加"cl.exe"的环境变量，示例: `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64`

- 下载安装CUDA11.4：https://developer.nvidia.com/cuda-downloads
- 通过Pip安装packages
  ```
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  pip install numpy opencv-python pyserial hidapi open3d pycuda
  ```

## Test

- Test 3D camera
  ```
  python struli_camera_main.py
  ```

## Progress of Calibration

1. 开启光机开关

2. 当前目录下新建文件夹"images/stereocali/", 开启相机预览 ``python stereo_calib.py "images/stereocali/" ``会弹出left和right窗后预览，用手挡一下左相机，检查左右是否插反，如果反了就在Config中设置一下switch_left_right

3. 调整光机投影视野范围，工作区间尽可能居中；调整左右相机视野尽可能全覆盖光机视野范围。对广角相机，左右视差不可太大，工作区域的物体在左右图中像素位置应尽量靠近

4. 对焦和光圈调整
  - 放置标定板工作区间中心位置，调整摄像头对焦使得标定板尽可能清晰
  - 调整光机镜头对焦使条纹中部和远部尽可能清晰
  - 调整光圈使得曝光合适且景深合适。

5. 抓图
  - 按e键切换抓图模式，相机将会自动增加一倍曝光（因为标定时曝光要向右曝光）。如需修改曝光参数，相机相关参数设置在config.py，曝光参数exp_time = ***。
  - 按S保存图片，每拍一张图更换一下标定板位置或高度。不同位置和高度的图片存够9张后按ESC退出抓图，自动执行标定。
  - 如果更换标定板，需要修改stereo_calib.py中corners_vertical，corners_horizontal，corner_distance为匹配的参数。

6. 等待标定执行完毕，检查标定结果，reprojection error一般情况下小于0.1像素：

7. 测试：运行``python struli_camera_main.py`` 按空格抓取3D图，Esc退出可以查看结果文件夹：点云应该横平竖直，现实中的平面在点云成像中各个角度观察都无明显弯曲。


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
