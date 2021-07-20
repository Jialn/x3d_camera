import numpy as np
import cv2

class StereoRectify():
    """ Class for Stereo undistor and Rectify, with scale
    """

    def __init__(self,
                 scale,
                 cali_file):
        """
        Args:
            scale(float or None): scale of image
            cali_file: path to calibar yml file. e.g., "./temp/binanry_test/cali_stereo.yml"
        """
        if scale is None:
            self.scale = 1.0
        else:
            self.scale = scale
        # read yml data
        infile = cv2.FileStorage(cali_file, cv2.FILE_STORAGE_READ)
        self.cameraMatrixL = infile.getNode("K1").mat()
        self.cameraMatrixR = infile.getNode("K2").mat()
        self.distCoeffsL = infile.getNode("D1").mat()
        self.distCoeffsR = infile.getNode("D2").mat()
        self.R = infile.getNode("R").mat()
        self.T = infile.getNode("T").mat()
        self.E = infile.getNode("E").mat()
        self.F = infile.getNode("F").mat()
        infile.release()

        self.remap_x_left_scaled=None
        self.remap_x_right_scaled=None
        self.remap_y_left_scaled=None
        self.remap_y_right_scaled=None
        self.rectified_camera_kd=None

    def rectify_image(self, src_image, left=True, interpolation=cv2.INTER_LINEAR):
        """
        Args:
            left: the image is left or right, left by default
            interpolation: cv2.INTER_LINEAR for rgb/gray image, cv2.INTER_NEAREST for depth
        """
        if self.remap_x_left_scaled is None:
            h, w = src_image.shape[:2]
            org_h, org_w = round(h / self.scale), round(w / self.scale)
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                self.cameraMatrixL, self.distCoeffsL, self.cameraMatrixR, self.distCoeffsR, (org_w, org_h), self.R, self.T, flags=0, alpha=-1) # cv2.CALIB_ZERO_DISPARITY
            self.R1, self.R2 = R1, R2
            self.P1, self.P2 = P1, P2
            # undistort images map
            remap_x_left, remap_y_left = cv2.initUndistortRectifyMap(self.cameraMatrixL, self.distCoeffsL, R1, P1, (org_w, org_h), cv2.CV_32FC1)
            remap_x_right, remap_y_right = cv2.initUndistortRectifyMap(self.cameraMatrixR, self.distCoeffsR, R2, P2, (org_w, org_h), cv2.CV_32FC1)

            def scale_map(img_map):
                return cv2.resize(img_map*self.scale, (w, h), interpolation=interpolation)

            self.remap_x_left_scaled, self.remap_y_left_scaled = scale_map(remap_x_left), scale_map(remap_y_left)
            self.remap_x_right_scaled, self.remap_y_right_scaled = scale_map(remap_x_right), scale_map(remap_y_right)
            self.remap_x_left_scaled = np.clip(self.remap_x_left_scaled, 0.0, w-1)
            self.remap_x_right_scaled = np.clip(self.remap_x_right_scaled, 0.0, w-1)
            self.remap_y_left_scaled = np.clip(self.remap_y_left_scaled, 0.0, h-1)
            self.remap_y_right_scaled = np.clip(self.remap_y_right_scaled, 0.0, h-1)
            self.rectified_camera_kd_l = self.P1[:,:3]*self.scale
            self.rectified_camera_kd_l[2,2] = 1.0
            self.rectified_camera_kd_r = self.P2[:,:3]*self.scale
            self.rectified_camera_kd_r[2,2] = 1.0
        if left:
            rectified_img = cv2.remap(src_image, self.remap_x_left_scaled, self.remap_y_left_scaled, interpolation)
            self.rectified_camera_kd = self.rectified_camera_kd_l
        else:
            rectified_img = cv2.remap(src_image, self.remap_x_right_scaled, self.remap_y_right_scaled, interpolation)
            self.rectified_camera_kd = self.rectified_camera_kd_r
        return rectified_img


# test with "python stereo_rectify.py", press q to exit
if __name__ == "__main__":
    rectifier = StereoRectify(scale=0.5, cali_file='./test_calib.yml')
    img = cv2.imread('./test.bmp', cv2.IMREAD_UNCHANGED)
    img_rectified = rectifier.rectify_image(img)
    cv2.imshow('img', img)
    cv2.imshow('img_rectified', img_rectified)

    while (True):
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    exit()