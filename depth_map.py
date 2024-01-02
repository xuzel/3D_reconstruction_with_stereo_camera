# # -*- coding: utf-8 -*-
# import pickle
# import cv2
# import numpy as np
#
# # 加载左右摄像头的校准数据
# with open("left_camera_calibration_data.pickle", "rb") as left_calibration_file:
#     left_calibration_data = pickle.load(left_calibration_file)
# mtx_l = left_calibration_data["camera_matrix"]
# dist_l = left_calibration_data["distortion_coefficients"]
# with open("right_camera_calibration_data.pickle", "rb") as right_calibration_file:
#     right_calibration_data = pickle.load(right_calibration_file)
# mtx_r = right_calibration_data["camera_matrix"]
# dist_r = right_calibration_data["distortion_coefficients"]
#
# # 加载双目摄像头的校准数据
# with open("Binocular_camera_calibration.pickle", "rb") as binocular_calibration_file:
#     binocular_camera_calibration_data = pickle.load(binocular_calibration_file)
# R = binocular_camera_calibration_data['R']
# T = binocular_camera_calibration_data['T']
#
# # 摄像头分辨率
# imageSize = (640, 480)
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, imageSize, R, T)
#
# # 计算校正映射
# map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, imageSize, cv2.CV_16SC2)
# map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, imageSize, cv2.CV_16SC2)
#
# # 设置视差匹配器
# stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=21, preFilterCap=16,
#                                uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
#                                disp12MaxDiff=1, P1=8 * 3 * 15 ** 2, P2=32 * 3 * 15 ** 2)
#
# # 创建WLS滤波器实例
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
# wls_filter.setLambda(8000)
# wls_filter.setSigmaColor(1.2)
#
# # 打开摄像头
# cv2.namedWindow("Left")
# cv2.namedWindow("Right")
# cv2.namedWindow("Disparity")
# camera = cv2.VideoCapture(0)
#
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# while True:
#     ret, frame = camera.read()
#     left_frame = frame[0:480, 0:640]
#     right_frame = frame[0:480, 640:1280]
#
#     gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
#     gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
#
#     # 应用立体校正
#     rectified_l = cv2.remap(gray_left, map1_l, map2_l, cv2.INTER_LINEAR)
#     rectified_r = cv2.remap(gray_right, map1_r, map2_r, cv2.INTER_LINEAR)
#
#     # 计算视差图
#     disparity = stereo.compute(rectified_l, rectified_r)
#
#     # 使用WLS滤波
#     filtered_disp = wls_filter.filter(disparity, rectified_l, None, rectified_r)
#     filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#     filtered_disp = np.uint8(filtered_disp)
#
#     # 显示图像
#     cv2.imshow("Left", left_frame)
#     cv2.imshow("Right", right_frame)
#     cv2.imshow("Disparity", filtered_disp)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# camera.release()
# cv2.destroyAllWindows()
#


import pickle
import cv2
import numpy as np

# 加载左右摄像头的校准数据
with open("left_camera_calibration_data.pickle", "rb") as left_calibration_file:
    left_calibration_data = pickle.load(left_calibration_file)
mtx_l = left_calibration_data["camera_matrix"]
dist_l = left_calibration_data["distortion_coefficients"]

with open("right_camera_calibration_data.pickle", "rb") as right_calibration_file:
    right_calibration_data = pickle.load(right_calibration_file)
mtx_r = right_calibration_data["camera_matrix"]
dist_r = right_calibration_data["distortion_coefficients"]

# 加载双目摄像头的校准数据
with open("Binocular_camera_calibration.pickle", "rb") as binocular_calibration_file:
    binocular_camera_calibration_data = pickle.load(binocular_calibration_file)
R = binocular_camera_calibration_data['R']
T = binocular_camera_calibration_data['T']

# 摄像头分辨率
imageSize = (640, 480)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, imageSize, R, T)

# 计算校正映射
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, imageSize, cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, imageSize, cv2.CV_16SC2)

# 设置视差匹配器
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=21, preFilterCap=16,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
                               disp12MaxDiff=1, P1=8 * 3 * 15 ** 2, P2=32 * 3 * 15 ** 2)

# 创建WLS滤波器实例
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.2)

# 打开摄像头
cv2.namedWindow("Left")
cv2.namedWindow("Right")
cv2.namedWindow("Disparity")
cv2.namedWindow("Depth")
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 从内参矩阵中提取焦距
focal_length = mtx_l[0, 0] # 假设焦距为内参矩阵中 (0,0) 位置的值
print(focal_length)

# 从外参矩阵提取基线距离
baseline = np.abs(T[0]) # 假设基线为 T 矩阵中第一个元素的绝对值

while True:
    ret, frame = camera.read()
    if not ret:
        break

    left_frame = frame[0:480, 0:640]
    right_frame = frame[0:480, 640:1280]

    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # 应用立体校正
    rectified_l = cv2.remap(gray_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(gray_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # 计算视差图
    disparity = stereo.compute(rectified_l, rectified_r)
    filtered_disp = wls_filter.filter(disparity, rectified_l, None, rectified_r)
    filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filtered_disp = np.uint8(filtered_disp)

    # 计算深度图
    with np.errstate(divide='ignore'): # 忽略除以零的警告
        depth = (focal_length * baseline) / (disparity.astype(np.float32) + 1e-6)
        depth[disparity < 1] = 0
    depth_display = cv2.normalize(src=depth, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    depth_display = np.uint8(depth_display)

    # 显示图像
    cv2.imshow("Left", left_frame)
    cv2.imshow("Right", right_frame)
    cv2.imshow("Disparity", filtered_disp)
    cv2.imshow("Depth", depth_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

