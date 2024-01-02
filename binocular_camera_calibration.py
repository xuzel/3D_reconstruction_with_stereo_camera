import pickle

import numpy as np
import cv2
import glob

# 设置棋盘格参数
CHECKERBOARD = (8, 11)
square_size = 30

# 设置亚像素角点查找的标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备3D点（棋盘格角点的世界坐标）
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 数组存储所有图像的角点
objpoints = []  # 3D points in real world space
imgpoints_l = []  # 2D points in image plane, left camera
imgpoints_r = []  # 2D points in image plane, right camera

# 读取左右摄像头的图像路径
images_left = glob.glob('data/camera/left_*.png')
images_right = glob.glob('data/camera/right_*.png')
images_right.sort()
images_left.sort()

for img_left, img_right in zip(images_left, images_right):
    print(img_left)
    print(img_right)
    print('\n')
    img_l = cv2.imread(img_left)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(img_right)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 在左摄像头图像中找到角点
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    # 在右摄像头图像中找到角点
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp)

        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(corners2_l)

        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_r.append(corners2_r)


with open("left_camera_calibration_data.pickle", "rb") as file:
    left_calibration_data = pickle.load(file)
left_camera_matrix = left_calibration_data["camera_matrix"]
left_distortion_coefficients = left_calibration_data["distortion_coefficients"]

with open("right_camera_calibration_data.pickle", "rb") as file:
    right_calibration_data = pickle.load(file)
right_camera_matrix = right_calibration_data["camera_matrix"]
right_distortion_coefficients = right_calibration_data["distortion_coefficients"]


# 单目标定结果
# 注意替换为实际的内参和畸变系数
mtx_l, dist_l = left_camera_matrix, left_distortion_coefficients  # 左摄像头的内参和畸变系数
mtx_r, dist_r = right_camera_matrix, right_distortion_coefficients  # 右摄像头的内参和畸变系数

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], criteria=criteria, flags=flags)


calibration_result = {
    'R': R,
    'T': T,
    'E': E,
    'F': F
}
with open("Binocular_camera_calibration.pickle", "wb") as file:
    pickle.dump(calibration_result, file)

print("旋转矩阵:\n", R)
print("平移向量:\n", T)
print("本征矩阵 E:\n", E)
print("基础矩阵 F:\n", F)
