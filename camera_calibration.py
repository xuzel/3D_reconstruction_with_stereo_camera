import numpy as np
import cv2
import glob
import pickle

# 棋盘格设置
CHECKERBOARD = (8, 11)  # 棋盘格的角点数量
calibration_paths = glob.glob('data/camera/left_*.png')  # 棋盘格图像的路径

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 世界坐标系中的点
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 用于存储3D点和2D点
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

for image_file in calibration_paths:

    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        # 通过亚像素精确化角点位置
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 绘制并显示角点
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


calibration_result = {
    "camera_matrix": mtx,
    "distortion_coefficients": dist,
    "rotation_vectors": rvecs,
    "translation_vectors": tvecs
}

# 使用pickle保存数据
with open("left_camera_calibration_data.pickle", "wb") as file:
    pickle.dump(calibration_result, file)


print("相机矩阵 :\n", mtx)
print("\n畸变参数 :\n", dist)
print("\n旋转向量 :\n", rvecs)
print("\n平移向量 :\n", tvecs)
