import numpy as np
import pcl
import pcl.pcl_visualization

from utils import *


class PCLCloudViewer(object):
    def __init__(self, point_cloud=None):
        self.point_cloud = point_cloud
        self.cloud = pcl.PointCloud_PointXYZRGBA()
        self.viewer = pcl.pcl_visualization.CloudViewing()

    def add_3dpoints(self, points_3d, image):
        self.point_cloud = DepthColor2PointXYZRGBA(points_3d, image)

    def show(self):
        self.cloud.from_array(self.point_cloud)
        self.viewer.ShowColorACloud(self.cloud)
        v = not (self.viewer.WasStopped())
        return v


def DepthColor2PointXYZRGBA(points_3d, image):
    height, width = points_3d.shape[0:2]
    size = height * width
    points_ = points_3d.reshape(height * width, 3)
    colors_ = image.reshape(height * width, 3).astype(np.int64)
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)
    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]
    idx1 = np.where(Z <= 0)
    idx2 = np.where(Z > 15000)
    idx3 = np.where(X > 10000)
    idx4 = np.where(X < -10000)
    idx5 = np.where(Y > 10000)
    idx6 = np.where(Y < -10000)
    idx = np.hstack((idx1[0], idx2[0], idx3[0], idx4[0], idx5[0], idx6[0]))
    dst_pointcloud = np.delete(pointcloud, idx, 0)
    return dst_pointcloud


def get_rectify_image(imgL, imgR, camera_config, color=cv2.COLOR_BGR2GRAY):
    left_map_x, left_map_y = camera_config["left_map_x"], camera_config["left_map_y"]
    right_map_x, right_map_y = camera_config["right_map_x"], camera_config["right_map_y"]
    rectifiedL = cv2.remap(imgL, left_map_x, left_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
    rectifiedR = cv2.remap(imgR, right_map_x, right_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)

    return rectifiedL, rectifiedR


def get_filter_disparity(imgL, imgR):
    blockSize = 3
    paramL = {"minDisparity": 0,
              "numDisparities": 5 * 16,
              "blockSize": blockSize,
              "P1": 8 * 3 * blockSize,
              "P2": 32 * 3 * blockSize,
              "disp12MaxDiff": 12,
              "uniquenessRatio": 10,
              "speckleWindowSize": 50,
              "speckleRange": 32,
              "preFilterCap": 63,
              "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }
    matcherL = cv2.StereoSGBM_create(**paramL)
    dispL = matcherL.compute(imgL, imgR)
    dispL = np.int16(dispL)
    matcherR = cv2.ximgproc.createRightMatcher(matcherL)
    dispR = matcherR.compute(imgR, imgL)
    dispR = np.int16(dispR)
    lmbda = 80000
    sigma = 1.3
    filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcherL)
    filter.setLambda(lmbda)
    filter.setSigmaColor(sigma)
    dispL = filter.filter(dispL, imgL, None, dispR)
    dispL = np.int16(dispL)
    dispL[dispL < 0] = 0
    dispL = dispL.astype(np.float32) / 16.
    return dispL


def get_3d_points(disparity, Q, scale=1.0):
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    points_3d = points_3d * scale
    points_3d = np.asarray(points_3d, dtype=np.float32)
    return points_3d


def get_visual_depth(depth, clip_max=6000):
    depth = np.clip(depth, 0, clip_max)
    depth = cv2.normalize(src=depth, dst=depth, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    depth = np.uint8(depth)
    depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth_colormap


def show_3d_cloud_for_pcl(pcl_viewer, frame, points_3d):
    pcl_viewer.add_3dpoints(points_3d / 1000, frame)
    pcl_viewer.show()


def get_visual_disparity(disp, clip_max=6000):
    disp = np.clip(disp, 0, clip_max)
    disp = np.uint8(disp)
    return disp


def addMouseCallback(winname, param, callbackFunc=None, info="%"):
    '''
     添加点击事件
    :param winname:
    :param param:
    :param callbackFunc:
    :return:
    '''
    cv2.namedWindow(winname)

    def default_callbackFunc(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("image(x,y)=({},{}),".format(x, y) + info.format(param[y][x]))

    if callbackFunc is None:
        callbackFunc = default_callbackFunc
    cv2.setMouseCallback(winname, callbackFunc, param)


def show_2dimage(frameL, frameR, points_3d, dispL, waitKey=0):
    x, y, depth = cv2.split(points_3d)
    xyz_coord = points_3d
    depth_colormap = get_visual_depth(depth)
    dispL_colormap = get_visual_disparity(dispL)
    addMouseCallback("left", xyz_coord, info="world coords=(x,y,depth)={}mm")
    addMouseCallback("right", xyz_coord, info="world coords=(x,y,depth)={}mm")
    addMouseCallback("disparity-color", xyz_coord, info="world coords=(x,y,depth)={}mm")
    addMouseCallback("depth-color", xyz_coord, info="world coords=(x,y,depth)={}mm")
    result = {"frameL": frameL, "frameR": frameR, "disparity": dispL_colormap, "depth": depth_colormap}
    cv2.imshow('left', frameL)
    cv2.imshow('right', frameR)
    cv2.imshow('disparity-color', dispL_colormap)
    cv2.imshow('depth-color', depth_colormap)


def show(camera_id: int, frame_info: typing.Dict[str, int], config_dict: typing.Dict[str, str]) -> None:
    pcl_view = PCLCloudViewer()
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_info["width"])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_info["height"])
    while True:
        ret, frame = camera.read()
        left_frame = frame[:, :int(frame_info["width"] / 2)]
        right_frame = frame[:, int(frame_info["width"] / 2):]
        left_frame, right_frame = get_rectify_image(left_frame, right_frame, config_dict)
        grayL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        disL = get_filter_disparity(grayL, grayR)
        point = get_3d_points(disL, config_dict["Q"])
        show_3d_cloud_for_pcl(pcl_view, left_frame, point)
        show_2dimage(left_frame, right_frame, point, disL)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


def main():
    config = read_config("config.ini")
    show(
        int(config["camera"]["id"]),
        {"width": int(config["camera"]["width"]), "height": int(config["camera"]["height"])},
        get_stereo_coefficients(config["path"]["binocular_camera_calibration_path"])
    )


if __name__ == '__main__':
    main()
