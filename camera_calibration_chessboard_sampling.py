from utils import *


def detect_chessboard(image, width: int, height: int,
                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image = cv2.drawChessboardCorners(image, (width, height), corners2, ret)
    return image


def obtain_image(camera_id: int, img_save_path: str, frame_info: typing.Dict[str, int],
                 chessboard: typing.Dict[str, int]) -> None:
    os.makedirs(img_save_path, exist_ok=True)
    cv2.namedWindow("left")
    cv2.namedWindow("right")
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_info["width"])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_info["height"])
    counter = 0
    while True:
        ret, frame = camera.read()
        left_frame = frame[:, :int(frame_info["width"] / 2)]
        right_frame = frame[:, int(frame_info["width"] / 2):]
        left_frame_with_chessboard = detect_chessboard(left_frame.copy(), chessboard["width"], chessboard["height"])
        right_frame_with_chessboard = detect_chessboard(right_frame.copy(), chessboard["width"], chessboard["height"])
        cv2.imshow("left", left_frame_with_chessboard)
        cv2.imshow("right", right_frame_with_chessboard)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(os.path.join(img_save_path, f"left_{counter}.png"), left_frame)
            cv2.imwrite(os.path.join(img_save_path, f"right_{counter}.png"), right_frame)
            print(f"save img {counter + 1}")
            counter += 1
    camera.release()
    cv2.destroyAllWindows()


def main():
    config_data = read_config("config.ini")
    camera_id = int(config_data["camera"]["id"])
    save_path = config_data["path"]["chessboard_path"]
    frame_info = {
        "width": int(config_data["camera"]["width"]),
        "height": int(config_data["camera"]["height"])
    }
    chessboard_info = {
        "width": int(config_data["chessboard"]["width"]),
        "height": int(config_data["chessboard"]["height"])
    }

    obtain_image(camera_id, save_path, frame_info, chessboard_info)


if __name__ == '__main__':
    main()
