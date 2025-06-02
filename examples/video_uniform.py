# video_uniform.py
import cv2
import numpy as np

def process_video_uniform(frame):
    # 프레임 리사이즈
    frame = cv2.resize(frame, (640, 360))

    # HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 초록색 마스크 (잔디)
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 파란 유니폼
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 흰색 유니폼
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 피부색
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 150, 255])
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

    # 선수 마스크 (공 제외)
    mask_players = cv2.bitwise_or(mask_blue, mask_white)
    mask_players = cv2.bitwise_or(mask_players, mask_skin)
    final_mask = cv2.bitwise_and(mask_players, cv2.bitwise_not(mask_green))

    # ROI 제한
    roi = cv2.bitwise_and(frame, frame, mask=final_mask)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge Detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)

    median_val = np.median(sobel)
    lower = int(max(0, 0.3 * median_val))
    upper = int(min(255, 3.0 * median_val))
    sobel[sobel < lower] = 0
    sobel[sobel > upper] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = cv2.morphologyEx(np.uint8(sobel), cv2.MORPH_CLOSE, kernel)

    final_edges = cv2.bitwise_and(result, final_mask)
    return final_edges
