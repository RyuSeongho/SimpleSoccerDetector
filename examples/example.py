import cv2
import numpy as np

def detect_players(frame):
    """
    입력: BGR 컬러 이미지 (frame)
    출력: 선수 후보들의 bounding boxes 리스트 [(x,y,w,h), ...]
    """
    # 1) ROI 마스크 (필요시 경기장 선 검출 → 생략하고 전체 프레임 사용)
    roi = frame.copy()

    # 2) 녹색 필드 제외: HSV로 변환 후 '녹색' 범위 마스크 생성
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 대략적인 잔디 ‘녹색’ 범위 (튜닝 필요)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # 선수 후보 마스크 = 녹색이 아닌 픽셀
    fg_mask = cv2.bitwise_not(green_mask)

    # 3) 모폴로지 정제
    # 노이즈 제거 (opening) + 홀 채우기 (closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4) 컨투어 검출 & 필터링
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    h_img, w_img = frame.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # 면적 및 종횡비 필터: 너무 작거나 너무 세로/가로로 길쭉한 영역 제외
        if area < 500 or area > 0.1 * w_img * h_img:
            continue
        aspect = w / float(h)
        if aspect < 0.3 or aspect > 1.2:
            continue
        bboxes.append((x, y, w, h))

    return bboxes, clean

if __name__ == "__main__":
    # 테스트 이미지 로드
    frame = cv2.imread("./project_image/frame.png")
    cv2.imshow("Original", frame)
    boxes, mask = detect_players(frame)

    # 결과 시각화
    vis = frame.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 윈도우 출력
    cv2.imshow("Original", frame)
    cv2.imshow("Foreground Mask", mask)
    cv2.imshow("Detected Players", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()