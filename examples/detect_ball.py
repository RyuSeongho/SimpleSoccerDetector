import cv2
import numpy as np

def detect_players(frame):
    # (이전과 동일)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35,40,40]), np.array([85,255,255])
    fg_mask = cv2.bitwise_not(cv2.inRange(hsv, lower_green, upper_green))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    h_img, w_img = frame.shape[:2]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        aspect = w/float(h) if h>0 else 0
        if 500 < area < 0.1*w_img*h_img and 0.3 < aspect < 1.2:
            bboxes.append((x,y,w,h))
    return bboxes, clean

def detect_ball(frame):
    """
    1) Canny로 에지 검출
    2) 컨투어별 원형도 계산 → 가장 둥근 후보 찾기
    3) 후보 영역 평균 HSV 확인 → 흰색 기준 통과 시 리턴
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 1) 에지 검출
    edges = cv2.Canny(gray, 50, 150)
    # 모폴로지로 끊긴 에지 연결 (선택)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2) 컨투어별 원형도 계산
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_circ = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 2000:  # 공 크기 범위 필터
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        # circularity가 1에 가까울수록 원형
        if circ > best_circ:
            best_circ = circ
            best = cnt

    # 3) 최종 후보의 색상 확인
    if best is not None and best_circ > 0.6:
        x,y,w,h = cv2.boundingRect(best)
        # 후보 영역 HSV 평균
        roi_hsv = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        mean_s = roi_hsv[:,:,1].mean()
        mean_v = roi_hsv[:,:,2].mean()
        # 흰색 기준: 채도 낮고 명도 높아야 함
        if mean_s < 30 and mean_v > 200:
            return [(x, y, w, h)], edges

    return [], edges

if __name__ == "__main__":
    frame = cv2.imread("./project_image/frame.png")
    if frame is None:
        raise IOError("이미지를 불러올 수 없습니다.")

    # 선수 검출
    player_boxes, player_mask = detect_players(frame)
    # 공 검출 (에지→원형도→색상)
    ball_boxes, ball_edges = detect_ball(frame)

    # 시각화
    vis = frame.copy()
    for (x,y,w,h) in player_boxes:
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 2)  # 빨간: 선수
    for (x,y,w,h) in ball_boxes:
        cv2.rectangle(vis, (x,y), (x+w, y+h), (255,0,0), 2)  # 파란: 공

    cv2.imshow("Original", frame)
    cv2.imshow("Player Mask", player_mask)
    cv2.imshow("Ball Edges", ball_edges)
    cv2.imshow("Detection Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
