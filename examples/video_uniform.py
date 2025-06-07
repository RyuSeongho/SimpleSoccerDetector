# video_uniform.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

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

def color_distance_hsv(hsv1, hsv2_array):
    """HSV 색상 공간에서의 거리 계산"""
    h_diff = np.minimum(np.abs(hsv1[0] - hsv2_array[:, 0]),
                        180 - np.abs(hsv1[0] - hsv2_array[:, 0]))
    s_diff = np.abs(hsv1[1] - hsv2_array[:, 1])
    v_diff = np.abs(hsv1[2] - hsv2_array[:, 2])
    return np.sqrt((h_diff * 1.0)**2 + (s_diff * 1.0)**2 + (v_diff * 0.5)**2)


def process_video_uniform2(frame, boxes):
    """동일한 박스에 팀 색상만 적용"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 팀 색상 HSV 값 정의
    team_colors_hsv = [
        np.array([280, 1, 92]),  # 팀1: [330, 24, 30] 자주색 계열
        np.array([249, 13, 21])  # 팀2: [15, 200, 150] 검은색 계열
    ]

    team_colors_bgr = [
        (67, 58, 200),  # 팀1: 67, 58, 76
        (200, 46, 47)  # 팀2:
    ]

    # 잔디 영역 마스크 생성
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_not_green = cv2.bitwise_not(mask_green)

    # 결과 마스크 초기화
    result_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # 전달받은 boxes를 그대로 사용
    for box in boxes:
        x_min, y_min, x_max, y_max = box

        # ROI 추출
        roi_hsv = hsv[y_min:y_max, x_min:x_max]
        roi_mask_not_green = mask_not_green[y_min:y_max, x_min:x_max]

        if roi_hsv.size == 0:
            continue

        # 잔디가 아닌 영역에서 히스토그램 계산
        hist = cv2.calcHist([roi_hsv], [0], roi_mask_not_green, [180], [0, 180])
        hist_sum = np.sum(hist)

        if hist_sum == 0:
            continue

        # 가장 많은 색상(Hue) 찾기
        dominant_hue = np.argmax(hist)

        # 해당 색상 영역에서 평균 HSV 계산
        hue_tolerance = 15
        lower_hue = max(0, dominant_hue - hue_tolerance)
        upper_hue = min(179, dominant_hue + hue_tolerance)

        # 3채널 형태로 처리
        lower_bound = np.array([lower_hue, 0, 0])
        upper_bound = np.array([upper_hue, 255, 255])
        mask_hue = cv2.inRange(roi_hsv, lower_bound, upper_bound)
        mask_hue = cv2.bitwise_and(mask_hue, roi_mask_not_green)

        # 해당 색상 픽셀들의 평균 HSV 계산
        dominant_pixels = roi_hsv[mask_hue > 0]
        if dominant_pixels.size == 0:
            continue

        mean_hsv = np.mean(dominant_pixels, axis=0)

        # 가장 가까운 팀 색상 찾기
        distances = [color_distance_hsv(mean_hsv, team_hsv) for team_hsv in team_colors_hsv]
        closest_team_idx = np.argmin(distances)
        closest_team_bgr = team_colors_bgr[closest_team_idx]

        # *** 핵심: 새로운 윤곽선을 찾지 않고 원본 박스 그대로 사용 ***
        # 결과 마스크에 원본 박스 추가
        cv2.rectangle(result_mask, (x_min, y_min), (x_max, y_max), 255, -1)

        # 원본 프레임에 팀 색상으로 원본 박스 그리기
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), closest_team_bgr, 2)

        # 팀 라벨 추가
        label = f"Team{closest_team_idx + 1}"
        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, closest_team_bgr, 1)

    return result_mask

## =============================

import cv2
import numpy as np


def color_distance_hsv_vectorized(hsv1, hsv2_array):
    """벡터화된 HSV 색상 거리 계산 - 여러 팀 색상을 한번에 계산"""
    h_diff = np.minimum(np.abs(hsv1[0] - hsv2_array[:, 0]),
                        180 - np.abs(hsv1[0] - hsv2_array[:, 0]))
    s_diff = np.abs(hsv1[1] - hsv2_array[:, 1])
    v_diff = np.abs(hsv1[2] - hsv2_array[:, 2])
    return np.sqrt((h_diff * 2.0) ** 2 + s_diff ** 2 + v_diff ** 2)


def process_video_uniform2_imp(frame, boxes):
    """최적화된 동일한 박스에 팀 색상 적용"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 팀 색상 HSV 값 정의 (numpy array로 미리 변환)
    team_colors_hsv = np.array([
        [170, 150, 85],  # 팀1: 마룬/버건디 (H=170-178 범위)
        [15, 100, 80]    # 팀2: 보정된 HSV 값
    ], dtype=np.float32)

    team_colors_bgr = [
        (37, 24, 85),     # 팀1: 마룬/버건디 (BGR 순서)
        (88, 24, 2)       # 팀2
    ]

    # 잔디 영역 마스크 생성 (한 번만 수행)
    lower_green = np.array([35, 40, 40])  # 더 정확한 잔디 색상 범위
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_not_green = cv2.bitwise_not(mask_green)

    # 결과 마스크 초기화
    result_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # 박스별 처리
    for box in boxes:
        x_min, y_min, w, h = map(int, box)
        x_max, y_max = x_min + w, y_min + h

        # 유효성 검사
        if x_max <= x_min or y_max <= y_min:
            continue

        # ROI 경계 체크
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # ROI 추출
        roi_hsv = hsv[y_min:y_max, x_min:x_max]
        roi_mask_not_green = mask_not_green[y_min:y_max, x_min:x_max]

        if roi_hsv.size == 0:
            continue

        # 잔디가 아닌 픽셀들만 추출 (마스킹 최적화)
        valid_pixels = roi_hsv[roi_mask_not_green > 0]

        if valid_pixels.size == 0 or len(valid_pixels) < 10:  # 최소 픽셀 수 체크
            continue

        # 통계 기반 대표 색상 추출 (히스토그램 대신 직접 계산)
        # Hue 값에 대한 통계적 접근
        hue_values = valid_pixels[:, 0]
        sat_values = valid_pixels[:, 1]
        val_values = valid_pixels[:, 2]

        # # 채도가 낮은 픽셀 제거 (회색 계열 제거)
        # high_sat_mask = sat_values > 30
        # if np.sum(high_sat_mask) > 5:
        #     hue_values = hue_values[high_sat_mask]
        #     sat_values = sat_values[high_sat_mask]
        #     val_values = val_values[high_sat_mask]

        # 잔디 색상 픽셀 제거 (회색 계열 제거 대신)
        lower_grass = np.array([35, 40, 40])
        upper_grass = np.array([85, 255, 255])

        # 잔디 색상 범위에 해당하는 픽셀 마스크 생성
        is_grass = (hue_values >= lower_grass[0]) & (hue_values <= upper_grass[0]) & \
                   (sat_values >= lower_grass[1]) & (sat_values <= upper_grass[1]) & \
                   (val_values >= lower_grass[2]) & (val_values <= upper_grass[2])

        # 잔디가 아닌 픽셀만 선택
        non_grass_mask = ~is_grass

        if np.sum(non_grass_mask) > 5:
            hue_values = hue_values[non_grass_mask]
            sat_values = sat_values[non_grass_mask]
            val_values = val_values[non_grass_mask]

        # 대표 HSV 계산 (중앙값 사용으로 노이즈 감소)
        representative_hsv = np.array([
            np.median(hue_values),
            np.median(sat_values),
            np.median(val_values)
        ], dtype=np.float32)

        # 벡터화된 거리 계산으로 가장 가까운 팀 찾기
        distances = color_distance_hsv_vectorized(representative_hsv, team_colors_hsv)
        closest_team_idx = np.argmin(distances)
        closest_team_bgr = team_colors_bgr[closest_team_idx]

        # 신뢰도 체크 (거리 임계값)
        min_distance = distances[closest_team_idx]
        if min_distance > 300:  # 너무 먼 색상은 제외
            continue

        # 결과 마스크에 원본 박스 추가
        cv2.rectangle(result_mask, (x_min, y_min), (x_max, y_max), 255, -1)

        # 원본 프레임에 팀 색상으로 원본 박스 그리기
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), closest_team_bgr, 2)

        # 팀 라벨 추가 (개선된 신뢰도)
        distances_all = color_distance_hsv_vectorized(representative_hsv, team_colors_hsv)
        min_distance = distances_all[closest_team_idx]
        max_distance = np.max(distances_all)

        # 적응적 신뢰도 계산
        if max_distance > min_distance:
            relative_confidence = 100 * (max_distance - min_distance) / max_distance
        else:
            relative_confidence = 50  # 기본값

        # 절대 거리 기반 신뢰도와 결합
        absolute_confidence = max(0, 100 - min_distance / 1.5)
        confidence = (relative_confidence + absolute_confidence) / 2

        label = f"Team{closest_team_idx + 1} ({confidence:.0f}%)"
        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, closest_team_bgr, 1)

    return result_mask

def process_video_uniform_pxCount(frame, boxes):
    """픽셀 카운트 기반 팀 유니폼 분류 및 박스 그리기"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 잔디 마스크
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_not_green = cv2.bitwise_not(cv2.inRange(hsv, lower_green, upper_green))

    # 팀 HSV 범위
    team_hsv_ranges = [
        (np.array([160, 80, 40]), np.array([180, 255, 200])),  # Burgundy
        (np.array([0, 0, 0]), np.array([30, 50, 100]))  # Black
    ]
    team_bgr = [(37, 24, 85), (88, 24, 2)]

    for box in boxes:
        x, y, w, h = map(int, box)
        x2, y2 = x + w, y + h
        if x2 <= x or y2 <= y: continue

        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        roi_hsv = hsv[y:y2, x:x2]
        roi_ng = mask_not_green[y:y2, x:x2]  # single-channel (h,w)
        if roi_hsv.size == 0 or np.count_nonzero(roi_ng) < 20:
            continue

        # A: cv2.inRange 시점 변경 버전
        counts = []
        for (low, high) in team_hsv_ranges:
            mask_team = cv2.inRange(roi_hsv, low, high)
            mask_team = cv2.bitwise_and(mask_team, mask_team, mask=roi_ng)
            counts.append(int(np.count_nonzero(mask_team)))

        team_idx = int(np.argmax(counts))
        color = team_bgr[team_idx]

        # 박스+라벨
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.putText(frame, f"Team{team_idx + 1}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame

def process_video_uniform_pxCount_70(frame, boxes):
    """픽셀 카운트 기반 팀 유니폼 분류 및 박스 그리기 (중앙 70%만 사용)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 잔디 마스크
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_not_green = cv2.bitwise_not(cv2.inRange(hsv, lower_green, upper_green))

    # 팀 HSV 범위
    team_hsv_ranges = [
        (np.array([160,  80,  40]), np.array([180, 255, 200])),  # Burgundy
        (np.array([  0,   0,   0]), np.array([ 30,  50, 100]))   # Black
    ]
    team_bgr = [(37, 24, 85), (88, 24, 2)]

    for box in boxes:
        x, y, w, h = map(int, box)
        x2, y2 = x + w, y + h
        if x2 <= x or y2 <= y:
            continue

        # 1) 박스의 중앙 70% 영역만 남기기
        margin_x = int(w * 0.30)
        margin_y = int(h * 0.30)
        xi, yi = x + margin_x, y + margin_y
        x2i, y2i = x2 - margin_x, y2 - margin_y

        # 경계 체크
        xi, yi = max(0, xi), max(0, yi)
        x2i, y2i = min(frame.shape[1], x2i), min(frame.shape[0], y2i)
        if x2i <= xi or y2i <= yi:
            continue

        # 2) ROI HSV + 잔디 제거 마스크
        roi_hsv = hsv[yi:y2i, xi:x2i]
        roi_ng  = mask_not_green[yi:y2i, xi:x2i]
        if roi_hsv.size == 0 or np.count_nonzero(roi_ng) < 20:
            continue

        # 3) 각 팀 픽셀 수 세기
        counts = []
        for (low, high) in team_hsv_ranges:
            m = cv2.inRange(roi_hsv, low, high)
            m = cv2.bitwise_and(m, m, mask=roi_ng)
            counts.append(int(np.count_nonzero(m)))

        # 4) 픽셀 수 많은 팀 선택
        team_idx = int(np.argmax(counts))
        color = team_bgr[team_idx]

        # 5) 원래 박스에 그리기
        cv2.rectangle(frame, (xi, yi), (x2i, y2i), color, 2)
        cv2.putText(frame, f"Team{team_idx+1}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
