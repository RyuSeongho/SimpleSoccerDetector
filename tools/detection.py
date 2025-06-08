import cv2
import numpy as np

def get_bounding_boxes(frame, mask, grass_color, min_area=10, tolerance=30):
    """마스크에서 바운딩 박스들을 찾아 반환합니다."""
    # 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    
    # 각 윤곽선에 대해 바운딩 박스 추출
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:  # 작은 노이즈 제거
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 바운딩 박스가 필드 위에 있는지 확인
            if is_on_field(frame, x, y, w, h, grass_color, tolerance):
                bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes

def draw_boxes_on_frame(frame, bounding_boxes, color=(0, 255, 0)):
    """프레임에 바운딩 박스들을 그려서 반환합니다."""
    result = frame.copy()
    
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    
    return result

def draw_bounding_boxes(frame, mask, grass_color, color=(0, 255, 0), min_area=10, tolerance=30):
    """마스크에서 윤곽선을 찾아 바운딩 박스를 그립니다. (기존 호환성을 위해 유지)"""
    # 바운딩 박스 추출
    bounding_boxes = get_bounding_boxes(frame, mask, grass_color, min_area, tolerance)
    
    # 바운딩 박스 그리기
    result = draw_boxes_on_frame(frame, bounding_boxes, color)
    
    return result

def is_on_field(frame, x, y, w, h, grass_color, tolerance):
    """바운딩 박스가 필드 위에 있는지 확인"""
    # 바운딩 박스의 좌우 픽셀 샘플링
    sample_points = []
    height, width = frame.shape[:2]
    
    # 좌측 픽셀 샘플링
    left_x = max(0, x - 10)
    for i in range(3):  # 3개의 점 샘플링
        sample_y = y + (h * (i + 1) // 4)
        if 0 <= sample_y < height:
            sample_points.append(frame[sample_y, left_x])
    
    # 우측 픽셀 샘플링
    right_x = min(width - 1, x + w + 10)
    for i in range(3):  # 3개의 점 샘플링
        sample_y = y + (h * (i + 1) // 4)
        if 0 <= sample_y < height:
            sample_points.append(frame[sample_y, right_x])
    
    # 샘플링된 픽셀들의 평균 색상 계산
    if not sample_points:
        return False
    
    avg_color = np.mean(sample_points, axis=0)
    
    # 잔디 색상과의 유사도 계산 (BGR 순서)
    color_diff = np.abs(avg_color - grass_color)
    return np.all(color_diff <= tolerance) 