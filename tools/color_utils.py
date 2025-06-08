import cv2
import numpy as np

def bgr_range(b, g, r, tolerance=50):
    """BGR 색상 범위를 생성합니다."""
    # 모든 값을 int로 안전하게 변환
    b, g, r = int(b), int(g), int(r)
    tolerance = int(tolerance)
    
    # BGR 각 채널의 범위 계산 (0-255 범위 내에서)
    lower_b = max(0, b - tolerance)
    upper_b = min(255, b + tolerance)
    lower_g = max(0, g - tolerance)
    upper_g = min(255, g + tolerance)
    lower_r = max(0, r - tolerance)
    upper_r = min(255, r + tolerance)
    
    # 명시적으로 uint8 타입 지정 (BGR 순서)
    lower_color = np.array([lower_b, lower_g, lower_r], dtype=np.uint8)
    upper_color = np.array([upper_b, upper_g, upper_r], dtype=np.uint8)
    
    return lower_color, upper_color

def create_uniform_mask(frame, team_color_bgr):
    """팀의 유니폼 색상 범위에 해당하는 마스크 생성"""
    # BGR 색상으로 마스크 생성
    lower_color, upper_color = bgr_range(team_color_bgr[0], team_color_bgr[1], team_color_bgr[2])
    
    # 마스크 생성
    mask_color = cv2.inRange(frame, lower_color, upper_color)
    return mask_color 