import cv2
import numpy as np
from typing import List, Tuple, Optional
import math
from .color_utils import create_uniform_mask, bgr_range

def calculate_compactness(contour) -> float:
    """윤곽선의 compactness 계산 (1에 가까울수록 원형)"""
    area = cv2.contourArea(contour)
    if area == 0:
        return 0
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    
    # Compactness = 4π * Area / Perimeter^2
    # 완전한 원의 경우 1이 됨
    compactness = (4 * math.pi * area) / (perimeter * perimeter)
    return compactness

def is_ball_candidate_edge(contour, min_perimeter: int = 2, max_perimeter: int = 80, 
                          min_compactness: float = 0.65) -> bool:
    """Edge 윤곽선이 공 후보인지 판단 (Edge Detection 전용)"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Edge의 경우 perimeter 기준으로 필터링 (area는 0에 가까울 수 있음)
    if perimeter < min_perimeter or perimeter > max_perimeter:
        return False
    
    # Area가 0이면 compactness 계산 불가
    if area < 1:
        return False
    
    # Compactness 필터링
    compactness = calculate_compactness(contour)
    if compactness < min_compactness:
        return False
    
    return True

def is_ball_candidate(contour, min_area: int = 4, max_area: int = 64, 
                     min_compactness: float = 0.75) -> bool:
    """윤곽선이 공 후보인지 판단"""
    area = cv2.contourArea(contour)
    
    # 크기 필터링
    if area < min_area or area > max_area:
        return False
    
    # Compactness 필터링
    compactness = calculate_compactness(contour)
    if compactness < min_compactness:
        return False
    
    return True


def is_near_player(ball_center: Tuple[int, int], player_bboxes: List[Tuple[int, int, int, int]], 
                   distance_threshold: int = 20) -> bool:
    """공 중심점이 선수 bbox 안에 있거나 너무 가까운지 확인"""
    if not player_bboxes:
        return False
    
    ball_x, ball_y = ball_center
    
    for bbox in player_bboxes:
        px, py, pw, ph = bbox
        
        # bbox 안에 있는지 확인
        if px <= ball_x <= px + pw and py <= ball_y <= py + ph:
            return True
        
        # bbox 근처에 있는지 확인 (확장된 영역)
        expanded_x1 = px - distance_threshold
        expanded_y1 = py - distance_threshold
        expanded_x2 = px + pw + distance_threshold
        expanded_y2 = py + ph + distance_threshold
        
        if expanded_x1 <= ball_x <= expanded_x2 and expanded_y1 <= ball_y <= expanded_y2:
            return True
    
    return False


def is_on_field(ball_center: Tuple[int, int], grass_mask: np.ndarray, 
                surrounding_radius: int = 15) -> bool:
    """공 위치 주변이 잔디(필드)인지 확인"""
    if grass_mask is None:
        return True  # 잔디 마스크가 없으면 모든 위치 허용
    
    ball_x, ball_y = ball_center
    height, width = grass_mask.shape
    
    # 경계 체크
    if ball_x < 0 or ball_x >= width or ball_y < 0 or ball_y >= height:
        return False
    
    # 주변 영역 정의
    x1 = max(0, ball_x - surrounding_radius)
    y1 = max(0, ball_y - surrounding_radius)
    x2 = min(width, ball_x + surrounding_radius + 1)
    y2 = min(height, ball_y + surrounding_radius + 1)
    
    # 주변 영역에서 잔디 픽셀 비율 계산
    surrounding_area = grass_mask[y1:y2, x1:x2]
    if surrounding_area.size == 0:
        return False
    
    grass_ratio = np.sum(surrounding_area > 0) / surrounding_area.size
    
    # 주변의 60% 이상이 잔디이면 필드로 간주
    return grass_ratio > 0.6


def detect_ball(frame: np.ndarray, grass_mask: np.ndarray = None, 
                ball_color: Tuple[int, int, int] = None, debug: bool = False,
                player_bboxes: List[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
    """프레임에서 축구공 감지 (SimpleBlobDetector 기반으로 고립된 흰색 원 감지)"""
    
    ball_candidates = []
    
    # 1단계: 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ball_mask = create_uniform_mask(frame, ball_color)
    ball_mask = cv2.GaussianBlur(ball_mask, (3, 3), 0)

    gray = cv2.Canny(gray, 50, 150)
    cv2.imshow("Canny", gray)

    # grass_mask가 제공된 경우 사용, 없으면 전체 영역 사용
    if grass_mask is not None:
        not_grass_mask = cv2.bitwise_not(grass_mask)
    else:
        not_grass_mask = np.ones_like(gray) * 255  # 전체 영역 사용

    gray = cv2.bitwise_or(ball_mask, not_grass_mask)

    
    # 2단계: SimpleBlobDetector 설정
    params = cv2.SimpleBlobDetector_Params()
    
    # 색상 기준 (밝은 blob 찾기 - 흰색 공)
    params.filterByColor = True
    params.blobColor = 255  # 흰색 blob 찾기
    
    # 면적 기준
    params.filterByArea = True
    params.minArea = 4       # 최소 면적
    params.maxArea = 40     # 최대 면적
    
    # 원형성 기준 (compactness와 유사)
    params.filterByCircularity = True
    params.minCircularity = 0.8  # 원형성 최소값
    
    # 볼록성 기준
    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    # 관성 비율 기준 
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    # 3단계: Blob Detector 생성 및 실행
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    
    if debug:
        # Blob 감지 결과 시각화
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), 
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Detected Blobs", im_with_keypoints)
        print(f"Found {len(keypoints)} blob candidates")
    
    # 4단계: Keypoint를 바운딩 박스로 변환
    for keypoint in keypoints:
        x, y = keypoint.pt
        size = keypoint.size
        
        # 바운딩 박스 계산
        radius = size / 2
        bbox_x = int(x - radius)
        bbox_y = int(y - radius)
        bbox_size = int(size)
        
        # 경계 체크
        if (bbox_x >= 0 and bbox_y >= 0 and 
            bbox_x + bbox_size < frame.shape[1] and bbox_y + bbox_size < frame.shape[0]):
            
            # 선수 근처인지 확인
            ball_center = (int(x), int(y))
            if player_bboxes and is_near_player(ball_center, player_bboxes):
                if debug:
                    print(f"Ball candidate rejected (near player): center=({int(x)},{int(y)})")
                continue
            
            # 필드 위에 있는지 확인 (관중석 제거)
            if not is_on_field(ball_center, grass_mask):
                if debug:
                    print(f"Ball candidate rejected (not on field): center=({int(x)},{int(y)})")
                continue
            
            ball_candidates.append((bbox_x, bbox_y, bbox_size, bbox_size))

            if debug:
                print(f"Ball candidate accepted: center=({int(x)},{int(y)}), size={size:.1f}, bbox=({bbox_x},{bbox_y},{bbox_size},{bbox_size})")
    
    if debug:
        print(f"Final ball candidates: {len(ball_candidates)}")
        
    
    return ball_candidates

def draw_ball_detection(frame: np.ndarray, ball_bboxes: List[Tuple[int, int, int, int]], 
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """공 감지 결과를 프레임에 그리기 (SimpleBlobDetector 기반)"""
    result = frame.copy()
    
    for bbox in ball_bboxes:
        x, y, w, h = bbox
        
        # 바운딩 박스 그리기
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # 중심점과 원형 표시
        center_x = x + w // 2
        center_y = y + h // 2
        radius = w // 2
        
        # 중심점 표시
        cv2.circle(result, (center_x, center_y), 3, color, -1)
        
        # 원 둘레 표시
        cv2.circle(result, (center_x, center_y), radius, color, 2)
        
        # "BALL" 텍스트 표시
        cv2.putText(result, f"BALL", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result

def filter_ball_by_field_position(ball_bboxes: List[Tuple[int, int, int, int]], 
                                 frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """필드 위치를 고려하여 공 후보 필터링"""
    if not ball_bboxes:
        return ball_bboxes
    
    height, width = frame_shape[:2]
    
    # 화면 가장자리 마진 (공이 화면 가장자리에 있을 가능성 낮음)
    margin_x = width // 10
    margin_y = height // 10
    
    filtered_balls = []
    
    for bbox in ball_bboxes:
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 화면 중앙 영역에 있는 공만 유효
        if (margin_x < center_x < width - margin_x and 
            margin_y < center_y < height - margin_y):
            filtered_balls.append(bbox)
    
    return filtered_balls 