import cv2
import numpy as np
import argparse
from examples.color_picker import RealTimeColorDisplay, integrate_realtime_colors

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

def draw_bounding_boxes(frame, mask, grass_color, color=(0, 255, 0), min_area=10, tolerance=30):
    """마스크에서 윤곽선을 찾아 바운딩 박스를 그립니다."""
    # 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 결과 프레임 복사
    result = frame.copy()
    
    # 각 윤곽선에 대해 바운딩 박스 그리기
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:  # 작은 노이즈 제거
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 바운딩 박스가 필드 위에 있는지 확인
            if is_on_field(frame, x, y, w, h, grass_color, tolerance):
                # 바운딩 박스 그리기
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    
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

def process_video(video_path, team1_color_rgb, team2_color_rgb):
    # RGB 입력을 BGR로 변환 (한 번만 변환)
    team1_color_bgr = [team1_color_rgb[2], team1_color_rgb[1], team1_color_rgb[0]]  # R,G,B -> B,G,R
    team2_color_bgr = [team2_color_rgb[2], team2_color_rgb[1], team2_color_rgb[0]]  # R,G,B -> B,G,R
    
    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # 첫 프레임에서 잔디 색상 추출
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return

    # 잔디 색상 분석 (BGR 색상 공간)
    color_display = RealTimeColorDisplay()
    all_mask = np.ones_like(first_frame, dtype=np.uint8) * 255
    dominant_colors = integrate_realtime_colors(first_frame, all_mask, color_display, color_space="bgr")  
    print("Grass color (BGR):", dominant_colors)

    # 윈도우 생성
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Team 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Team 2", cv2.WINDOW_NORMAL)
    

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 크기 조정
            frame = cv2.resize(frame, (640, 360))
            
            # 잔디 색상 마스크 생성 (BGR)
            # dominant_colors는 이미 BGR 순서이므로 순서대로 사용
            lower_green, upper_green = bgr_range(dominant_colors[0], dominant_colors[1], dominant_colors[2], tolerance=60)
            mask_green = cv2.inRange(frame, lower_green, upper_green)
            mask_not_green = cv2.bitwise_not(mask_green)
            
            # 각 팀의 유니폼 마스크 생성 (BGR 색상 사용)
            mask_team1 = create_uniform_mask(frame, team1_color_bgr)
            mask_team2 = create_uniform_mask(frame, team2_color_bgr)
            
            # 잔디가 아니고 각 팀의 유니폼 색상인 부분만 마스킹
            mask_team1_final = cv2.bitwise_and(mask_not_green, mask_team1)
            mask_team2_final = cv2.bitwise_and(mask_not_green, mask_team2)
            
            # 노이즈 제거를 위한 모폴로지 연산
            kernel = np.ones((5,5), np.uint8)
            mask_team1_final = cv2.morphologyEx(mask_team1_final, cv2.MORPH_CLOSE, kernel)
            mask_team2_final = cv2.morphologyEx(mask_team2_final, cv2.MORPH_CLOSE, kernel)
            
            # 바운딩 박스 그리기 (잔디 색상 정보 포함)
            result_team1 = draw_bounding_boxes(frame, mask_team1_final, dominant_colors, color=(255, 255, 255))  # 흰색
            result_team2 = draw_bounding_boxes(frame, mask_team2_final, dominant_colors, color=(0, 0, 0))  # 검은색
            
            # 결과 표시
            cv2.imshow("Original", frame)
            cv2.imshow("Team 1", result_team1)
            cv2.imshow("Team 2", result_team2)
            # cv2.imshow("mask_team1_final", mask_team1_final)
            # cv2.imshow("mask_team2_final", mask_team2_final)
            # cv2.imshow("mask_not_green", mask_not_green)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # s 키로 일시 정지
                cv2.waitKey(0)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='축구 선수 추적 프로그램')
    parser.add_argument('video_path', type=str, help='처리할 비디오 파일 경로')
    parser.add_argument('--team1-color', type=int, nargs=3, required=True,
                      help='팀1 유니폼 색상 (RGB 형식, 예: 255 0 0)')
    parser.add_argument('--team2-color', type=int, nargs=3, required=True,
                      help='팀2 유니폼 색상 (RGB 형식, 예: 0 0 255)')
    
    args = parser.parse_args()
    
    process_video(args.video_path, args.team1_color, args.team2_color)
