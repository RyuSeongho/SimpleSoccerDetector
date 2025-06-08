import cv2
import numpy as np
import argparse
from examples.color_picker import RealTimeColorDisplay, integrate_realtime_colors
from tools.color_utils import bgr_range, create_uniform_mask
from tools.detection import draw_bounding_boxes

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
