import cv2
import numpy as np
import argparse
from examples.color_picker import RealTimeColorDisplay, integrate_realtime_colors
from tools.color_utils import bgr_range, create_uniform_mask
from tools.detection import draw_bounding_boxes, get_bounding_boxes, draw_boxes_on_frame
from tools.player_tracker import PlayerTrackerManager
from tools.ball_detection import detect_ball, draw_ball_detection, filter_ball_by_field_position
from tools.ball_tracker import BallTrackerManager

def process_video(video_path, team1_color_rgb, team2_color_rgb, ball_color_rgb=None, tracker_debug_mode=False):
    # RGB 입력을 BGR로 변환 (한 번만 변환)
    team1_color_bgr = [team1_color_rgb[2], team1_color_rgb[1], team1_color_rgb[0]]  # R,G,B -> B,G,R
    team2_color_bgr = [team2_color_rgb[2], team2_color_rgb[1], team2_color_rgb[0]]  # R,G,B -> B,G,R
    
    # 공 색상도 BGR로 변환 (기본값: 흰색)
    if ball_color_rgb is None:
        ball_color_bgr = [255, 255, 255]  # 흰색 (BGR)
    else:
        ball_color_bgr = [ball_color_rgb[2], ball_color_rgb[1], ball_color_rgb[0]]  # R,G,B -> B,G,R
    
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

    # 프레임 크기 조정 후 PlayerTrackerManager 초기화
    first_frame = cv2.resize(first_frame, (640, 360))
    frame_height, frame_width = first_frame.shape[:2]
    
    # 잔디 색상 분석 (BGR 색상 공간)
    color_display = RealTimeColorDisplay()
    all_mask = np.ones_like(first_frame, dtype=np.uint8) * 255
    dominant_colors = integrate_realtime_colors(first_frame, all_mask, color_display, color_space="bgr")  
    print("Grass color (BGR):", dominant_colors)
    print("Ball color (BGR):", ball_color_bgr)
    
    # PlayerTrackerManager 초기화 (잔디색 전달)
    tracker_manager = PlayerTrackerManager(frame_width, frame_height, dominant_colors)
    
    # BallTrackerManager 초기화
    ball_tracker_manager = BallTrackerManager(frame_width, frame_height, dominant_colors)

    # 추적 모드 출력
    mode_text = "추적 전용 모드 (첫 프레임만 등록)" if tracker_debug_mode else "일반 모드 (매 프레임 등록/업데이트)"
    print(f"실행 모드: {mode_text}")

    # 윈도우 생성
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Team 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Team 2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tracked Players", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Ball Detection", cv2.WINDOW_NORMAL)

    # 첫 프레임 플래그 (tracker_debug_mode에서만 사용)
    is_first_frame = True
    frame_count = 0

    try:
        # 비디오를 처음부터 다시 읽기 위해 재설정
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
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
            
            # 각 팀의 바운딩 박스 감지
            team1_detected_bboxes = get_bounding_boxes(frame, mask_team1_final, dominant_colors)
            team2_detected_bboxes = get_bounding_boxes(frame, mask_team2_final, dominant_colors)
            
            # 공 감지
            detected_ball_bboxes = detect_ball(frame, mask_green, ball_color_bgr, debug=True)
            detected_ball_bboxes = filter_ball_by_field_position(detected_ball_bboxes, frame.shape)
            
            # 공 추적 업데이트
            ball_tracker_manager.update_ball_tracking(detected_ball_bboxes, tracker_manager, frame, frame_count)
            
            # 추적된 공 바운딩 박스 가져오기
            tracked_ball_bbox = ball_tracker_manager.get_ball_bbox()
            ball_bboxes = [tracked_ball_bbox] if tracked_ball_bbox is not None else []
            
            # 추적 모드에 따른 처리
            if tracker_debug_mode:
                # 추적 전용 모드: 첫 프레임에서만 등록, 나머지는 추적만
                if is_first_frame:
                    tracker_manager.initialize_trackers(team1_detected_bboxes, team2_detected_bboxes)
                    is_first_frame = False
                else:
                    tracker_manager.update_trackers_only(team1_detected_bboxes, team2_detected_bboxes, frame)
            else:
                # 일반 모드: 매 프레임 등록/업데이트
                tracker_manager.update_trackers(team1_detected_bboxes, team2_detected_bboxes, frame)
            
            # 추적된 바운딩 박스들 가져오기
            team1_tracked_bboxes, team2_tracked_bboxes = tracker_manager.get_all_bboxes()
            
            # 결과 그리기
            # 감지된 bbox로 그리기 (기존 방식)
            result_team1 = draw_boxes_on_frame(frame, team1_detected_bboxes, color=(255, 255, 255))  # 흰색
            result_team2 = draw_boxes_on_frame(frame, team2_detected_bboxes, color=(0, 0, 0))  # 검은색
            
            # 추적된 bbox로 그리기 (새로운 방식)
            result_tracked = frame.copy()
            result_tracked = draw_boxes_on_frame(result_tracked, team1_tracked_bboxes, color=(0, 255, 255))  # 노란색
            result_tracked = draw_boxes_on_frame(result_tracked, team2_tracked_bboxes, color=(255, 0, 255))  # 마젠타색
            
            # 공 감지 결과 그리기
            result_ball = draw_ball_detection(frame, ball_bboxes, color=(0, 255, 0))  # 초록색
            
            # 추적 정보 표시
            team1_count, team2_count = tracker_manager.get_tracker_count()
            cv2.putText(result_tracked, f"Frame: {frame_count} | Team1: {team1_count}, Team2: {team2_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 추적 모드 표시
            mode_display = "only tracking" if tracker_debug_mode else "tracking and registering"
            cv2.putText(result_tracked, f"Mode: {mode_display}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 공 정보 표시
            ball_info = ball_tracker_manager.get_ball_info()
            if ball_info['active']:
                possession_info = ball_info['possession']
                cv2.putText(result_ball, f"Ball Active | Frames Lost: {ball_info['frames_lost']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if possession_info['in_possession']:
                    cv2.putText(result_ball, f"Possession: Team {possession_info['team']} (Player {possession_info['player_id']})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(result_ball, f"Free Ball | Closest: Team {possession_info['team']} ({possession_info['distance']:.1f}px)", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(result_ball, f"Ball Not Tracked", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 결과 표시
            cv2.imshow("Original", frame)
            cv2.imshow("Team 1", result_team1)
            cv2.imshow("Team 2", result_team2)
            cv2.imshow("Tracked Players", result_tracked)
            cv2.imshow("Ball Detection", result_ball)
            
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
    parser.add_argument('--ball-color', type=int, nargs=3,
                      help='공 색상 (RGB 형식, 예: 255 255 255)')
    parser.add_argument('--tracker_debug', action='store_true',
                      help='추적 전용 모드 활성화 (첫 프레임에서만 플레이어 등록)')
    
    args = parser.parse_args()
    
    process_video(args.video_path, args.team1_color, args.team2_color, args.ball_color, args.tracker_debug)
