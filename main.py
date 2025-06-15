import cv2
import numpy as np
import argparse
import os
import json
from datetime import datetime
from tools.color_picker import RealTimeColorDisplay, integrate_realtime_colors
from tools.color_utils import bgr_range, create_uniform_mask
from tools.detection import get_bounding_boxes, draw_boxes_on_frame
from tools.player_tracker import PlayerTrackerManager
from tools.ball_detection import detect_ball, draw_ball_detection, filter_ball_by_field_position
from tools.ball_tracker import BallTrackerManager
import base64
import sys
import struct
from pathlib import Path

def process_video(video_path, team1_color_rgb, team2_color_rgb, ball_color_rgb=None, tracker_debug_mode=False, output_path=None):
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

    # 비디오 정보 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 경로 설정
    if output_path is None:
        # 현재 디렉토리에 output 폴더 생성
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 입력 비디오 파일명을 기반으로 출력 파일명 생성
        input_filename = os.path.basename(video_path)
        output_filename = f"tracked_{input_filename}"
        output_path = os.path.join(output_dir, output_filename)

    # JSON 출력 파일 설정
    json_output_dir = os.path.dirname(output_path)
    os.makedirs(json_output_dir, exist_ok=True)
    json_filename = f"tracking_data.json"
    json_output_path = os.path.join(json_output_dir, json_filename)

    # 비디오 작성자 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))

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
    print(f"출력 파일: {output_path}")
    print(f"JSON 파일: {json_output_path}")

    # 윈도우 생성
    cv2.namedWindow("Soccer Tracking", cv2.WINDOW_NORMAL)

    # 첫 프레임 플래그 (tracker_debug_mode에서만 사용)
    is_first_frame = True
    frame_count = 0

    # JSON 데이터를 저장할 리스트
    tracking_data = []

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
            
            # 공 감지 (선수 bbox와 관중석 필터링 포함)
            all_player_bboxes = team1_detected_bboxes + team2_detected_bboxes
            detected_ball_bboxes = detect_ball(frame, mask_green, ball_color_bgr, player_bboxes=all_player_bboxes)
            detected_ball_bboxes = filter_ball_by_field_position(detected_ball_bboxes, frame.shape)
            
            # 공 추적 업데이트
            ball_tracker_manager.update_ball_tracking(detected_ball_bboxes, tracker_manager, frame, frame_count)
            
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
            
            # 현재 프레임의 추적 데이터 수집
            frame_data = {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,  # 초 단위 타임스탬프
                "players": {
                    "team1": [],
                    "team2": []
                },
                "ball": None
            }

            # 팀1 선수 데이터 수집
            for bbox in team1_tracked_bboxes:
                x, y, w, h = bbox
                player_data = {
                    "position": {
                        "x": int(x + w/2),  # 중심점 x 좌표
                        "y": int(y + h/2)   # 중심점 y 좌표
                    },
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                }
                frame_data["players"]["team1"].append(player_data)

            # 팀2 선수 데이터 수집
            for bbox in team2_tracked_bboxes:
                x, y, w, h = bbox
                player_data = {
                    "position": {
                        "x": int(x + w/2),  # 중심점 x 좌표
                        "y": int(y + h/2)   # 중심점 y 좌표
                    },
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                }
                frame_data["players"]["team2"].append(player_data)

            # 공 데이터 수집
            ball_info = ball_tracker_manager.get_ball_info()
            if ball_info['active']:
                ball_bbox = ball_tracker_manager.get_ball_bbox()
                if ball_bbox:
                    x, y, w, h = ball_bbox
                    frame_data["ball"] = {
                        "position": {
                            "x": int(x + w/2),  # 중심점 x 좌표
                            "y": int(y + h/2)   # 중심점 y 좌표
                        },
                        "bbox": {
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h)
                        },
                        "possession": ball_info['possession']
                    }

            # 현재 프레임 데이터를 전체 데이터에 추가
            tracking_data.append(frame_data)
            
            # 결과 그리기
            # 추적된 bbox로 그리기
            result_frame = frame.copy()
            result_frame = draw_boxes_on_frame(result_frame, team1_tracked_bboxes, color=(0, 255, 255))  # 노란색
            result_frame = draw_boxes_on_frame(result_frame, team2_tracked_bboxes, color=(255, 0, 255))  # 마젠타색
            
            # 공 감지 결과 그리기
            result_frame = draw_ball_detection(result_frame, detected_ball_bboxes, color=(0, 255, 0))  # 초록색
            
            # 추적 정보 표시
            team1_count, team2_count = tracker_manager.get_tracker_count()
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames} | Team1: {team1_count}, Team2: {team2_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 추적 모드 표시
            mode_display = "only tracking" if tracker_debug_mode else "tracking and registering"
            cv2.putText(result_frame, f"Mode: {mode_display}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 공 정보 표시
            ball_info = ball_tracker_manager.get_ball_info()
            if ball_info['active']:
                possession_info = ball_info['possession']
                cv2.putText(result_frame, f"Ball Active | Frames Lost: {ball_info['frames_lost']}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if possession_info['in_possession']:
                    cv2.putText(result_frame, f"Possession: Team {possession_info['team']} (Player {possession_info['player_id']})", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(result_frame, f"Free Ball | Closest: Team {possession_info['team']} ({possession_info['distance']:.1f}px)", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(result_frame, f"Ball Not Tracked", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 프레임 데이터 전송 (Electron으로)
            frame_bytes = result_frame.tobytes()
            frame_size = len(frame_bytes)
            
            # 프레임 크기와 데이터 전송
            sys.stdout.buffer.write(struct.pack('<IHH', frame_size, frame_width, frame_height))
            sys.stdout.buffer.write(frame_bytes)
            sys.stdout.buffer.flush()
            
            # 결과 프레임 저장
            out.write(result_frame)
            
            # 진행률 표시
            if frame_count % 30 == 0:  # 30프레임마다 진행률 출력
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})", file=sys.stderr)
            
            # 프레임 카운트 증가
            frame_count += 1

    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        print(f"Video saved to: {output_path}", file=sys.stderr)
        
        # JSON 파일 저장
        try:
            with open(json_output_path, 'w') as f:
                json.dump({
                    "metadata": {
                        "video_path": video_path,
                        "fps": fps,
                        "total_frames": total_frames,
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                        "team1_color": team1_color_rgb,
                        "team2_color": team2_color_rgb,
                        "ball_color": ball_color_rgb if ball_color_rgb else [255, 255, 255],
                        "tracker_debug_mode": tracker_debug_mode
                    },
                    "frames": tracking_data
                }, f, indent=2)
            print(f"Tracking data saved to: {json_output_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving JSON file: {e}", file=sys.stderr)

def process_frame(frame, team1_color, team2_color):
    # 여기에 프레임 처리 로직 추가
    # 예: 팀 색상 기반으로 선수 추적 등
    return frame

def send_frame(frame):
    """프레임을 바이너리로 전송"""
    # 프레임 크기 정보 전송
    height, width = frame.shape[:2]
    size = struct.pack('!II', width, height)
    sys.stdout.buffer.write(size)
    sys.stdout.buffer.flush()
    
    # 프레임 데이터 전송
    sys.stdout.buffer.write(frame.tobytes())
    sys.stdout.buffer.flush()

def main():
    parser = argparse.ArgumentParser(description='Soccer Player Tracking')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--team1-color', nargs=3, type=int, help='Team 1 color (RGB)')
    parser.add_argument('--team2-color', nargs=3, type=int, help='Team 2 color (RGB)')
    args = parser.parse_args()

    # 팀 색상 설정
    team1_color = tuple(args.team1_color) if args.team1_color else (255, 0, 0)
    team2_color = tuple(args.team2_color) if args.team2_color else (0, 0, 255)

    # 출력 경로 설정
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'tracked_video.mp4'

    # process_video 함수 호출하여 실제 축구 추적 수행
    process_video(
        video_path=args.video_path,
        team1_color_rgb=team1_color,
        team2_color_rgb=team2_color,
        output_path=str(output_path)
    )

if __name__ == '__main__':
    main()
