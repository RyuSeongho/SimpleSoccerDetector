import numpy as np
from typing import List, Tuple, Optional
import math

class BallTracker:
    """축구공을 추적하는 클래스 (하나의 공만 추적)"""
    
    def __init__(self, grass_color: Tuple[int, int, int] = (0, 128, 0)):
        self.position = None  # (x, y) - 공의 중심점
        self.radius = 0  # 공의 반지름
        self.frames_lost_score = 0  # 연속으로 공을 찾지 못한 점수
        self.dx = 0  # x 방향 속도
        self.dy = 0  # y 방향 속도
        self.grass_color = grass_color  # 잔디 색상 (BGR)
        self.closest_player_id = None  # 가장 가까운 플레이어 ID
        self.closest_player_team = None  # 가장 가까운 플레이어의 팀
        self.closest_player_distance = float('inf')  # 가장 가까운 플레이어와의 거리
        
        # 추적 설정
        self.max_lost_frames = 45
        self.max_assignment_distance = 30  # 공 할당 최대 거리
        self.ball_possession_distance = 18  # 플레이어가 공을 점유했다고 판단하는 거리
        
        # 속도 제한 설정
        self.max_velocity = 50  # 한 프레임당 최대 이동 거리 (픽셀)
        self.max_velocity_change = 30  # 한 프레임당 최대 속도 변화량
        
        # 내부 상태
        self.is_active = False  # 공이 활성 상태인지
        self.previous_position = None  # 이전 위치
        
    def limit_velocity(self, new_dx: int, new_dy: int) -> Tuple[int, int]:
        """속도에 제한을 가하는 함수"""
        # 1단계: 속도 절대값 제한
        velocity_magnitude = math.sqrt(new_dx*new_dx + new_dy*new_dy)
        
        if velocity_magnitude > self.max_velocity:
            # 방향은 유지하되 크기를 제한
            scale = self.max_velocity / velocity_magnitude
            limited_dx = int(new_dx * scale)
            limited_dy = int(new_dy * scale)
        else:
            limited_dx = new_dx
            limited_dy = new_dy
        
        # 2단계: 속도 변화량 제한 (이전 속도가 있는 경우)
        if hasattr(self, 'dx') and hasattr(self, 'dy'):
            dx_change = limited_dx - self.dx
            dy_change = limited_dy - self.dy
            change_magnitude = math.sqrt(dx_change*dx_change + dy_change*dy_change)
            
            if change_magnitude > self.max_velocity_change:
                # 변화량을 제한
                change_scale = self.max_velocity_change / change_magnitude
                final_dx = self.dx + int(dx_change * change_scale)
                final_dy = self.dy + int(dy_change * change_scale)
            else:
                final_dx = limited_dx
                final_dy = limited_dy
        else:
            final_dx = limited_dx
            final_dy = limited_dy
        
        return final_dx, final_dy
    
    def register_ball(self, ball_bbox: Tuple[int, int, int, int]):
        """새로운 공 등록"""
        x, y, w, h = ball_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2
        
        self.previous_position = self.position
        self.position = (center_x, center_y)
        self.radius = radius
        self.frames_lost_score = 0
        self.is_active = True
        
        # 속도 계산 (이전 위치가 있는 경우)
        if self.previous_position is not None:
            raw_dx = center_x - self.previous_position[0]
            raw_dy = center_y - self.previous_position[1]
            
            # 속도 제한 적용
            self.dx, self.dy = self.limit_velocity(raw_dx, raw_dy)
            
            # 속도가 제한되었는지 확인
            if abs(raw_dx) > self.max_velocity or abs(raw_dy) > self.max_velocity:
                print(f"    속도제한: 원본({raw_dx},{raw_dy}) → 제한({self.dx},{self.dy})")
        else:
            self.dx = 0
            self.dy = 0
    
    def update_ball(self, ball_bbox: Tuple[int, int, int, int]):
        """기존 공 위치 업데이트"""
        self.register_ball(ball_bbox)  # 동일한 로직 사용
    
    def predict_next_position(self) -> Tuple[int, int]:
        """속도를 기반으로 다음 위치 예측"""
        if self.position is None:
            return None
        
        x, y = self.position
        pred_x = x + self.dx
        pred_y = y + self.dy
        
        return (int(pred_x), int(pred_y))
    
    def calculate_distance_to_bbox(self, ball_bbox: Tuple[int, int, int, int]) -> float:
        """현재 위치에서 주어진 ball bbox까지의 거리 계산"""
        if self.position is None:
            return float('inf')
        
        bbox_center_x = ball_bbox[0] + ball_bbox[2] // 2
        bbox_center_y = ball_bbox[1] + ball_bbox[3] // 2
        
        dx = self.position[0] - bbox_center_x
        dy = self.position[1] - bbox_center_y
        return math.sqrt(dx*dx + dy*dy)
    
    def calculate_predicted_distance_to_bbox(self, ball_bbox: Tuple[int, int, int, int]) -> float:
        """예측 위치에서 주어진 ball bbox까지의 거리 계산"""
        predicted_pos = self.predict_next_position()
        if predicted_pos is None:
            return float('inf')
        
        bbox_center_x = ball_bbox[0] + ball_bbox[2] // 2
        bbox_center_y = ball_bbox[1] + ball_bbox[3] // 2
        
        dx = predicted_pos[0] - bbox_center_x
        dy = predicted_pos[1] - bbox_center_y
        return math.sqrt(dx*dx + dy*dy)
    
    def find_closest_player(self, player_tracker_manager):
        """가장 가까운 플레이어 찾기"""
        if self.position is None:
            self.closest_player_id = None
            self.closest_player_team = None
            self.closest_player_distance = float('inf')
            return
        
        min_distance = float('inf')
        closest_tracker = None
        closest_team = None
        
        # 모든 플레이어 추적자들을 확인
        for tracker in player_tracker_manager.trackers:
            player_center = tracker.get_center()
            distance = math.sqrt((self.position[0] - player_center[0])**2 + 
                               (self.position[1] - player_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_tracker = tracker
                closest_team = tracker.team_id
        
        if closest_tracker is not None:
            self.closest_player_id = closest_tracker.tracker_id
            self.closest_player_team = closest_team
            self.closest_player_distance = min_distance
        else:
            self.closest_player_id = None
            self.closest_player_team = None
            self.closest_player_distance = float('inf')
    
    def is_ball_in_player_possession(self) -> bool:
        """공이 플레이어에게 점유되고 있는지 확인"""
        return (self.closest_player_distance < self.ball_possession_distance and 
                self.closest_player_id is not None)
    
    def move_with_closest_player(self, player_tracker_manager):
        """가장 가까운 플레이어와 함께 이동 (공이 보이지 않을 때)"""
        if self.closest_player_id is None:
            return
        
        # 해당 플레이어 찾기
        closest_tracker = None
        for tracker in player_tracker_manager.trackers:
            if tracker.tracker_id == self.closest_player_id:
                closest_tracker = tracker
                break
        
        if closest_tracker is not None:
            # 플레이어의 현재 위치로 공 이동
            player_center = closest_tracker.get_center()
            self.previous_position = self.position
            self.position = player_center
            
            # 플레이어의 속도를 공의 속도로 사용
            self.dx = closest_tracker.dx
            self.dy = closest_tracker.dy
    
    def increment_lost_score(self, frame: np.ndarray = None):
        """유실 점수 증가 (잔디색일 때 더 빠르게 증가)"""
        if frame is not None and self.is_on_grass(frame):
            # 잔디색 영역에 있으면 3배로 증가
            self.frames_lost_score += 3
        else:
            # 일반적인 경우 1씩 증가
            self.frames_lost_score += 0.5
    
    def is_on_grass(self, frame: np.ndarray) -> bool:
        """현재 공 위치가 잔디색 영역에 있는지 확인"""
        if self.position is None or frame is None:
            return False
        
        x, y = self.position
        frame_height, frame_width = frame.shape[:2]
        
        # 위치가 프레임 경계를 벗어나지 않도록 클리핑
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        
        # 중심점 주변의 픽셀들 샘플링
        sample_size = 2
        x1 = max(0, x - sample_size)
        y1 = max(0, y - sample_size)
        x2 = min(frame_width, x + sample_size + 1)
        y2 = min(frame_height, y + sample_size + 1)
        
        # 샘플 영역의 평균 색상 계산
        sample_region = frame[y1:y2, x1:x2]
        if sample_region.size == 0:
            return False
        
        avg_color = np.mean(sample_region, axis=(0, 1))
        
        # 잔디색과의 거리 계산
        grass_color_distance = np.sqrt(np.sum((avg_color - np.array(self.grass_color)) ** 2))
        
        # 거리 임계값
        grass_threshold = 40
        return grass_color_distance < grass_threshold
    
    def should_be_removed(self) -> bool:
        """제거되어야 하는지 판단"""
        return self.frames_lost_score > self.max_lost_frames
    
    def is_ball_inside_any_player_bbox(self, ball_bbox: Tuple[int, int, int, int], 
                                     player_tracker_manager) -> bool:
        """공이 어떤 플레이어의 bbox 안에 있는지 확인"""
        ball_center_x = ball_bbox[0] + ball_bbox[2] // 2
        ball_center_y = ball_bbox[1] + ball_bbox[3] // 2
        
        for tracker in player_tracker_manager.trackers:
            px, py, pw, ph = tracker.current_bbox
            
            # 공의 중심점이 플레이어 bbox 안에 있는지 확인
            if (px <= ball_center_x <= px + pw and 
                py <= ball_center_y <= py + ph):
                return True
        
        return False
    
    def get_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """현재 공의 바운딩 박스 반환"""
        if self.position is None:
            return None
        
        x, y = self.position
        size = self.radius * 2
        
        return (x - self.radius, y - self.radius, size, size)
    
    def get_possession_info(self) -> dict:
        """공 점유 정보 반환"""
        return {
            'player_id': self.closest_player_id,
            'team': self.closest_player_team,
            'distance': self.closest_player_distance,
            'in_possession': self.is_ball_in_player_possession()
        }


class BallTrackerManager:
    """BallTracker를 관리하는 클래스"""
    
    def __init__(self, frame_width: int, frame_height: int, grass_color: Tuple[int, int, int] = (0, 128, 0)):
        self.ball_tracker = BallTracker(grass_color)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grass_color = grass_color
        self.previous_detected_ball_bboxes = []  # 이전 프레임에서 감지된 공들
        self.bbox_match_threshold = 20  # bbox 매칭 거리 임계값
    
    def calculate_bbox_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """두 bbox 간의 중심점 거리 계산"""
        center1_x = bbox1[0] + bbox1[2] // 2
        center1_y = bbox1[1] + bbox1[3] // 2
        center2_x = bbox2[0] + bbox2[2] // 2
        center2_y = bbox2[1] + bbox2[3] // 2
        
        dx = center1_x - center2_x
        dy = center1_y - center2_y
        return math.sqrt(dx*dx + dy*dy)
    
    def find_best_matching_ball(self, detected_ball_bboxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """이전 감지된 공들과 현재 감지된 공들 중 가장 잘 매칭되는 공 찾기"""
        if not self.previous_detected_ball_bboxes or not detected_ball_bboxes:
            return None
        
        best_current_bbox = None
        min_total_distance = float('inf')
        
        # 현재 감지된 각 공에 대해
        for current_bbox in detected_ball_bboxes:
            total_distance = 0
            valid_matches = 0

            prev_bboxes = self.previous_detected_ball_bboxes[:]
            
            # 이전 감지된 공들과의 거리 계산
            for prev_bbox in prev_bboxes:
                distance = self.calculate_bbox_distance(current_bbox, prev_bbox)
                if distance < self.bbox_match_threshold:
                    total_distance += distance
                    valid_matches += 1
            
            # 유효한 매치가 있고, 평균 거리가 가장 작은 경우
            if valid_matches > 0:
                avg_distance = total_distance / valid_matches
                if avg_distance < min_total_distance:
                    min_total_distance = avg_distance
                    best_current_bbox = current_bbox
        
        return best_current_bbox
    
    def update_ball_tracking(self, detected_ball_bboxes: List[Tuple[int, int, int, int]], 
                           player_tracker_manager, frame: np.ndarray = None, frame_count: int = 0):
        """새로운 로직의 공 추적 업데이트"""
        
        debug_prefix = f"[Frame {frame_count:4d}]"
        
        # 1단계: detected_ball_bboxes가 있는 경우
        if detected_ball_bboxes:
            
            # 1-1: previous_detected_ball_bboxes가 없는 경우 (첫 번째 또는 리셋 후)
            if not self.previous_detected_ball_bboxes:
                # 이번 턴은 쉬고, 단순히 previous로 등록만
                self.previous_detected_ball_bboxes = detected_ball_bboxes[:]
                print(f"{debug_prefix} 첫 감지: {len(detected_ball_bboxes)}개 후보 등록")
                return
            
            # 1-2: previous_detected_ball_bboxes가 있는 경우 - 매칭 시도
            best_match = self.find_best_matching_ball(detected_ball_bboxes)
            
            if best_match is not None:
                # 매칭되는 공이 있으면 진짜 공으로 등록/업데이트
                if self.ball_tracker.is_active:
                    self.ball_tracker.update_ball(best_match)
                    pos = self.ball_tracker.position
                    print(f"{debug_prefix} 공 업데이트: ({pos[0]},{pos[1]}) | 속도:({self.ball_tracker.dx},{self.ball_tracker.dy})")
                else:
                    self.ball_tracker.register_ball(best_match)
                    pos = self.ball_tracker.position
                    print(f"{debug_prefix} 공 신규등록: ({pos[0]},{pos[1]})")
                
                # 가장 가까운 플레이어 업데이트
                self.ball_tracker.find_closest_player(player_tracker_manager)
                
                # 현재 공의 위치의 가까운 bbox만 previous에 저장
                for bbox in detected_ball_bboxes:
                    if self.calculate_bbox_distance(bbox, best_match) < self.bbox_match_threshold * 1.8:
                        self.previous_detected_ball_bboxes.append(bbox)
            else:
                # 매칭되는 공이 없으면 유실 점수 증가
                if self.ball_tracker.is_active:
                    self.ball_tracker.increment_lost_score(frame)
                    print(f"{debug_prefix} 매칭실패 | 유실점수: {self.ball_tracker.frames_lost_score}")
                
                # 매칭 실패시에도 현재 감지된 공들을 이전 공들로 업데이트
                self.previous_detected_ball_bboxes = detected_ball_bboxes[:]
        
        # 2단계: detected_ball_bboxes가 없는 경우
        else:
            # 이전 dx, dy를 바탕으로 공을 이동
            if self.ball_tracker.is_active and self.ball_tracker.position is not None:
                # 예측된 위치로 이동
                predicted_pos = self.ball_tracker.predict_next_position()
                if predicted_pos is not None:
                    
                    #가장 가까운 플레이어와 함께 이동 (선택적)
                    self.ball_tracker.find_closest_player(player_tracker_manager)
                    if self.ball_tracker.is_ball_in_player_possession():
                        self.ball_tracker.move_with_closest_player(player_tracker_manager)
                    
                    else:
                        self.ball_tracker.previous_position = self.ball_tracker.position
                        self.ball_tracker.position = predicted_pos
                        
                    # 유실 점수 증가
                    self.ball_tracker.increment_lost_score(frame)
                    
                    pos = self.ball_tracker.position
                    print(f"{debug_prefix} 예측이동: ({pos[0]},{pos[1]}) | 유실: {self.ball_tracker.frames_lost_score}")
            
            # previous_detected_ball_bboxes는 유지 (다음 프레임에서 사용하기 위해)
        
        # 3단계: 공 제거 확인
        if self.ball_tracker.should_be_removed():
            self.ball_tracker.is_active = False
            self.ball_tracker.position = None
            self.previous_detected_ball_bboxes = []  # 리셋
            print(f"{debug_prefix} 공 추적 종료 - 리셋")
    
    def reset_ball_tracking(self):
        """공 추적 리셋"""
        self.ball_tracker.is_active = False
        self.ball_tracker.position = None
        self.previous_detected_ball_bboxes = []
        print("[BallTracker] 수동 리셋")
    
    def get_ball_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """현재 공의 바운딩 박스 반환"""
        if self.ball_tracker.is_active:
            return self.ball_tracker.get_bbox()
        return None
    
    def get_ball_info(self) -> dict:
        """공의 상태 정보 반환"""
        if not self.ball_tracker.is_active:
            return {'active': False}
        
        return {
            'active': True,
            'position': self.ball_tracker.position,
            'radius': self.ball_tracker.radius,
            'velocity': (self.ball_tracker.dx, self.ball_tracker.dy),
            'frames_lost': self.ball_tracker.frames_lost_score,
            'possession': self.ball_tracker.get_possession_info()
        } 