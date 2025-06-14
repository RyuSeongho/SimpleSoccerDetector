import numpy as np
import cv2
from typing import List, Tuple, Optional

class BallTracker:
    """개별 공을 추적하는 클래스"""
    
    def __init__(self, initial_position: Tuple[int, int], tracker_id: int, initial_score: int = 100):
        self.tracker_id = tracker_id  # 고유 ID
        self.position = initial_position  # (x, y) 중심점 좌표
        self.previous_position = initial_position
        self.score = initial_score  # 초기 점수 (거리에 따라 조정 가능)
        self.dx = 0  # x 방향 속도
        self.dy = 0  # y 방향 속도
        self.frames_since_last_update = 0  # 마지막 업데이트 이후 프레임 수
        
    def update_position(self, new_position: Tuple[int, int]):
        """위치 업데이트 및 속도 계산"""
        self.previous_position = self.position
        self.position = new_position
        
        # 속도 계산 (프레임 간격 고려)
        frames_gap = max(1, self.frames_since_last_update + 1)  # 최소 1프레임
        self.dx = (new_position[0] - self.previous_position[0]) / frames_gap
        self.dy = (new_position[1] - self.previous_position[1]) / frames_gap
        
        # 업데이트 프레임 카운터 리셋
        self.frames_since_last_update = 0
    
    def update_score_success(self, distance: float):
        """선택되었을 때 점수 업데이트 (거리에 반비례해서 5~20점 상승, 최대 300점)"""
        # 거리가 0에 가까울수록 높은 점수, 30에 가까울수록 낮은 점수
        # distance 0 -> 20점, distance 30 -> 5점
        score_increase = max(0, 20 - (distance / 30) * 30)
        self.score += score_increase
        
        # 최대 점수 300점 제한
        if self.score > 300:
            self.score = 300
        
    def update_score_failure(self):
        """선택받지 못했을 때 점수 5점 감소"""
        self.score -= 5
        if self.score < 0:
            self.score = 0
    
    def get_center(self) -> Tuple[int, int]:
        """현재 위치 반환"""
        return self.position
    
    def calculate_distance_to_position(self, position: Tuple[int, int]) -> float:
        """현재 위치에서 주어진 위치까지의 거리 계산"""
        dx = self.position[0] - position[0]
        dy = self.position[1] - position[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def get_predicted_position(self) -> Tuple[int, int]:
        """속도를 고려한 예측 위치 계산"""
        # frames_since_last_update + 1 프레임만큼 예측
        prediction_frames = self.frames_since_last_update + 1
        
        predicted_x = self.position[0] + (self.dx * prediction_frames)
        predicted_y = self.position[1] + (self.dy * prediction_frames)
        
        return (int(predicted_x), int(predicted_y))
    
    def calculate_predicted_distance_to_position(self, position: Tuple[int, int]) -> float:
        """예측 위치에서 주어진 위치까지의 거리 계산"""
        predicted_pos = self.get_predicted_position()
        dx = predicted_pos[0] - position[0]
        dy = predicted_pos[1] - position[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def increment_frames_since_update(self):
        """업데이트되지 않은 프레임 수 증가"""
        self.frames_since_last_update += 1


class BallTrackerManager:
    """BallTracker들을 관리하는 클래스"""
    
    def __init__(self, frame_width: int, frame_height: int, grass_color: Tuple[int, int, int]):
        self.trackers = []  # BallTracker 객체들의 리스트
        self.next_tracker_id = 0
        self.distance_threshold = 30  # blob 할당 거리 임계값
        self.min_distance_to_player = 25  # 선수와의 최소 거리 (이보다 가까우면 tracker 생성 안함)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grass_color = grass_color
        
        # 기존 인터페이스 호환성을 위한 변수들
        self.current_ball_bbox = None
        self.frames_lost = 0
        self.max_lost_frames = 30
        
        # 이전 최고 점수 tracker의 위치 기억
        self.last_best_position = None
        
    def update_trackers(self, ball_candidates: List[Tuple[int, int, int, int]], 
                       player_positions: List[Tuple[int, int]]):
        """새로운 공 후보들로 tracker들 업데이트"""
        
        # 1. ball_candidates를 중심점 좌표로 변환
        ball_centers = []
        for bbox in ball_candidates:
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            ball_centers.append((center_x, center_y))
        
        # 2. 선수와 너무 가까운 후보들 제거 (새 tracker 생성용)
        filtered_ball_centers = self._filter_candidates_near_players(ball_centers, player_positions)
        self._create_new_trackers(filtered_ball_centers)
        
        # 3. 기존 tracker들과 모든 후보들 매칭 (선수 근처 포함 - 업데이트는 가능)
        assignments = self._assign_candidates_to_trackers(ball_centers)
        
        # 4. 모든 tracker들의 프레임 카운터 증가 (할당 전)
        for tracker in self.trackers:
            tracker.increment_frames_since_update()
        
        # 5. 할당 결과에 따라 tracker들 업데이트
        self._update_tracker_positions_and_scores(assignments, ball_centers)
        
        # 6. 점수가 0 이하인 tracker들 제거
        self._cleanup_trackers()
        
        # 7. 현재 최고 점수 tracker의 위치 업데이트 (다음 프레임 생성시 참고용)
        self._update_last_best_position()
    
    def _filter_candidates_near_players(self, ball_centers: List[Tuple[int, int]], 
                                      player_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """선수와 너무 가까운 공 후보들 제거 (새 tracker 생성 방지용, 기존 tracker 업데이트에는 사용 가능)"""
        filtered_centers = []
        
        for ball_center in ball_centers:
            too_close_to_player = False
            
            for player_pos in player_positions:
                distance = self._calculate_distance(ball_center, player_pos)
                if distance < self.min_distance_to_player:
                    too_close_to_player = True
                    break
            
            if not too_close_to_player:
                filtered_centers.append(ball_center)
        
        return filtered_centers
    

    
    def _create_new_trackers(self, ball_centers: List[Tuple[int, int]]):
        """새로운 공 후보들로 tracker 생성 (이전 최고 위치와의 거리에 따라 초기 점수 조정)"""
        for center in ball_centers:
            # 기존 tracker들과 너무 가까운지 확인
            too_close_to_existing = False
            for tracker in self.trackers:
                distance = tracker.calculate_distance_to_position(center)
                if distance < self.distance_threshold:
                    too_close_to_existing = True
                break
            
            # 기존 tracker와 충분히 떨어져 있으면 새 tracker 생성
            if not too_close_to_existing:
                # 이전 최고 위치와의 거리에 따라 초기 점수 계산
                initial_score = self._calculate_initial_score(center)
                new_tracker = BallTracker(center, self.next_tracker_id, initial_score)
                self.trackers.append(new_tracker)
                self.next_tracker_id += 1
    
    def _assign_candidates_to_trackers(self, ball_centers: List[Tuple[int, int]]) -> dict:
        """각 tracker에 가장 가까운 공 후보 할당 (거리 임계값 30)"""
        assignments = {}
        used_candidates = set()
        
        # 각 tracker별로 가장 가까운 후보 찾기
        tracker_candidate_distances = []
        
        for i, tracker in enumerate(self.trackers):
            best_candidate = None
            best_distance = float('inf')
            best_candidate_idx = -1
            
            for j, candidate in enumerate(ball_centers):
                if j in used_candidates:
                    continue
                
                distance = tracker.calculate_predicted_distance_to_position(candidate)
                if distance <= self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate
                    best_candidate_idx = j
            
            if best_candidate is not None:
                tracker_candidate_distances.append((i, best_candidate_idx, best_distance))
        
        # 거리 순으로 정렬하여 충돌 해결 (가장 가까운 것부터 우선 할당)
        tracker_candidate_distances.sort(key=lambda x: x[2])
        
        for tracker_idx, candidate_idx, distance in tracker_candidate_distances:
            if candidate_idx not in used_candidates:
                assignments[tracker_idx] = (candidate_idx, distance)
                used_candidates.add(candidate_idx)
        
        return assignments
    
    def _update_tracker_positions_and_scores(self, assignments: dict, ball_centers: List[Tuple[int, int]]):
        """할당 결과에 따라 tracker 위치와 점수 업데이트"""
        for i, tracker in enumerate(self.trackers):
            if i in assignments:
                # 선택된 tracker: 위치 업데이트 및 점수 상승 (거리에 반비례해서 5~20)
                candidate_idx, distance = assignments[i]
                new_position = ball_centers[candidate_idx]
                tracker.update_position(new_position)
                tracker.update_score_success(distance)
            else:
                # 선택되지 않은 tracker: 점수 5 감소만
                tracker.update_score_failure()
    
    def _cleanup_trackers(self):
        """점수가 0이 되면 tracker들 제거"""
        self.trackers = [tracker for tracker in self.trackers if tracker.score > 0]
    
    def _update_last_best_position(self):
        """현재 최고 점수 tracker의 위치를 기억"""
        if self.trackers:
            best_tracker = max(self.trackers, key=lambda t: t.score)
            self.last_best_position = best_tracker.get_center()
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """두 점 사이의 거리 계산"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def _calculate_initial_score(self, new_position: Tuple[int, int]) -> int:
        """이전 최고 위치와의 거리에 따라 초기 점수 계산 (거리 제곱에 반비례)"""
        if self.last_best_position is None:
            # 처음 tracker 생성시에는 기본 점수 100
            return 100
        
        # 이전 최고 위치와의 거리 계산
        distance = self._calculate_distance(new_position, self.last_best_position)
        
        # 거리의 제곱에 반비례한 점수 계산
        # 거리 0 -> 300점, 거리 150 이상 -> 50점
        max_distance = 150  # 최대 거리 임계값
        min_score = 50      # 최소 점수
        max_score = 300     # 최대 점수
        
        if distance >= max_distance:
            return min_score
        else:
            # 거리의 제곱에 반비례하여 점수 계산 (급격한 감소)
            score_range = max_score - min_score  # 250
            distance_ratio = distance / max_distance  # 0~1
            squared_ratio = distance_ratio ** 2  # 0~1 (급격하게 증가)
            initial_score = max_score - (squared_ratio * score_range)
            return int(initial_score)
    
    def get_best_ball_position(self) -> Optional[Tuple[int, int]]:
        """가장 점수가 높은 tracker의 위치 반환"""
        if not self.trackers:
            return None
        
        best_tracker = max(self.trackers, key=lambda t: t.score)
        return best_tracker.get_center()
    
    def get_all_ball_positions(self) -> List[Tuple[Tuple[int, int], float]]:
        """모든 tracker의 위치와 점수 반환"""
        return [(tracker.get_center(), tracker.score) for tracker in self.trackers]
    
    def get_tracker_count(self) -> int:
        """현재 tracker 개수 반환"""
        return len(self.trackers)
    
    def update_ball_tracking(self, ball_candidates: List[Tuple[int, int, int, int]], 
                           player_tracker_manager, frame: np.ndarray, frame_count: int):
        """기존 인터페이스 호환성을 위한 메소드"""
        # PlayerTrackerManager에서 모든 플레이어 위치 가져오기
        team1_bboxes, team2_bboxes = player_tracker_manager.get_all_bboxes()
        
        # bbox를 중심점으로 변환
        player_positions = []
        for bbox in team1_bboxes + team2_bboxes:
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            player_positions.append((center_x, center_y))
        
        # tracker 업데이트
        self.update_trackers(ball_candidates, player_positions)
        
        # 현재 ball bbox 업데이트
        best_position = self.get_best_ball_position()
        if best_position is not None:
            # 위치를 bbox 형태로 변환 (임시로 10x10 크기)
            x, y = best_position
            self.current_ball_bbox = (x - 5, y - 5, 10, 10)
            self.frames_lost = 0
        else:
            self.current_ball_bbox = None
            self.frames_lost += 1
    
    def get_ball_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """현재 추적 중인 공의 bbox 반환"""
        return self.current_ball_bbox
    
    def get_ball_info(self) -> dict:
        """공 추적 정보 반환 (기존 인터페이스 호환성)"""
        best_position = self.get_best_ball_position()
        is_active = best_position is not None
        
        # 간단한 possession 정보 (더 정교한 구현 가능)
        possession_info = {
            'in_possession': False,
            'team': 1,
            'player_id': 0,
            'distance': 0.0
        }
        
        return {
            'active': is_active,
            'frames_lost': self.frames_lost,
            'possession': possession_info
        }
    
    def draw_all_trackers(self, frame: np.ndarray, show_window: bool = True) -> np.ndarray:
        """모든 BallTracker들을 점수와 함께 시각화"""
        result = frame.copy()
        
        if not self.trackers:
            # tracker가 없으면 "No Ball Trackers" 텍스트 표시
            cv2.putText(result, "No Ball Trackers", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if show_window:
                cv2.imshow("Ball Trackers", result)
            return result
        
        # 가장 높은 점수 찾기 (강조 표시용)
        max_score = max(tracker.score for tracker in self.trackers)
        
        for i, tracker in enumerate(self.trackers):
            x, y = tracker.get_center()
            predicted_x, predicted_y = tracker.get_predicted_position()
            score = tracker.score
            
            # 가장 높은 점수인지 확인
            is_best = (score == max_score)
            
            # 색상 설정
            if is_best:
                circle_color = (0, 255, 0)  # 초록색 (최고 점수)
                text_color = (0, 255, 0)
                circle_thickness = 3
            else:
                circle_color = (255, 255, 0)  # 청록색 (일반)
                text_color = (255, 255, 255)  # 흰색
                circle_thickness = 2
            
            # 원 그리기 (점수에 비례한 크기)
            radius = max(8, int(8 + score / 50))  # 최소 8, 점수에 비례해서 커짐
            cv2.circle(result, (int(x), int(y)), radius, circle_color, circle_thickness)
            
            # 중심점 표시
            cv2.circle(result, (int(x), int(y)), 2, circle_color, -1)
            
            # 예측 위치 표시 (작은 점)
            cv2.circle(result, (int(predicted_x), int(predicted_y)), 3, circle_color, 1)
            
            # 현재 위치에서 예측 위치로의 화살표
            if tracker.frames_since_last_update > 0:
                cv2.arrowedLine(result, (int(x), int(y)), (int(predicted_x), int(predicted_y)), 
                               circle_color, 1, tipLength=0.3)
            
            # 점수 텍스트 표시
            score_text = f"{score:.1f}"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = int(x - text_size[0] // 2)
            text_y = int(y - radius - 8)
            
            # 텍스트 배경 (가독성을 위해)
            cv2.rectangle(result, (text_x - 2, text_y - text_size[1] - 2), 
                         (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
            
            # 점수 텍스트
            cv2.putText(result, score_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # ID 표시 (작게)
            id_text = f"ID:{tracker.tracker_id}"
            cv2.putText(result, id_text, (int(x - 15), int(y + radius + 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        
        # 정보 텍스트 표시
        info_text = f"Ball Trackers: {len(self.trackers)} | Best Score: {max_score:.1f}"
        cv2.putText(result, info_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 범례 표시
        cv2.putText(result, "Green: Best | Cyan: Others", (20, result.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if show_window:
            cv2.imshow("Ball Trackers", result)
        
        return result 