import numpy as np
from typing import List, Tuple, Optional

class PlayerTracker:
    """개별 선수를 추적하는 클래스"""
    
    def __init__(self, team_id: int, initial_bbox: Tuple[int, int, int, int], tracker_id: int, grass_color: Tuple[int, int, int]):
        self.team_id = team_id  # 팀 정보 (1 또는 2)
        self.tracker_id = tracker_id  # 고유 ID
        self.current_bbox = initial_bbox  # (x, y, w, h)
        self.previous_bbox = initial_bbox
        self.dx = 0  # x 방향 속도
        self.dy = 0  # y 방향 속도
        self.frames_lost_score = 0  # 연속으로 bbox를 찾지 못한 점수 (잔디색일 때 3배)
        self.max_lost_frames_in_bounds = 45  # 화면 안에서의 최대 허용 유실 프레임
        self.max_lost_frames_out_bounds = 2   # 화면 밖에서의 최대 허용 유실 프레임 (즉시 제거)
        self.out_of_bounds_frames = 0  # 화면 밖에 있었던 연속 프레임 수
        self.grass_color = grass_color  # 잔디 색상 (BGR)
        
    def update_bbox(self, new_bbox: Tuple[int, int, int, int]):
        """bbox 업데이트 및 속도 계산"""
        self.previous_bbox = self.current_bbox
        self.current_bbox = new_bbox
        
        # 속도 계산 (중심점 기준)
        prev_center_x = self.previous_bbox[0] + self.previous_bbox[2] // 2
        prev_center_y = self.previous_bbox[1] + self.previous_bbox[3] // 2
        curr_center_x = new_bbox[0] + new_bbox[2] // 2
        curr_center_y = new_bbox[1] + new_bbox[3] // 2
        
        self.dx = curr_center_x - prev_center_x
        self.dy = curr_center_y - prev_center_y
        
        # 성공적으로 업데이트했으므로 카운터 리셋
        self.frames_lost_score = 0
        self.out_of_bounds_frames = 0
    
    def predict_next_position(self) -> Tuple[int, int, int, int]:
        """속도를 기반으로 다음 위치 예측"""
        x, y, w, h = self.current_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 예측된 중심점
        pred_center_x = center_x + self.dx
        pred_center_y = center_y + self.dy
        
        # 예측된 bbox
        pred_x = pred_center_x - w // 2
        pred_y = pred_center_y - h // 2
        
        return (int(pred_x), int(pred_y), w, h)
    
    def get_center(self) -> Tuple[int, int]:
        """현재 bbox의 중심점 반환"""
        x, y, w, h = self.current_bbox
        return (x + w // 2, y + h // 2)
    
    def calculate_distance_to_bbox(self, bbox: Tuple[int, int, int, int]) -> float:
        """현재 위치에서 주어진 bbox까지의 거리 계산"""
        curr_center = self.get_center()
        bbox_center_x = bbox[0] + bbox[2] // 2
        bbox_center_y = bbox[1] + bbox[3] // 2
        
        dx = curr_center[0] - bbox_center_x
        dy = curr_center[1] - bbox_center_y
        return np.sqrt(dx*dx + dy*dy)
    
    def calculate_predicted_distance_to_bbox(self, bbox: Tuple[int, int, int, int]) -> float:
        """예측 위치에서 주어진 bbox까지의 거리 계산"""
        predicted_bbox = self.predict_next_position()
        pred_center_x = predicted_bbox[0] + predicted_bbox[2] // 2
        pred_center_y = predicted_bbox[1] + predicted_bbox[3] // 2
        
        bbox_center_x = bbox[0] + bbox[2] // 2
        bbox_center_y = bbox[1] + bbox[3] // 2
        
        dx = pred_center_x - bbox_center_x
        dy = pred_center_y - bbox_center_y
        return np.sqrt(dx*dx + dy*dy)
    
    def increment_lost_frames(self):
        """유실 프레임 카운트 증가"""
        self.frames_lost_score += 1
    
    def increment_out_of_bounds_frames(self):
        """화면 밖 프레임 카운트 증가"""
        self.out_of_bounds_frames += 1
    
    def is_out_of_bounds(self, frame_width: int, frame_height: int) -> bool:
        """화면 밖으로 나갔는지 확인"""
        x, y, w, h = self.current_bbox
        
        # 현재 bbox가 완전히 화면 밖에 있는 경우
        completely_out = (x + w < 0 or x > frame_width or 
                         y + h < 0 or y > frame_height)
        
        if completely_out:
            return True
        
        # 화면 경계 근처에 있으면서 밖으로 향하고 있는지 확인
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 예측된 다음 위치
        next_center_x = center_x + self.dx * self.frames_lost_score
        next_center_y = center_y + self.dy * self.frames_lost_score
        
        # 경계 마진 (bbox 크기의 절반)
        margin_x = w // 2
        margin_y = h // 2
        
        # 현재 화면 경계 근처에 있고, 다음 위치가 화면 밖으로 향하는 경우
        near_left_edge = x <= margin_x and self.dx < 0
        near_right_edge = x + w >= frame_width - margin_x and self.dx > 0
        near_top_edge = y <= margin_y and self.dy < 0
        near_bottom_edge = y + h >= frame_height - margin_y and self.dy > 0
        
        # 다음 예측 위치가 화면 밖에 있는 경우
        next_out_of_bounds = (next_center_x - margin_x < 0 or 
                             next_center_x + margin_x > frame_width or
                             next_center_y - margin_y < 0 or 
                             next_center_y + margin_y > frame_height)
        
        # 경계 근처에서 밖으로 향하거나, 다음 위치가 밖인 경우
        return (near_left_edge or near_right_edge or near_top_edge or near_bottom_edge or 
                next_out_of_bounds)
    
    def should_be_removed(self, frame_width: int, frame_height: int) -> bool:
        """제거되어야 하는지 판단"""
        is_out_bounds = self.is_out_of_bounds(frame_width, frame_height)
        
        if is_out_bounds:
            self.increment_out_of_bounds_frames()
            # 화면 밖에 있는 경우: 엄격한 기준 적용
            return self.out_of_bounds_frames > self.max_lost_frames_out_bounds
        else:
            # 화면 안에 있는 경우: 관대한 기준 적용
            return self.frames_lost_score > self.max_lost_frames_in_bounds
    
    def is_bbox_on_grass(self, frame: np.ndarray) -> bool:
        """현재 bbox가 잔디색 영역에 있는지 확인"""
        if frame is None:
            return False
        
        x, y, w, h = self.current_bbox
        frame_height, frame_width = frame.shape[:2]
        
        # bbox가 프레임 경계를 벗어나지 않도록 클리핑
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        w = max(1, min(w, frame_width - x))
        h = max(1, min(h, frame_height - y))
        
        # bbox 중심점 주변의 픽셀들 샘플링 (5x5 영역)
        center_x = x + w // 2
        center_y = y + h // 2
        
        sample_size = 2  # 중심점 주변 2픽셀
        x1 = max(0, center_x - sample_size)
        y1 = max(0, center_y - sample_size)
        x2 = min(frame_width, center_x + sample_size + 1)
        y2 = min(frame_height, center_y + sample_size + 1)
        
        # 샘플 영역의 평균 색상 계산
        sample_region = frame[y1:y2, x1:x2]
        if sample_region.size == 0:
            return False
        
        avg_color = np.mean(sample_region, axis=(0, 1))
        
        # 잔디색과의 거리 계산 (BGR)
        grass_color_distance = np.sqrt(np.sum((avg_color - np.array(self.grass_color)) ** 2))
        
        # 거리 임계값 (잔디색 허용 범위)
        grass_threshold = 50  # 조정 가능한 값
        
        return grass_color_distance < grass_threshold
    
    def increment_lost_score(self, frame: np.ndarray = None):
        """유실 점수 증가 (잔디색일 때 3배)"""
        if frame is not None and self.is_bbox_on_grass(frame):
            # 잔디색 영역에 있으면 6배로 증가
            self.frames_lost_score += 6
        else:
            # 일반적인 경우 1씩 증가
            self.frames_lost_score += 1

class PlayerTrackerManager:
    """PlayerTracker들을 관리하는 클래스"""
    
    def __init__(self, frame_width: int, frame_height: int, grass_color: Tuple[int, int, int] = (0, 128, 0)):
        self.trackers = []  # PlayerTracker 객체들의 리스트
        self.next_tracker_id = 0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_assignment_distance = 50  # bbox 할당 최대 거리 (더 엄격하게)
        self.initialization_complete = False  # 초기화 완료 플래그
        self.grass_color = grass_color  # 잔디 색상 (BGR)
    
    def initialize_trackers(self, team1_bboxes: List[Tuple[int, int, int, int]], 
                           team2_bboxes: List[Tuple[int, int, int, int]]):
        """첫 프레임에서 초기 tracker들 생성"""
        if self.initialization_complete:
            return
        
        # 팀1 tracker들 생성
        for bbox in team1_bboxes:
            new_tracker = PlayerTracker(1, bbox, self.next_tracker_id, self.grass_color)
            self.trackers.append(new_tracker)
            self.next_tracker_id += 1
        
        # 팀2 tracker들 생성
        for bbox in team2_bboxes:
            new_tracker = PlayerTracker(2, bbox, self.next_tracker_id, self.grass_color)
            self.trackers.append(new_tracker)
            self.next_tracker_id += 1
        
        self.initialization_complete = True
        print(f"초기화 완료: Team1 {len(team1_bboxes)}명, Team2 {len(team2_bboxes)}명 등록")
    
    def update_trackers_only(self, team1_bboxes: List[Tuple[int, int, int, int]], 
                            team2_bboxes: List[Tuple[int, int, int, int]], frame: np.ndarray = None):
        """기존 tracker들만 업데이트 (새로운 tracker 생성 안함)"""
        if not self.initialization_complete:
            return
        
        all_bboxes = [(bbox, 1) for bbox in team1_bboxes] + [(bbox, 2) for bbox in team2_bboxes]
        
        # 1단계: 각 tracker에 대해 가장 가까운 bbox 찾기
        assignments = self._assign_bboxes_to_trackers(all_bboxes)
        
        # 2단계: 충돌 해결 (여러 tracker가 같은 bbox를 원하는 경우)
        assignments = self._resolve_conflicts(assignments, all_bboxes)
        
        # 3단계: tracker 업데이트
        self._update_tracker_positions(assignments, frame)
        
        # 4단계: 유실되거나 화면 밖으로 나간 tracker 정리
        self._cleanup_trackers()
    
    def update_trackers(self, team1_bboxes: List[Tuple[int, int, int, int]], 
                       team2_bboxes: List[Tuple[int, int, int, int]], frame: np.ndarray = None):
        """모든 tracker들을 새로운 bbox 정보로 업데이트 (기존 방식 - 하위 호환성)"""
        all_bboxes = [(bbox, 1) for bbox in team1_bboxes] + [(bbox, 2) for bbox in team2_bboxes]
        
        # 1단계: 각 tracker에 대해 가장 가까운 bbox 찾기
        assignments = self._assign_bboxes_to_trackers(all_bboxes)
        
        # 2단계: 충돌 해결 (여러 tracker가 같은 bbox를 원하는 경우)
        assignments = self._resolve_conflicts(assignments, all_bboxes)
        
        # 3단계: tracker 업데이트
        self._update_tracker_positions(assignments, frame)
        
        # 4단계: 새로운 bbox들로 새 tracker 생성
        self._create_new_trackers(all_bboxes, assignments)
        
        # 5단계: 유실되거나 화면 밖으로 나간 tracker 정리
        self._cleanup_trackers()
    
    def _assign_bboxes_to_trackers(self, all_bboxes: List[Tuple[Tuple[int, int, int, int], int]]) -> dict:
        """각 tracker에 가장 가까운 bbox 할당 (예측 위치 기반)"""
        assignments = {}
        
        for tracker in self.trackers:
            best_bbox = None
            best_distance = float('inf')
            best_idx = -1
            
            for idx, (bbox, team_id) in enumerate(all_bboxes):
                # 같은 팀만 고려
                if team_id != tracker.team_id:
                    continue
                
                # 예측 위치를 기반으로 거리 계산
                distance = tracker.calculate_predicted_distance_to_bbox(bbox)
                
                if distance < best_distance and distance < self.max_assignment_distance:
                    best_distance = distance
                    best_bbox = bbox
                    best_idx = idx
            
            if best_bbox is not None:
                assignments[tracker.tracker_id] = (best_idx, best_bbox, best_distance)
        
        return assignments
    
    def _resolve_conflicts(self, assignments: dict, all_bboxes: List) -> dict:
        """여러 tracker가 같은 bbox를 원하는 경우 해결"""
        bbox_conflicts = {}
        
        # 충돌 찾기
        for tracker_id, (bbox_idx, bbox, distance) in assignments.items():
            if bbox_idx not in bbox_conflicts:
                bbox_conflicts[bbox_idx] = []
            bbox_conflicts[bbox_idx].append((tracker_id, distance))
        
        # 충돌 해결
        resolved_assignments = {}
        losing_trackers = []  # 충돌에서 진 tracker들을 저장
        
        for bbox_idx, competing_trackers in bbox_conflicts.items():
            if len(competing_trackers) == 1:
                # 충돌 없음
                tracker_id = competing_trackers[0][0]
                resolved_assignments[tracker_id] = assignments[tracker_id]
            else:
                # 충돌 있음 - 가장 가까운 tracker에게 할당
                competing_trackers.sort(key=lambda x: x[1])  # 거리순 정렬
                winner_tracker_id = competing_trackers[0][0]
                resolved_assignments[winner_tracker_id] = assignments[winner_tracker_id]
                
                # 나머지 tracker들은 losing_trackers에 추가
                for tracker_id, _ in competing_trackers[1:]:
                    losing_trackers.append(tracker_id)
        
        # 충돌에서 진 tracker들에게 할당되지 않은 bbox 중 가장 가까운 것 할당
        self._assign_unassigned_bboxes_to_losing_trackers(
            losing_trackers, all_bboxes, resolved_assignments
        )
        
        return resolved_assignments
    
    def _assign_unassigned_bboxes_to_losing_trackers(self, losing_tracker_ids: List[int], 
                                                   all_bboxes: List, resolved_assignments: dict):
        """충돌에서 진 tracker들에게 할당되지 않은 bbox 중 가장 가까운 것 할당"""
        if not losing_tracker_ids:
            return
        
        # 이미 할당된 bbox들의 인덱스 구하기
        assigned_bbox_indices = set()
        for _, (bbox_idx, _, _) in resolved_assignments.items():
            if bbox_idx >= 0:  # -1은 분할된 bbox였으므로 제외
                assigned_bbox_indices.add(bbox_idx)
        
        # 할당되지 않은 bbox들 찾기
        unassigned_bboxes = []
        for idx, (bbox, team_id) in enumerate(all_bboxes):
            if idx not in assigned_bbox_indices:
                unassigned_bboxes.append((idx, bbox, team_id))
        
        # 각 losing tracker에 대해 가장 가까운 할당되지 않은 bbox 찾기
        for tracker_id in losing_tracker_ids:
            tracker = next((t for t in self.trackers if t.tracker_id == tracker_id), None)
            if tracker is None:
                continue
            
            best_bbox = None
            best_distance = float('inf')
            best_idx = -1
            
            for idx, bbox, team_id in unassigned_bboxes:
                # 같은 팀만 고려
                if team_id != tracker.team_id:
                    continue
                
                # 예측 위치를 기반으로 거리 계산
                distance = tracker.calculate_predicted_distance_to_bbox(bbox)
                
                if distance < best_distance and distance < self.max_assignment_distance:
                    best_distance = distance
                    best_bbox = bbox
                    best_idx = idx
            
            # 적절한 bbox를 찾았다면 할당
            if best_bbox is not None:
                resolved_assignments[tracker_id] = (best_idx, best_bbox, best_distance)
                # 할당된 bbox는 unassigned_bboxes에서 제거
                unassigned_bboxes = [(i, b, t) for i, b, t in unassigned_bboxes if i != best_idx]
    
    def _update_tracker_positions(self, assignments: dict, frame: np.ndarray = None):
        """assignments에 따라 tracker 위치 업데이트"""
        for tracker in self.trackers:
            if tracker.tracker_id in assignments:
                _, bbox, _ = assignments[tracker.tracker_id]
                tracker.update_bbox(bbox)
            else:
                # bbox를 찾지 못한 경우
                tracker.increment_lost_score(frame)
                
                # 화면 밖에 있는지 확인하고 카운터 증가
                if tracker.is_out_of_bounds(self.frame_width, self.frame_height):
                    tracker.increment_out_of_bounds_frames()
    
    def _create_new_trackers(self, all_bboxes: List, assignments: dict):
        """할당되지 않은 bbox들로 새로운 tracker 생성"""
        assigned_indices = set()
        for _, (bbox_idx, _, _) in assignments.items():
            if bbox_idx >= 0:  # -1은 분할된 bbox
                assigned_indices.add(bbox_idx)
        
        for idx, (bbox, team_id) in enumerate(all_bboxes):
            if idx not in assigned_indices:
                new_tracker = PlayerTracker(team_id, bbox, self.next_tracker_id, self.grass_color)
                self.trackers.append(new_tracker)
                self.next_tracker_id += 1
    
    def _cleanup_trackers(self):
        """제거되어야 할 tracker들 정리"""
        self.trackers = [
            tracker for tracker in self.trackers 
            if not tracker.should_be_removed(self.frame_width, self.frame_height)
        ]
    
    def get_all_bboxes(self) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """팀별로 모든 tracker의 bbox 반환"""
        team1_bboxes = []
        team2_bboxes = []
        
        for tracker in self.trackers:
            if tracker.team_id == 1:
                team1_bboxes.append(tracker.current_bbox)
            else:
                team2_bboxes.append(tracker.current_bbox)
        
        return team1_bboxes, team2_bboxes
    
    def get_tracker_count(self) -> Tuple[int, int]:
        """팀별 tracker 수 반환"""
        team1_count = sum(1 for t in self.trackers if t.team_id == 1)
        team2_count = sum(1 for t in self.trackers if t.team_id == 2)
        return team1_count, team2_count 