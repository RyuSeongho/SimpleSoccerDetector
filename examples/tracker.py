import cv2
import numpy as np
from centroid import get_centroid
import random

# ---- 개선된 PlayerTracker 클래스 ----
class PlayerTracker:
    def __init__(self, grass_color=(0, 0, 0)):
        self.next_object_id = 0
        self.objects = {}      # {object_id: {'centroid': (x,y), 'bbox': (x1,y1,x2,y2), 'template': np.array}}
        self.current_frame = 0
        self.grass_color = grass_color
        self.consecutive_failures = {}  # 각 객체별 연속 실패 횟수를 추적

    def setColor(self, grass_color):
        """플레이어 추적에 사용할 잔디 색상 설정"""
        self.grass_color = grass_color
        print(f"잔디 색상 설정: {self.grass_color}")

    
    def is_adjacent_to_grass(self, centroid, frame, h_range=10, s_range=80, v_range=80):
        """
        patch의 접한 픽셀들이 grass_color_hsv와 유사한지 판단.
        """
        grass_h, grass_s, grass_v = self.grass_color

        # 인접 픽셀들 추출
        adjacent_left_pixel = frame[centroid[1], max(centroid[0] - 6, 0)]
        adjacent_right_pixel = frame[centroid[1], min(centroid[0] + 6, frame.shape[1] - 1)]
        adjacent_top_pixel = frame[max(centroid[1] - 6, 0), centroid[0]]
        adjacent_bottom_pixel = frame[min(centroid[1] + 6, frame.shape[0] - 1), centroid[0]]
        
        adjacent_pixels = [adjacent_left_pixel, adjacent_right_pixel, adjacent_top_pixel, adjacent_bottom_pixel]
        
        similar_count = 0
        total_pixels = len(adjacent_pixels)
        
        for pixel_bgr in adjacent_pixels:
            # BGR 픽셀을 HSV로 변환
            pixel_hsv = cv2.cvtColor(pixel_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            
            # HSV 차이 계산
            h_diff = abs(int(pixel_hsv[0]) - grass_h)
            s_diff = abs(int(pixel_hsv[1]) - grass_s)
            v_diff = abs(int(pixel_hsv[2]) - grass_v)
            
            # Hue는 원형이므로 보정
            h_diff = min(h_diff, 180 - h_diff)
            
            # 범위 내에 있는지 확인
            if h_diff <= h_range and s_diff <= s_range and v_diff <= v_range:
                similar_count += 1
        
        grass_ratio = similar_count / total_pixels
        
        print(f"인접 픽셀 잔디 유사도: {grass_ratio:.2f} (기준: 0.5)")
        
        # 50% 이상이면 잔디로 판단하여 등록하지 않음 (True 반환)
        return grass_ratio >= 0.5

        

    def is_color_similar_to_grass(self, patch, h_range=10, s_range=80, v_range=80):
        """
        patch의 픽셀들이 grass_color_hsv와 유사한지 판단.
        patch 내 픽셀 중 35% 이상이 grass 색상 범위에 속하면 True (등록하지 않음), 그렇지 않으면 False를 반환.
        """
        if patch is None or patch.size == 0:
            return False

        patch_hsv = patch.copy()
        grass_h, grass_s, grass_v = self.grass_color

        # HSV 차이 계산
        h_diff = np.abs(patch_hsv[:, :, 0].astype(int) - grass_h)
        s_diff = np.abs(patch_hsv[:, :, 1].astype(int) - grass_s)
        v_diff = np.abs(patch_hsv[:, :, 2].astype(int) - grass_v)

        # Hue는 원형이므로 보정
        h_diff = np.minimum(h_diff, 180 - h_diff)

        # 각 픽셀이 grass 색상 범위 안에 있는지 여부 (불리언 배열)
        # <= 사용하여 경계값 포함
        grass_mask = (h_diff <= h_range) & (s_diff <= s_range) & (v_diff <= v_range)

        # grass-like 픽셀 비율 계산 (올바른 total_pixels 계산)
        total_pixels = patch_hsv.shape[0] * patch_hsv.shape[1]
        grass_pixels = np.count_nonzero(grass_mask)
        grass_ratio = grass_pixels / total_pixels

        if grass_ratio >= 0.2:
            print(f"잔디 색상과 유사한 패치 발견: {grass_ratio:.2f}")
            return True
        else:
            print(f"잔디 색상과 유사하지 않음: {grass_ratio:.2f}")
            return False

        
    def register(self, bbox, frame):
        """새로운 플레이어 등록 (오직 object detection으로만)"""
        # bbox는 (x, y, w, h) 형식
        centroid = get_centroid(bbox)

        print(f"플레이어 등록 시도: centroid {centroid}, bbox {bbox}")

        if self.is_inside_existing_player(centroid):
            return False

        x, y, w, h = bbox
        
        # 중앙 5x5 픽셀 영역 추출
        cx = x + w // 2
        cy = y + h // 2
        
        patch_x1 = max(cx - 4, 0)
        patch_y1 = max(cy - 8, 0)
        patch_x2 = min(cx + 4 + 1, frame.shape[1])
        patch_y2 = min(cy + 8 + 1, frame.shape[0])
        
        center_patch = frame[patch_y1:patch_y2, patch_x1:patch_x2]
        
        # 중앙 패치가 잔디 색상과 유사한지 확인
        if self.is_color_similar_to_grass(center_patch):
            return False
        
        if not self.is_adjacent_to_grass(centroid, frame):
            print(f"중앙 패치가 잔디와 인접하여 등록하지 않음: centroid {centroid}")
            return False
        
        # 템플릿 추출 시 올바른 좌표 사용 (전체 bbox)
        template = frame[y:y + h, x:x + w]  # [y:y+h, x:x+w] 순서 주의

        
        # 중앙 5x5 패치에서 히스토그램 계산
        center_histogram = cv2.calcHist([center_patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,  # (x, y, w, h) 형식 유지
            'template': template.copy(),
            'histogram': center_histogram  
        }
        self.consecutive_failures[self.next_object_id] = 0  # 새로운 객체의 실패 횟수 초기화
        print(f"새 플레이어 등록: ID {self.next_object_id} at {centroid}")
        self.next_object_id += 1
        return True
        

    def deregister(self, object_id):
        """플레이어 등록 해제"""
        print(f"플레이어 제거: ID {object_id}")
        del self.objects[object_id]
        del self.consecutive_failures[object_id]  # 실패 횟수도 함께 제거
        
    def is_inside_existing_player(self, centroid):
        """새 탐지점이 기존 등록된 플레이어 영역 안에 있는지 확인"""
        for obj_id, obj_data in self.objects.items():
            x, y, w, h = obj_data['bbox']  # (x, y, w, h) 형식
            # (x, y, w, h)를 (x1, y1, x2, y2)로 변환
            x1, y1, x2, y2 = x, y, x + w, y + h
            cx, cy = centroid
            
            # 바운딩 박스 내부에 있는지 확인 (약간의 여유 공간 추가)
            margin = 13
            if (x1 - margin <= cx <= x2 + margin and 
                y1 - margin <= cy <= y2 + margin):
                return True
        
        return False
    

        

    def update(self, frame, threshold=0.6, padding=20):
        """매 프레임마다 추적 상태 업데이트 (예측 위치 계산)"""
        self.current_frame += 1
        self.update_with_detections(frame, threshold, padding)
        
            
    def update_with_detections(self, frame, histogram_threshold=0.7, padding=12, dilation_size=2):
        """히스토그램 비교를 사용한 플레이어 추적 업데이트"""
        objects_to_remove = []
        
        for obj_id, obj_data in self.objects.items():
            # 1. 현재 객체 O의 정보 가져오기
            old_bbox = obj_data['bbox']  # (x, y, w, h) 형식
            old_template = obj_data['template']
            old_histogram = obj_data['histogram']
            
            if old_template is None or old_template.size == 0:
                print(f"Object ID {obj_id} has no valid template, skipping.")
                continue
                
            # 2. (x, y, w, h)를 (x1, y1, x2, y2)로 변환하여 padded box 생성
            x, y, w, h = old_bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            frame_h, frame_w = frame.shape[:2]
            
            padded_x1 = max(0, x1 - padding)
            padded_y1 = max(0, y1 - padding)
            padded_x2 = min(frame_w, x2 + padding)
            padded_y2 = min(frame_h, y2 + padding)
            
            # Padded 영역 추출
            search_region = frame[padded_y1:padded_y2, padded_x1:padded_x2]
            
            if search_region.size == 0:
                print(f"Object ID {obj_id} search region is empty, skipping.")
                continue

            # 3. Dilation을 통한 후보 영역 생성
            template_h = 16
            template_w = 8
            
            # 검색 영역에서 가능한 모든 위치에서 히스토그램 비교
            best_score = -1
            best_position = None
            
            # Dilation 크기만큼 스텝으로 검색
            step_size = dilation_size
            
            for search_y in range(0, search_region.shape[0] - template_h + 1, step_size):
                for search_x in range(0, search_region.shape[1] - template_w + 1, step_size):
                    # 후보 영역 추출
                    candidate_region = search_region[search_y:search_y + template_h, 
                                                search_x:search_x + template_w]
                    
                    if candidate_region.shape[:2] != (template_h, template_w):
                        continue
                    
                    # 히스토그램 계산
                    candidate_histogram = cv2.calcHist([candidate_region], [0, 1, 2], None, 
                                                    [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    
                    # 히스토그램 비교 (correlation 방법 사용)
                    correlation = cv2.compareHist(old_histogram, candidate_histogram, 
                                                cv2.HISTCMP_CORREL)
                    
                    if correlation > best_score:
                        best_score = correlation
                        best_position = (search_x, search_y)
            
            # 4. 점수가 threshold보다 높으면 업데이트
            if best_score >= histogram_threshold and best_position is not None:
                search_x, search_y = best_position
                
                # 새로운 위치 계산 (padded 좌표계에서 전체 이미지 좌표계로 변환)
                new_x1 = padded_x1 + search_x
                new_y1 = padded_y1 + search_y
                new_x2 = new_x1 + template_w
                new_y2 = new_y1 + template_h
                
                # (x1, y1, x2, y2)를 (x, y, w, h)로 변환
                new_x = new_x1
                new_y = new_y1
                new_w = new_x2 - new_x1
                new_h = new_y2 - new_y1
                
                # 새로운 중심점 계산
                new_centroid = (new_x + new_w // 2, new_y + new_h // 2)
                new_bbox = (new_x, new_y, new_w, new_h)  # (x, y, w, h) 형식 유지
                
                # 새로운 템플릿과 히스토그램 저장
                new_template = frame[new_y:new_y + new_h, new_x:new_x + new_w]
                new_histogram = cv2.calcHist([new_template], [0, 1, 2], None, 
                                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
                
                if self.is_color_similar_to_grass(new_template):
                    print(f"새로운 템플릿이 잔디 색상과 유사하여 업데이트하지 않음")
                    self.consecutive_failures[obj_id] += 1
                    if self.consecutive_failures[obj_id] >= 5:
                        objects_to_remove.append(obj_id)
                    continue
                
                # 객체 정보 업데이트
                self.objects[obj_id]['centroid'] = new_centroid
                self.objects[obj_id]['bbox'] = new_bbox
                self.objects[obj_id]['quality_score'] = best_score
                self.objects[obj_id]['template'] = new_template.copy()
                #self.objects[obj_id]['histogram'] = new_histogram
                
                # 성공적으로 추적했으므로 실패 횟수 초기화
                self.consecutive_failures[obj_id] = 0
                
            else:
                # 5. 점수가 낮으면 실패 횟수 증가
                self.consecutive_failures[obj_id] += 1
                print(f"Object ID {obj_id} tracking failed: histogram score {best_score:.3f} < threshold {histogram_threshold} (연속 실패: {self.consecutive_failures[obj_id]}/5)")
                
                # 5번 연속 실패 시에만 제거 대상으로 표시
                if self.consecutive_failures[obj_id] >= 5:
                    objects_to_remove.append(obj_id)
        
        # 추적 실패한 객체들 제거
        for obj_id in objects_to_remove:
            self.deregister(obj_id)

                    
    def get_current_boxes(self):
        """현재 추적 중인 모든 플레이어의 정보 반환"""
        boxes = []
        confidences = []
        object_ids = []
        
        for obj_id, obj_data in self.objects.items():
            boxes.append(obj_data['bbox'])
            object_ids.append(obj_id)
            confidences.append(0.5)
            
        return boxes, confidences, object_ids
    