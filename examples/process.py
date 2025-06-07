import cv2
import numpy as np
from video_uniform import process_video_uniform, process_video_uniform_pxCount, process_video_uniform_pxCount_70
from ball import detect_ball
from blur import apply_blur, remove_spectator_region
from color_picker import get_dominant_colors, RealTimeColorDisplay, integrate_realtime_colors
from compactness_filter import remove_low_compactness
import watershed as ws
from object_detection import extract_player_bounding_boxes, merge_nearby_boxes, draw_player_boxes
import matplotlib
matplotlib.use("TkAgg")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    color_display = RealTimeColorDisplay()
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        index += 1

        frame = cv2.resize(frame, (640, 360))
        cv2.imshow("original", frame)
        
        # 색상 분석 (선택사항)
        #all 1 mask
        if(index == 1):
            all_mask = np.ones_like(frame, dtype=np.uint8) * 255
            dominant_colors = integrate_realtime_colors(frame, all_mask, color_display)
        

        # HSV 변환 및 초록색 제거
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #lower_green = np.array([30, 20, 20])
        #upper_green = np.array([90, 255, 255])
        #mask_green = cv2.inRange(hsv, lower_green, upper_green)

        print("Dominant Colors:", dominant_colors)

        # dominant color 범위로 초록색 마스크 생성
        lower_green = np.array([dominant_colors[0] - 40,
                                dominant_colors[1] - 120,
                                dominant_colors[2] - 120])
        upper_green = np.array([dominant_colors[0] + 40,
                                dominant_colors[1] + 120,
                                dominant_colors[2] + 120])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        mask_not_green = cv2.bitwise_not(mask_green)

        # 초록색 제거된 부분에서 엣지 검출
        roi = cv2.bitwise_and(frame, frame, mask=mask_not_green)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobel_x, sobel_y)

        median_val = np.median(sobel)
        lower = int(max(0, 0.1 * median_val))
        upper = int(min(255, 5.0 * median_val))
        sobel[sobel < lower] = 0
        sobel[sobel > upper] = 255

        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
        result = cv2.morphologyEx(np.uint8(sobel), cv2.MORPH_CLOSE, kernel_merge)
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_noise)

        # 윤곽선 추출
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_person_shape = np.zeros(result.shape, dtype=np.uint8)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, x + w, y + h])
                cv2.drawContours(mask_person_shape, [cnt], -1, 255, thickness=cv2.FILLED)

        # 윤곽선 추출, contours → mask_person_shape 생성 (unchanged)

        # Remove spectator area using the helper function
        mask_person_shape, cutoff_rows = remove_spectator_region(mask_person_shape, 4)
        def boxes_overlap(box1, box2, threshold=30):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            center1 = ((x1_min + x1_max) // 2, (y1_min + y1_max) // 2)
            center2 = ((x2_min + x2_max) // 2, (y2_min + y2_max) // 2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            return distance < threshold

        merged = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            group = [box1]
            for j, box2 in enumerate(boxes):
                if j != i and j not in used and boxes_overlap(box1, box2):
                    group.append(box2)
                    used.add(j)
            used.add(i)

            x_min = min(b[0] for b in group)
            y_min = min(b[1] for b in group)
            x_max = max(b[2] for b in group)
            y_max = max(b[3] for b in group)

        ball_mask = detect_ball(frame)
        blur_mask = apply_blur(frame, lower_green = lower_green, upper_green = upper_green)
        cv2.imshow("White Near Green", ball_mask)
        cv2.imshow("Blurred Frame", blur_mask)

        cv2.imshow("Mask Person Shape", mask_person_shape)
        cv2.imshow("blur_mask", blur_mask)
        cv2.imshow("ball_mask", ball_mask)
        
        combined_mask_1 = cv2.bitwise_and(blur_mask, ball_mask)
        combined_mask_1 = remove_low_compactness(combined_mask_1, compactness_threshold=0.3)
        cv2.imshow("Combined Mask", combined_mask_1)

        combined_mask_2 = cv2.bitwise_and(mask_person_shape, blur_mask)
        cv2.imshow("Combined Mask 2", combined_mask_2)
        
        # 최종 병합 마스크
        combined_mask = cv2.bitwise_or(combined_mask_2, combined_mask_1)
        # Mask spectator region (above cutoff_row) to remove audience area
        for slice_idx, cutoff_row in enumerate(cutoff_rows):
            # 한 슬라이스의 가로 범위를 계산
            slice_width = frame.shape[1] // len(cutoff_rows)
            start_col = slice_idx * slice_width
            # 마지막 슬라이스는 남는 영역을 모두 포함
            end_col = (slice_idx + 1) * slice_width if slice_idx < len(cutoff_rows) - 1 else frame.shape[1]
            # 해당 슬라이스 구간에서, cutoff_row 위쪽을 모두 0(검은색)으로 마스킹
            mask_person_shape[:cutoff_row, start_col:end_col] = 0
        
        # **선수 영역 bounding box 추출**
        player_boxes = extract_player_bounding_boxes(combined_mask)
        #player_boxes = fast_object_detection_connected_components(combined_mask)
        #player_boxes = ws.improved_watershed(combined_mask, min_connection_width=3)  
        merged_boxes = merge_nearby_boxes(player_boxes, merge_threshold=25)
        
        # frame에 combined_mask를 적용하여 최종 결과 생성
        final_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        combined_colored = cv2.bitwise_and(frame, final_mask)
        
        # **선수 bounding box를 combined_colored에 그리기**
        combined_colored_with_boxes = draw_player_boxes(combined_colored, merged_boxes, 
                                                       color=(0, 255, 255), thickness=2)
        
        # 원본 프레임에도 bounding box 표시
        frame_with_boxes = draw_player_boxes(frame, merged_boxes, 
                                           color=(255, 0, 0), thickness=2)
        
        # Mask spectator region (above cutoff_row) to remove audience area
        combined_mask[:cutoff_row, :] = 0
        
        # 정보 표시
        cv2.rectangle(combined_colored_with_boxes, (0, 0), (640, 360), (0, 255, 0), 2)
        cv2.putText(combined_colored_with_boxes, f"Players Detected: {len(merged_boxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_colored_with_boxes, "Final Result with Bounding Boxes", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 화면 출력
        cv2.imshow("Final Result with Boxes", combined_colored_with_boxes)
        cv2.imshow("Original with Boxes", frame_with_boxes)

        # 유니폼 기반 추출 결과
        # edges_uniform = process_video_uniform2_imp(frame, merged_boxes)
        # cv2.imshow("original3", edges_uniform)  # 바운딩 박스가 그려진 원본

        # process_video_uniform2_imp(frame, merged_boxes)
        # cv2.imshow("masked_teams", frame)
        #cv2.imshow("pxCount_masked_teams", frame)

        process_video_uniform_pxCount_70(frame, merged_boxes)
        cv2.imshow("pxCount_masked_teams", frame)

        # 선수 정보 출력
        if merged_boxes:
            print(f"Frame: {len(merged_boxes)} players detected")
            for i, (x, y, w, h) in enumerate(merged_boxes):
                print(f"  Player {i+1}: ({x}, {y}) - {w}x{h}")

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("soccer_video2.mp4")
