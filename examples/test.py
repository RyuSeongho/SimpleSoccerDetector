import cv2
import numpy as np
from video_uniform import process_video_uniform

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        cv2.imshow("original", frame)

        # HSV 변환 및 초록색 제거
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 20, 20])
        upper_green = np.array([90, 255, 255])
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

        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))  # 수직으로 긴 커널
        result = cv2.morphologyEx(np.uint8(sobel), cv2.MORPH_CLOSE, kernel_merge)
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_noise)

        # 윤곽선 추출
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_person_shape = np.zeros(result.shape, dtype=np.uint8)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area :  # 너무 작은/큰 노이즈 제거
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, x + w, y + h])
                cv2.drawContours(mask_person_shape, [cnt], -1, 255, thickness=cv2.FILLED)

        # 윤곽 box 병합 (가까운 박스 병합)
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
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        # 유니폼 기반 추출 결과
        edges_uniform = process_video_uniform(frame)

        # 최종 병합 및 출력
        combined_mask = cv2.bitwise_or(mask_person_shape, edges_uniform)
        combined_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Final Edge Detection", combined_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("soccer_video.mp4")
