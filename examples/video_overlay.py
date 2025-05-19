import cv2
import numpy as np
import time

def detect_players(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    fg_mask = cv2.bitwise_not(green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    h_img, w_img = frame.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 500 or area > 0.1 * w_img * h_img:
            continue
        aspect = w / float(h)
        if aspect < 0.3 or aspect > 1.2:
            continue
        bboxes.append((x, y, w, h))
    return bboxes, clean

def play_with_overlay(input_path, output_path=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    # 비디오 저장이 필요하면 VideoWriter 설정
    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps      = cap.get(cv2.CAP_PROP_FPS)
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer   = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, mask = detect_players(frame)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

        # FPS 표시
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # 화면 출력
        cv2.imshow("Player Detection", frame)

        # 파일로 저장
        if writer:
            writer.write(frame)

        # 종료 조건: 'q' 키
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 입력 비디오 파일 경로
    input_mp4  = "soccer_match.mp4"
    # 저장하려면 파일명 지정, 아니면 None
    output_mp4 = "annotated_match.mp4"
    play_with_overlay(input_mp4, output_mp4)
