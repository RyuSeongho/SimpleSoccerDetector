import cv2
import numpy as np

class Uniform:
    """
    팀 유니폼 색상을 객체로 캡슐화.
    """
    def __init__(self, name: str, bgr_color: np.ndarray):
        self.name = name
        # float32로 변환 (거리 계산 시 활용)
        self.color = bgr_color.astype(np.float32)

    def distance(self, avg_color: np.ndarray) -> float:
        """
        ROI 평균 색과 이 유니폼 색상 간의 유클리디언 거리 반환.
        """
        return np.linalg.norm(avg_color - self.color)


class UniformClassifier:
    """
    프레임과 미리 정의된 Uniform 목록으로부터 선수 박스별 팀 라벨을 그려줌.
    """
    def __init__(self, uniforms: list[Uniform], dist_thresh: float = None):
        self.uniforms = uniforms
        self.dist_thresh = dist_thresh  # 거리 임계값(넘으면 Unknown)

    @staticmethod
    def mask_grass(hsv_roi: np.ndarray) -> np.ndarray:
        """
        HSV 공간에서 잔디에 해당하는 녹색 영역 마스크 생성.
        """
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        return cv2.inRange(hsv_roi, lower_green, upper_green)

    @staticmethod
    def dominant_color_bgr(bgr_roi: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        mask==0(잔디 아님) 영역의 평균 BGR 계산.
        """
        if mask is not None:
            sel = (mask == 0)
            if not np.any(sel):
                return np.zeros(3, dtype=np.float32)
            vals = bgr_roi[sel]
        else:
            vals = bgr_roi.reshape(-1, 3)
        return vals.mean(axis=0)

    def pick_closest(self, avg_color: np.ndarray) -> Uniform:
        """
        avg_color에 대해 가장 거리 가까운 Uniform 반환.
        """
        return min(self.uniforms, key=lambda u: u.distance(avg_color))

    def classify_and_draw(self, frame: np.ndarray, detections: list[tuple[int,int,int,int]]) -> np.ndarray:
        """
        frame: BGR 이미지
        detections: (x,y,w,h) 박스 리스트
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for (x, y, w, h) in detections:
            roi_bgr = frame[y:y+h, x:x+w]
            roi_hsv = hsv[y:y+h, x:x+w]

            grass_mask = self.mask_grass(roi_hsv)
            avg_bgr = self.dominant_color_bgr(roi_bgr, grass_mask)

            best = self.pick_closest(avg_bgr)
            dist = best.distance(avg_bgr)

            # 임계값 지정 시, 거리가 크면 Unknown 처리
            if self.dist_thresh is not None and dist > self.dist_thresh:
                label = "Unknown"
                box_color = (0, 0, 0)
            else:
                label = best.name
                box_color = tuple(int(c) for c in best.color)

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        return frame


# ─────────────────────────────────────────────
# 빨간 박스 동적 탐지 함수
# lower_red1=(0, 70, 50), upper_red1=(10, 255, 255),
# lower_red2=(170, 70, 50), upper_red2=(180, 255, 255),
# ─────────────────────────────────────────────
def detect_red_boxes(frame: np.ndarray,
                    lower_red1=(0, 60, 60), upper_red1=(15, 255, 255),
                    lower_red2=(170, 240, 240), upper_red2=(180, 255, 255),
                    area_thresh=1500) -> list[tuple[int,int,int,int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(lower_red1), np.array(upper_red1))
    mask2 = cv2.inRange(hsv, np.array(lower_red2), np.array(upper_red2))
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= area_thresh:
            boxes.append((x, y, w, h))
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = "image.png"
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

    # 1) 후보 영역 검출
    detections = detect_red_boxes(frame)

    # 2) 사전 정의된 유니폼 색상
    red_team   = Uniform("RedTeam",   np.array([ 247,  223, 229], dtype=np.float32))
    blue_team  = Uniform("BlueTeam",  np.array([ 47,  53,  72], dtype=np.float32))
    #green_team = Uniform("GreenTeam", np.array([ 68, 113,  64], dtype=np.float32))
    uniforms = [red_team, blue_team]
    #uniforms = [red_team, blue_team, green_team]

    # Optional: 색상 매칭 임계값 설정 (None이면 사용 안 함)
    DIST_THRESH = 150

    # 3) 분류 및 시각화
    classifier = UniformClassifier(uniforms, dist_thresh=DIST_THRESH)
    result = classifier.classify_and_draw(frame, detections)

    # 4) 결과 출력
    try:
        from google.colab.patches import cv2_imshow
        cv2_imshow(result)
    except ImportError:
        cv2.imshow("Classified", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
