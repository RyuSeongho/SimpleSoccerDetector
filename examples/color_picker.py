import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

def get_dominant_colors(image, mask, n_colors=3, color_space="hsv"):
    """numpy만 사용하여 dominant 색상 추출, color_space 지원"""
    # 마스크 영역의 픽셀만 추출
    masked_pixels = image[mask == 255]
    
    if len(masked_pixels) == 0:
        return []
    
    # 색상 공간을 8단계로 줄여서 계산 속도 향상
    reduced_pixels = (masked_pixels // 32) * 32
    
    # 고유한 색상과 빈도 계산
    unique_colors, counts = np.unique(reduced_pixels.reshape(-1, 3), 
                                     axis=0, return_counts=True)
    
    # 빈도 순으로 정렬
    sorted_indices = np.argsort(counts)[::-1]
    
    result = []
    total_pixels = len(masked_pixels)
    
    for i in range(min(n_colors, len(unique_colors))):
        idx = sorted_indices[i]
        color = unique_colors[idx]
        
        # color_space에 따라 변환
        if color_space.lower() == "rgb":
            color_rgb = tuple(color[::-1])  # BGR to RGB
            color_bgr = tuple(color)
        elif color_space.lower() == "bgr":
            color_bgr = tuple(color)
            color_rgb = tuple(color[::-1])
        elif color_space.lower() == "hsv":
            # BGR to HSV 변환
            color_bgr_np = np.uint8([[color]])
            color_hsv_np = cv2.cvtColor(color_bgr_np, cv2.COLOR_BGR2HSV)
            color_hsv = tuple(int(c) for c in color_hsv_np[0, 0])
            color_bgr = tuple(color)
            color_rgb = tuple(color[::-1])
        else:
            # 기본은 BGR
            color_bgr = tuple(color)
            color_rgb = tuple(color[::-1])
        
        percentage = counts[idx] / total_pixels
        
        color_info = {
            'color_bgr': color_bgr,
            'color_rgb': color_rgb,
            'percentage': percentage
        }
        
        # HSV 색상 공간인 경우 HSV 값도 추가
        if color_space.lower() == "hsv":
            color_info['color_hsv'] = color_hsv
        
        result.append(color_info)
    
    return result

def create_color_bar_numpy(color_info, width=400, height=100):
    """numpy만 사용하여 색상 막대 생성"""
    if not color_info:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    color_bar = np.zeros((height, width, 3), dtype=np.uint8)
    start_x = 0
    
    for info in color_info:
        end_x = start_x + int(info['percentage'] * width)
        color_rgb = info['color_rgb']
        color_bar[:, start_x:end_x, :] = color_rgb
        start_x = end_x
    
    return color_bar

class RealTimeColorDisplay:
    def __init__(self):
        plt.ion()  # 인터랙티브 모드 활성화
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # 색상 막대 표시용
        self.ax1.set_title('Top 3 Dominant Colors')
        self.ax1.set_xlim(0, 400)
        self.ax1.set_ylim(0, 100)
        self.ax1.axis('off')
        
        # 색상 정보 텍스트 표시용
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.axis('off')
        
    def update_display(self, color_info, color_space="hsv"):
        # 이전 플롯 지우기
        self.ax1.clear()
        self.ax2.clear()
        
        # 축 설정 재적용
        self.ax1.set_title(f'Top 3 Dominant Colors (Real-time) - {color_space.upper()}')
        self.ax1.set_xlim(0, 400)
        self.ax1.set_ylim(0, 100)
        self.ax1.axis('off')
        
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.axis('off')
        
        if color_info:
            # 색상 막대 생성
            color_bar = create_color_bar_numpy(color_info)
            
            # matplotlib에서 표시 (RGB 형태로 변환)
            color_bar_rgb = color_bar / 255.0  # 0-1 범위로 정규화
            self.ax1.imshow(color_bar_rgb, aspect='auto', extent=[0, 400, 0, 100])
            
            # 색상 정보 텍스트 표시 (color_space에 따라)
            text_info = ""
            for i, info in enumerate(color_info):
                if color_space.lower() == "rgb":
                    color_val = info['color_rgb']
                    text_info += f"Color {i+1}: RGB{color_val} - {info['percentage']:.1%}\n"
                elif color_space.lower() == "bgr":
                    color_val = info['color_bgr']
                    text_info += f"Color {i+1}: BGR{color_val} - {info['percentage']:.1%}\n"
                elif color_space.lower() == "hsv":
                    color_val = info.get('color_hsv', info['color_bgr'])
                    text_info += f"Color {i+1}: HSV{color_val} - {info['percentage']:.1%}\n"
                else:
                    color_val = info['color_bgr']
                    text_info += f"Color {i+1}: BGR{color_val} - {info['percentage']:.1%}\n"
            
            self.ax2.text(0.05, 0.8, text_info, fontsize=12, 
                         verticalalignment='top', fontfamily='monospace')
        
        # 화면 업데이트
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def integrate_realtime_colors(frame, combined_mask, color_display, color_space="hsv"):
    """
    기존 process_video 함수에 추가할 수 있는 함수.
    color_space에 따라 지정된 형식으로 dominant color 반환
    
    Args:
        frame: 원본 프레임 (BGR)
        combined_mask: 마스크
        color_display: RealTimeColorDisplay 객체
        color_space: "rgb", "bgr", "hsv" 중 하나
    
    Returns:
        지정된 color_space의 가장 우세한 색상 튜플
    """
    
    # Top 3 dominant colors 추출
    dominant_colors = get_dominant_colors(frame, combined_mask, n_colors=3, color_space=color_space)
    
    # 실시간 display 업데이트
    color_display.update_display(dominant_colors, color_space)
    
    # 콘솔에도 출력 (지정된 color_space 기준)
    if dominant_colors:
        print(f"Top 3 Colors ({color_space.upper()}):")
        for i, color in enumerate(dominant_colors):
            if color_space.lower() == "rgb":
                color_val = color['color_rgb']
            elif color_space.lower() == "bgr":
                color_val = color['color_bgr']
            elif color_space.lower() == "hsv":
                color_val = color.get('color_hsv', color['color_bgr'])
            else:
                color_val = color['color_bgr']
            
            print(f"  {i+1}. {color_space.upper()}{color_val} - {color['percentage']:.1%}")
    
    # dominant_colors가 비어있으면 None 반환
    if not dominant_colors:
        return None
    
    # 가장 우세한 색상을 지정된 color_space로 반환
    dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
    top_color = dominant_colors[0]
    
    if color_space.lower() == "rgb":
        return top_color['color_rgb']
    elif color_space.lower() == "bgr":
        return top_color['color_bgr']
    elif color_space.lower() == "hsv":
        return top_color.get('color_hsv', top_color['color_bgr'])
    else:
        # 기본은 BGR
        return top_color['color_bgr']


# 사용 예제
if __name__ == "__main__":
    # 실시간 색상 display 객체 생성
    color_display = RealTimeColorDisplay()
    
    # 기존 process_video 함수에서 다음과 같이 사용:
    # dominant_colors = integrate_realtime_colors(frame, combined_mask, color_display)
