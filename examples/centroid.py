def get_centroid(box):
    """
    바운딩 박스의 중심 좌표를 계산
    
    Args:
        box: (x, y, w, h) 형태의 바운딩 박스
    
    Returns:
        (cx, cy): 중심 좌표
    """
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy