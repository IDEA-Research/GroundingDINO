"""螢幕 bbox 工具：座標一律為相對於畫面大小的 [x0, y0, x1, y1]（0~1）。"""

SAME_SCREEN_IOU = 0.3
SAME_SCREEN_CONTAINMENT = 0.5


def valid_bbox(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def cxcywh_to_xyxy(box):
    """GroundingDINO 的正規化 cxcywh 轉為正規化 xyxy，並限制在畫面內。"""
    cx, cy, w, h = (float(value) for value in box)
    coords = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return [round(min(1.0, max(0.0, value)), 4) for value in coords]


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0 and height > 0 else 0.0


def iou(a, b):
    a, b = valid_bbox(a), valid_bbox(b)
    if a is None or b is None:
        return 0.0
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a, b):
    """交集佔較小框面積的比例。"""
    a, b = valid_bbox(a), valid_bbox(b)
    if a is None or b is None:
        return 0.0
    smaller = min(_area(a), _area(b))
    return _intersection(a, b) / smaller if smaller > 0 else 0.0


def same_screen(a, b):
    """兩個框是否為同一台設備；任一方沒有座標時無法區分，視為同一台。"""
    if valid_bbox(a) is None or valid_bbox(b) is None:
        return True
    return iou(a, b) > SAME_SCREEN_IOU or containment(a, b) > SAME_SCREEN_CONTAINMENT
