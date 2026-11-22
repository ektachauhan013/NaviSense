import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_free_space(depth_map: np.ndarray) -> dict:
    h, w = depth_map.shape
    y_start = int(h * 0.60)
    y_end = h
    
    c_x_start, c_x_end = int(w * 0.35), int(w * 0.65)
    l_x_start, l_x_end = int(w * 0.05), int(w * 0.30)
    r_x_start, r_x_end = int(w * 0.70), int(w * 0.95)
    
    center_roi = depth_map[y_start:y_end, c_x_start:c_x_end]
    left_roi = depth_map[y_start:y_end, l_x_start:l_x_end]
    right_roi = depth_map[y_start:y_end, r_x_start:r_x_end]

    center_depth = np.percentile(center_roi, 50) if center_roi.size > 0 else 0.0
    left_depth = np.percentile(left_roi, 50) if left_roi.size > 0 else 0.0
    right_depth = np.percentile(right_roi, 50) if right_roi.size > 0 else 0.0

    # STRICTER THRESHOLD: 3.2 instead of 4.0
    THRESHOLD = 3.2
    
    return {
        "center_clear": bool(center_depth < THRESHOLD),
        "left_clear": bool(left_depth < THRESHOLD),
        "right_clear": bool(right_depth < THRESHOLD),
        "depths": {
            "center": float(center_depth),
            "left": float(left_depth),
            "right": float(right_depth)
        }
    }