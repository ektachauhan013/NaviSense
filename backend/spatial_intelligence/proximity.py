import numpy as np
import logging

logger = logging.getLogger(__name__)

def estimate_proximity(depth_map: np.ndarray, bbox: list) -> dict:
    """
    Estimates the distance of an object using Object-Centric Depth Sampling.
    
    WHY: Bounding boxes often contain background pixels. By cropping the inner core 
    and looking at the bottom, we get the true ground-contact distance of the object.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = depth_map.shape
    
    # 1. Boundary Check: Ensure coordinates are within the image
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    
    # 2. Inner Core Crop: Ignore the outer 20% on all sides.
    # WHY? The edges of a bounding box are most likely to contain background noise.
    box_w, box_h = x2 - x1, y2 - y1
    cx1 = int(x1 + box_w * 0.20)
    cx2 = int(x2 - box_w * 0.20)
    cy1 = int(y1 + box_h * 0.20)
    cy2 = int(y2 - box_h * 0.20)
    
    # Fallback if the box is too small to crop
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cx2, cy1, cy2 = x1, x2, y1, y2

    # 3. Ground Contact: Look at the bottom 30% of the core crop.
    # WHY? A person's head is far away, but their feet are close. Ground contact is the best depth indicator.
    y_bottom_start = int(cy1 + (cy2 - cy1) * 0.70)
    roi_depth = depth_map[y_bottom_start:cy2, cx1:cx2]
    
    if roi_depth.size == 0:
        return {"proximity": "UNKNOWN", "raw_depth": None}

    # 4. Use 80th percentile (favors the closer pixels in the region).
    # WHY? If any part of the object is close, we want to know about it for safety.
    object_depth = np.percentile(roi_depth, 80)
    
    # 5. Rule-Based Classification
    if object_depth > 3.0:
        proximity = "NEAR"
    elif object_depth > 1.5:
        proximity = "MID"
    else:
        proximity = "FAR"
        
    return {"proximity": proximity, "raw_depth": float(object_depth)}