import numpy as np
import logging

logger = logging.getLogger(__name__)

def estimate_proximity(depth_map: np.ndarray, bbox: list) -> dict:
    """
    Extracts the depth of an object. To improve accuracy, we look at the BOTTOM 20%
    of the bounding box, which represents where the object touches the floor.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    h, w = depth_map.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    
    # THE UPGRADE: Look at the bottom 20% of the object (the floor contact point)
    # This prevents tall objects (like people) from being falsely classified as far away.
    y_bottom_start = int(y1 + (y2 - y1) * 0.80)
    roi_depth = depth_map[y_bottom_start:y2, x1:x2]
    
    if roi_depth.size == 0:
        # Fallback to whole box if bottom is too small
        roi_depth = depth_map[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return {"proximity": "UNKNOWN", "raw_depth": None}

    # Use the median of the bottom slice
    object_depth = np.percentile(roi_depth, 50)
    
    if object_depth > 3.0:
        proximity = "NEAR"
    elif object_depth > 1.5:
        proximity = "MID"
    else:
        proximity = "FAR"
        
    return {
        "proximity": proximity,
        "raw_depth": float(object_depth),
    }
        
    logger.info(f"Box: [{x1},{y1},{x2},{y2}] -> Depth: {object_depth:.2f} -> {proximity}")
    
    return {
        "proximity": proximity,
        "raw_depth": float(object_depth),
        "std_dev": float(depth_std)
    }