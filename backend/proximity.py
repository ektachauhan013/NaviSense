import numpy as np
import logging

logger = logging.getLogger(__name__)

def estimate_proximity(depth_map: np.ndarray, bbox: list) -> dict:
    """
    Extracts the depth of an object inside a bounding box and classifies its proximity.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # 1. Boundary Check: Ensure coordinates are within the image
    h, w = depth_map.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    
    # 2. ROI (Region of Interest) Extraction
    roi_depth = depth_map[y1:y2, x1:x2]
    
    # If the box is too small or empty, return an unknown state
    if roi_depth.size == 0:
        return {"proximity": "UNKNOWN", "raw_depth": None, "confidence": 0.0}

    # 3. Aggregation: We use the 50th percentile (Median)
    object_depth = np.percentile(roi_depth, 50)
    
    # Calculate consistency
    depth_std = np.std(roi_depth)
    
    # 4. Rule-Based Classification
    if object_depth > 3.0:
        proximity = "NEAR"
    elif object_depth > 1.5:
        proximity = "MID"
    else:
        proximity = "FAR"
        
    logger.info(f"Box: [{x1},{y1},{x2},{y2}] -> Depth: {object_depth:.2f} -> {proximity}")
    
    return {
        "proximity": proximity,
        "raw_depth": float(object_depth),
        "std_dev": float(depth_std)
    }