import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_free_space(depth_map: np.ndarray) -> dict:
    """
    Analyzes the bottom-center of the image (the path of travel) to see if it is clear.
    
    Args:
        depth_map: A 2D NumPy array of relative depth values.
        
    Returns:
        A dictionary with a boolean 'is_clear' and the 'path_depth'.
    """
    h, w = depth_map.shape
    
    # 1. Define the Path of Travel ROI (Region of Interest)
    # We look at the bottom 40% of the image (the floor)
    # We look at the center 40% of the width (directly ahead)
    y_start = int(h * 0.60)
    y_end = h
    x_start = int(w * 0.30)
    x_end = int(w * 0.70)
    
    path_roi = depth_map[y_start:y_end, x_start:x_end]
    
    if path_roi.size == 0:
        return {"is_clear": False, "path_depth": None}

    # 2. Calculate the median depth of the floor ahead
    path_depth = np.percentile(path_roi, 50)
    
    # 3. Rule-based threshold. 
    # If the path depth is HIGH (e.g., > 3.5), it means something is very close.
    if path_depth > 3.5:
        is_clear = False
        logger.info(f"Path BLOCKED. Depth: {path_depth:.2f}")
    else:
        is_clear = True
        logger.info(f"Path CLEAR. Depth: {path_depth:.2f}")
        
    return {
        "is_clear": is_clear,
        "path_depth": float(path_depth)
    }