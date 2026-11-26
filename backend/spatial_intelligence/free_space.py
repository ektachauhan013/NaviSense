import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_free_space(depth_map: np.ndarray) -> dict:
    """
    Analyzes free space using a Continuous 1D Clearance Profile.
    
    WHY: By dividing the floor into 15 thin strips, we can detect narrow gaps 
    and accurately determine if there is a navigable path forward.
    """
    h, w = depth_map.shape
    y_start, y_end = int(h * 0.60), h  # Look at the bottom 40% (the floor)
    
    # 1. Divide the horizontal field of view into 15 vertical strips
    num_strips = 15
    strip_width = w // num_strips
    THRESHOLD = 3.2 # If depth is higher than this, it's blocked
    
    clear_strips = []
    depths = []
    
    # 2. Evaluate each strip individually
    for i in range(num_strips):
        sx1 = i * strip_width
        sx2 = (i + 1) * strip_width
        roi = depth_map[y_start:y_end, sx1:sx2]
        
        if roi.size == 0:
            clear_strips.append(False)
            depths.append(0.0)
            continue
            
        depth = np.percentile(roi, 50) # Median depth of the strip
        depths.append(float(depth))
        clear_strips.append(bool(depth < THRESHOLD))

    # 3. Determine Center, Left, Right clearance
    # Center is strips 5 to 10 (the middle third). ALL strips must be clear to walk straight.
    center_clear = all(clear_strips[5:11])
    
    # Left is strips 0 to 4, Right is strips 11 to 14. 
    # ANY clear strip means there is an escape route on that side.
    left_clear = any(clear_strips[0:5])
    right_clear = any(clear_strips[11:15])

    return {
        "center_clear": center_clear,
        "left_clear": left_clear,
        "right_clear": right_clear,
        "depths": {
            "center": float(np.mean(depths[5:11])),
            "left": float(np.mean(depths[0:5])),
            "right": float(np.mean(depths[11:15]))
        }
    }