import logging

logger = logging.getLogger(__name__)

def classify_spatial_zone(bbox: list, image_width: int, image_height: int) -> str:
    """
    Classifies the object into LEFT, CENTER, or RIGHT based on Perspective-Weighted Zones.
    
    WHY: A human walking path is a trapezoid. We dynamically adjust the zone boundaries
    based on the object's vertical position (Y-axis) to accurately model 3D space.
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # 1. Calculate vertical ratio (0.0 = top of image, 1.0 = bottom of image)
    vertical_ratio = center_y / image_height
    
    # 2. Dynamic Boundaries
    # If the object is low (near feet, vertical_ratio ~ 1.0), we shrink the CENTER zone.
    # If the object is high (near horizon, vertical_ratio ~ 0.0), we expand the CENTER zone.
    left_boundary = image_width * (0.25 + 0.10 * (1.0 - vertical_ratio))
    right_boundary = image_width * (0.75 - 0.10 * (1.0 - vertical_ratio))
    
    # 3. Classify
    if center_x < left_boundary:
        zone = "LEFT"
    elif center_x > right_boundary:
        zone = "RIGHT"
    else:
        zone = "CENTER"
        
    return zone