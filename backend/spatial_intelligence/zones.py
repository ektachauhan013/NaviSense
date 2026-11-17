import logging

logger = logging.getLogger(__name__)

def classify_spatial_zone(bbox: list, image_width: int) -> str:
    """
    Determines which horizontal zone (LEFT, CENTER, RIGHT) the bounding box is in.
    """
    x1, y1, x2, y2 = bbox
    
    # 1. Calculate the center x-coordinate of the bounding box
    center_x = (x1 + x2) / 2.0
    
    # 2. Define the boundaries (33% and 66% of the image width)
    left_boundary = image_width / 3.0
    right_boundary = (image_width / 3.0) * 2.0
    
    # 3. Classify the zone
    if center_x < left_boundary:
        zone = "LEFT"
    elif center_x > right_boundary:
        zone = "RIGHT"
    else:
        zone = "CENTER"
        
    logger.info(f"Box center X: {center_x:.1f} -> Zone: {zone}")
    
    return zone