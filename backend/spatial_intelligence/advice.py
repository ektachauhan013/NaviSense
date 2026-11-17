import logging

logger = logging.getLogger(__name__)

def generate_advice(free_space_result: dict, objects: list) -> str:
    """
    Takes the free space analysis and a list of detected objects (with proximity and zone)
    and generates a human-readable advice string.
    """
    advice_parts = []
    
    # 1. Assess the path of travel (most important)
    if not free_space_result["is_clear"]:
        advice_parts.append("Stop. Obstacle directly ahead.")
    else:
        advice_parts.append("Path ahead appears clear.")

    # 2. Assess the surrounding objects
    # We prioritize objects that are NEAR.
    near_objects = [obj for obj in objects if obj["proximity"] == "NEAR"]
    
    for obj in near_objects:
        zone = obj["zone"]
        if zone == "LEFT":
            advice_parts.append("Obstacle very close on your left.")
        elif zone == "RIGHT":
            advice_parts.append("Obstacle very close on your right.")
        elif zone == "CENTER":
            # Center is handled by free_space mostly, but just in case
            advice_parts.append("Obstacle very close in front of you.")

    # 3. If path is clear and no near objects, give a positive reinforcement
    if free_space_result["is_clear"] and len(near_objects) == 0:
        advice_parts.append("Continue forward.")

    # Combine into one string
    final_advice = " ".join(advice_parts)
    
    logger.info(f"Generated Advice: {final_advice}")
    return final_advice