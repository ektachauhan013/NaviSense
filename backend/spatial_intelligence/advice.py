import logging

logger = logging.getLogger(__name__)

def generate_advice(free_space_result: dict, objects: list) -> str:
    center_clear = free_space_result["center_clear"]
    left_clear = free_space_result["left_clear"]
    right_clear = free_space_result["right_clear"]

    # 1. HARD STOP: If center is blocked, find an alternate route or stop.
    if not center_clear:
        if left_clear and right_clear:
            return "Obstacle directly ahead. Steer left or right to go around."
        elif left_clear and not right_clear:
            return "Obstacle directly ahead. Steer left immediately."
        elif right_clear and not left_clear:
            return "Obstacle directly ahead. Steer right immediately."
        else:
            return "STOP. Obstacle directly ahead. No clear path around. Turn around."

    # 2. Check for HIGH risk objects in the center
    near_center_danger = any(obj["zone"] == "CENTER" and obj["proximity"] == "NEAR" for obj in objects)
    if near_center_danger:
        return "STOP. Object extremely close directly in front of you."

    # 3. Check side objects
    near_left = any(obj["zone"] == "LEFT" and obj["proximity"] == "NEAR" for obj in objects)
    near_right = any(obj["zone"] == "RIGHT" and obj["proximity"] == "NEAR" for obj in objects)

    if near_left and near_right:
        return "Caution. Obstacles close on both left and right. Proceed straight."
    if near_left:
        return "Path ahead clear, but obstacle close on your left. Stay right."
    if near_right:
        return "Path ahead clear, but obstacle close on your right. Stay left."

    return "Path ahead appears clear. Continue forward."