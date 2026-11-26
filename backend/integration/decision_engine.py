import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# --- THRESHOLDS FOR SAFETY STATE MACHINE ---
UNCERTAINTY_VARIANCE_THRESHOLD = 0.15 # If depth variance is below this, it's a flat surface (wall/glass)
UNCERTAINTY_DEPTH_THRESHOLD = 2.5     # ...but only if it's somewhat close.
COOLDOWN_SECONDS = 2.5                # Wait 2.5s before speaking the same warning again

class DecisionEngine:
    """
    The final safety arbiter. Resolves conflicts between EC's perception (risk/motion) 
    and EY's spatial intelligence (proximity/free space).
    """
    def __init__(self):
        self.last_state = "CLEAR"
        self.last_advice = ""
        self.last_spoken_time = 0.0

    def _check_uncertainty(self, depth_map: np.ndarray) -> bool:
        """
        Low-Texture Failsafe: Detects blank walls, glass doors, or blocked cameras.
        If the center of the image has almost zero depth variation, we are likely facing a flat surface.
        """
        h, w = depth_map.shape
        # Look at the center 20% of the image
        cy1, cy2 = int(h * 0.40), int(h * 0.60)
        cx1, cx2 = int(w * 0.40), int(w * 0.60)
        
        center_roi = depth_map[cy1:cy2, cx1:cx2]
        if center_roi.size == 0:
            return True
            
        depth_variance = np.var(center_roi)
        median_depth = np.median(center_roi)
        
        # If variance is super low AND it's close, it's a wall/glass
        if depth_variance < UNCERTAINTY_VARIANCE_THRESHOLD and median_depth > UNCERTAINTY_DEPTH_THRESHOLD:
            logger.warning(f"Uncertainty Failsafe Triggered: Var={depth_variance:.2f}, Depth={median_depth:.2f}")
            return True
        return False

    def evaluate(
        self, 
        ranked_objects: List[Dict[str, Any]], 
        free_space_result: Dict[str, Any],
        depth_map: np.ndarray
    ) -> Dict[str, Any]:
        """
        Takes the perception and spatial data, and outputs a final safe decision.
        """
        state = "CLEAR"
        advice = "Path ahead appears clear. Continue forward."
        reason = "Default clear"

        # 1. FAILSAFE: Blank Wall / Glass / Blocked Camera
        if self._check_uncertainty(depth_map):
            state = "STOP"
            advice = "Stop. Uncertain visual path. Possible flat surface or blocked camera."
            reason = "Uncertainty_Override"
            
        else:
            # 2. HARD SAFETY PRECEDENCE MATRIX
            center_blocked = not free_space_result.get("center_clear", True)
            
            # Find highest risk object
            top_threat = ranked_objects[0] if ranked_objects else None
            
            # Rule A: Approaching High-Risk Object in Center -> IMMEDIATE STOP
            if top_threat and top_threat["zone"] == "CENTER" and top_threat["risk_level"] == "HIGH":
                state = "STOP"
                advice = f"Stop immediately. {top_threat['label'].capitalize()} approaching directly ahead."
                reason = "Approaching_Center_Threat"
                
            # Rule B: Center path blocked, but sides are open -> CAUTION
            elif center_blocked:
                if free_space_result.get("left_clear", False) and not free_space_result.get("right_clear", False):
                    state = "CAUTION"
                    advice = "Obstacle directly ahead. Steer left immediately."
                    reason = "Center_Blocked_Steer_Left"
                elif free_space_result.get("right_clear", False) and not free_space_result.get("left_clear", False):
                    state = "CAUTION"
                    advice = "Obstacle directly ahead. Steer right immediately."
                    reason = "Center_Blocked_Steer_Right"
                elif free_space_result.get("left_clear", False) and free_space_result.get("right_clear", False):
                    state = "CAUTION"
                    advice = "Obstacle directly ahead. Steer left or right to go around."
                    reason = "Center_Blocked_Both_Sides_Open"
                else:
                    state = "STOP"
                    advice = "Stop. Obstacle directly ahead. No clear path around. Turn around."
                    reason = "Center_Blocked_No_Exit"
                    
            # Rule C: Center is clear, but there are near objects on the sides
            else:
                near_objects = [o for o in ranked_objects if o["proximity"] == "NEAR"]
                near_left = any(o["zone"] == "LEFT" for o in near_objects)
                near_right = any(o["zone"] == "RIGHT" for o in near_objects)
                
                if near_left and near_right:
                    state = "CAUTION"
                    advice = "Caution. Obstacles close on both left and right. Proceed straight."
                    reason = "Side_Threats_Both"
                elif near_left:
                    state = "CAUTION"
                    advice = "Path ahead clear, but obstacle close on your left. Stay right."
                    reason = "Side_Threat_Left"
                elif near_right:
                    state = "CAUTION"
                    advice = "Path ahead clear, but obstacle close on your right. Stay left."
                    reason = "Side_Threat_Right"

        # 3. COOLDOWN & HYSTERESIS (Prevent audio spam)
        now = time.time()
        escalated = (
            (self.last_state == "CLEAR" and state in ("CAUTION", "STOP")) or
            (self.last_state == "CAUTION" and state == "STOP")
        )
        
        if advice == self.last_advice and not escalated:
            if (now - self.last_spoken_time) < COOLDOWN_SECONDS:
                # Suppress the advice (return empty string) because it's on cooldown
                final_advice = ""
            else:
                final_advice = advice
                self.last_spoken_time = now
        else:
            # New advice or escalated danger -> speak immediately
            final_advice = advice
            self.last_spoken_time = now
            self.last_state = state

        if final_advice:
            self.last_advice = advice
            logger.info(f"Decision: [{state}] {final_advice} (Reason: {reason})")

        return {
            "state": state,
            "advice": final_advice,
            "reason": reason,
            "suppressed": final_advice == ""
        }