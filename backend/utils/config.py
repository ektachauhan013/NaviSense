# backend/utils/config.py

# --- DEPTH ESTIMATOR (EMA) CONSTANTS ---
ALPHA_MOVING = 0.8   # Trust new frame more when user is moving
ALPHA_STATIC = 0.3   # Smooth heavily when user is stationary
MOTION_DIFF_THRESHOLD = 0.5 # Mean absolute diff to detect motion

# --- DECISION ENGINE CONSTANTS ---
COOLDOWN_SECONDS = 2.5
UNCERTAINTY_VARIANCE_THRESHOLD = 0.15
UNCERTAINTY_DEPTH_THRESHOLD = 2.5

# --- SPATIAL CONSTANTS ---
NEAR_THRESHOLD = 3.0
MID_THRESHOLD = 1.5
FREE_SPACE_THRESHOLD = 3.2