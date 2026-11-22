import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# MOTION ANALYSIS CONSTANTS & THRESHOLDS
# =============================================================================
# Minimum observations in track history before motion can be classified
MIN_OBSERVATIONS: int = 3

# Window size (N): only consider the last N observations for slope estimation
MOTION_WINDOW: int = 5

# Maximum allowable elapsed time between consecutive observations.
# If the time gap between the latest observations exceeds this limit,
# tracking continuity is broken and motion is classified as UNKNOWN.
MAX_GAP_SECONDS: float = 3.0

# Growth rate threshold (fractional area change per second).
# Area growth slope of log(area) vs time: > +0.15/s = APPROACHING, < -0.15/s = RECEDING
GROWTH_RATE_THRESHOLD: float = 0.15

# Lateral speed threshold (horizontal center shift in bbox-widths per second).
# Center x slope vs time normalized by average bbox width: > 0.20/s = CROSSING
LATERAL_SPEED_THRESHOLD: float = 0.20

# =============================================================================
# KNOWN SENSOR DYNAMICS LIMITATION
# =============================================================================
# Note: The camera is held or worn by a walking user.
# As the user walks forward, stationary obstacles naturally expand in the image
# frame (apparent approach) and shift sideways when the user turns their head/body.
# Therefore, bounding box motion captures relative camera-obstacle dynamics,
# not true world-coordinate velocity. Downstream modules (risk.py) must keep
# motion weights low so user walking does not trigger false alarm panics.
# =============================================================================


def analyze_motion_for_track(track: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes historical observations of a single track and determines motion label.
    
    Args:
        track: Dictionary containing 'history' and 'observation_count' from Tracker.
        
    Returns:
        Dict with keys:
            - label: 'APPROACHING' | 'RECEDING' | 'CROSSING' | 'STATIC' | 'UNKNOWN'
            - trend_value: normalized magnitude of the dominant motion signal
            - growth_rate_per_s: slope of log(area) vs time (1/s)
            - lateral_speed_per_s: slope of center_x vs time in bbox-widths/s
    """
    history = track.get("history", [])
    obs_count = track.get("observation_count", len(history))

    # Rule 1: Insufficient observations for reliable trend fitting
    if obs_count < MIN_OBSERVATIONS or len(history) < MIN_OBSERVATIONS:
        return {
            "label": "UNKNOWN",
            "trend_value": 0.0,
            "growth_rate_per_s": 0.0,
            "lateral_speed_per_s": 0.0,
        }

    # Slice the most recent N observations
    window = list(history)[-MOTION_WINDOW:]

    # Rule 2: Check for temporal discontinuity (excessive time gap between observations)
    timestamps = np.array([obs["timestamp"] for obs in window], dtype=float)
    time_diffs = np.diff(timestamps)
    if np.any(time_diffs > MAX_GAP_SECONDS) or np.any(time_diffs <= 0.0):
        return {
            "label": "UNKNOWN",
            "trend_value": 0.0,
            "growth_rate_per_s": 0.0,
            "lateral_speed_per_s": 0.0,
        }

    total_time = timestamps[-1] - timestamps[0]
    if total_time <= 1e-4:
        return {
            "label": "UNKNOWN",
            "trend_value": 0.0,
            "growth_rate_per_s": 0.0,
            "lateral_speed_per_s": 0.0,
        }

    # Normalized time vector starting from 0.0
    dt = timestamps - timestamps[0]

    # 1. Growth Rate Estimation via Least-Squares Slope of log(area)
    # Using log(area) gives d/dt(ln(A)) = (1/A)(dA/dt), representing fractional
    # percentage expansion per second, invariant to absolute object size.
    areas = np.array([max(1.0, obs["area"]) for obs in window], dtype=float)
    log_areas = np.log(areas)
    # deg=1 returns [slope, intercept]
    growth_slope = float(np.polyfit(dt, log_areas, deg=1)[0])

    # 2. Lateral Speed Estimation via Least-Squares Slope of center_x
    # Divided by the average bounding box width in the window so the velocity
    # is scale-invariant and independent of image pixel resolution.
    centers_x = np.array([obs["center"][0] for obs in window], dtype=float)
    widths = np.array([max(1.0, obs["bbox"][2] - obs["bbox"][0]) for obs in window], dtype=float)
    avg_width = float(np.mean(widths))
    lateral_pixel_slope = float(np.polyfit(dt, centers_x, deg=1)[0])
    lateral_speed = float(lateral_pixel_slope / avg_width)

    # 3. Signal Threshold Evaluation
    has_growth = abs(growth_slope) >= GROWTH_RATE_THRESHOLD
    has_lateral = abs(lateral_speed) >= LATERAL_SPEED_THRESHOLD

    # Normalization against their respective thresholds to compare signal strength
    norm_growth = abs(growth_slope) / GROWTH_RATE_THRESHOLD
    norm_lateral = abs(lateral_speed) / LATERAL_SPEED_THRESHOLD

    # 4. Dominant Motion Classification
    # When both growth (approaching/receding) and lateral displacement (crossing)
    # exceed their thresholds, we normalize both signals by their respective
    # detection thresholds. The larger normalized ratio represents the statistically
    # stronger physical behavior observed in the camera frame.
    if has_growth and has_lateral:
        if norm_growth >= norm_lateral:
            label = "APPROACHING" if growth_slope > 0 else "RECEDING"
            trend_val = round(norm_growth, 2)
        else:
            label = "CROSSING"
            trend_val = round(norm_lateral, 2)
    elif has_growth:
        label = "APPROACHING" if growth_slope > 0 else "RECEDING"
        trend_val = round(norm_growth, 2)
    elif has_lateral:
        label = "CROSSING"
        trend_val = round(norm_lateral, 2)
    else:
        label = "STATIC"
        trend_val = 0.0

    return {
        "label": label,
        "trend_value": trend_val,
        "growth_rate_per_s": round(growth_slope, 3),
        "lateral_speed_per_s": round(lateral_speed, 3),
    }


def analyze_motion(tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Annotates a list of tracked objects with motion classifications.
    
    Args:
        tracked_objects: List of dicts returned by Tracker.update()
        
    Returns:
        The same list of objects, each enriched with a 'motion' key.
    """
    for obj in tracked_objects:
        obj["motion"] = analyze_motion_for_track(obj)
    return tracked_objects


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING MOTION MODULE TEST SUITE WITH ASSERTIONS")
    print("=" * 80)

    # Helper function to generate simulated track dictionaries
    def make_track(t_id: int, observations: List[Tuple[float, float, float, float, float]]) -> Dict[str, Any]:
        """
        observations: list of (timestamp, cx, cy, width, height)
        """
        history = []
        for t, cx, cy, w, h in observations:
            history.append({
                "timestamp": t,
                "center": (cx, cy),
                "area": float(w * h),
                "bbox": [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            })
        return {
            "track_id": t_id,
            "label": "chair",
            "confidence": 0.90,
            "class_id": 56,
            "bbox": history[-1]["bbox"],
            "observation_count": len(history),
            "history": history,
        }

    test_tracks = []

    # Case 1: Still object with small random pixel jitter (dx ~ +/- 1px, dw ~ +/- 1px) -> STATIC
    obs_still = [
        (0.0, 100.0, 200.0, 100.0, 100.0),
        (0.5, 101.0, 199.5, 100.5, 99.5),
        (1.0, 99.5, 200.2, 99.8, 100.2),
        (1.5, 100.2, 199.8, 100.1, 100.0),
    ]
    test_tracks.append(("Still Object", make_track(1, obs_still), "STATIC"))

    # Case 2: Growing object (Area expands by > 20% per second) -> APPROACHING
    # t=0: 100x100=10000; t=1: 115x115=13225; t=2: 132x132=17424 (growth ~ +15% per second)
    obs_growing = [
        (0.0, 300.0, 300.0, 100.0, 100.0),
        (1.0, 300.0, 300.0, 115.0, 115.0),
        (2.0, 300.0, 300.0, 132.0, 132.0),
    ]
    test_tracks.append(("Growing Object", make_track(2, obs_growing), "APPROACHING"))

    # Case 3: Shrinking object (Area contracts by > 20% per second) -> RECEDING
    obs_shrinking = [
        (0.0, 500.0, 500.0, 140.0, 140.0),
        (1.0, 500.0, 500.0, 120.0, 120.0),
        (2.0, 500.0, 500.0, 100.0, 100.0),
    ]
    test_tracks.append(("Shrinking Object", make_track(3, obs_shrinking), "RECEDING"))

    # Case 4: Moving sideways (cx shifts by > 0.25 bbox widths per second) -> CROSSING
    # width=100. At 1.0s cx shifts by 35px = 0.35 widths/s
    obs_crossing = [
        (0.0, 200.0, 400.0, 100.0, 100.0),
        (1.0, 235.0, 400.0, 100.0, 100.0),
        (2.0, 270.0, 400.0, 100.0, 100.0),
    ]
    test_tracks.append(("Crossing Object", make_track(4, obs_crossing), "CROSSING"))

    # Case 5A: Insufficient observations (1 observation) -> UNKNOWN
    obs_one = [
        (0.0, 200.0, 400.0, 100.0, 100.0),
    ]
    test_tracks.append(("1 Observation", make_track(5, obs_one), "UNKNOWN"))

    # Case 5B: Insufficient observations (2 observations) -> UNKNOWN
    obs_two = [
        (0.0, 200.0, 400.0, 100.0, 100.0),
        (1.0, 250.0, 400.0, 120.0, 120.0),
    ]
    test_tracks.append(("2 Observations", make_track(6, obs_two), "UNKNOWN"))

    # Case 6: Large time gap (> MAX_GAP_SECONDS = 3.0s) between observations -> UNKNOWN
    obs_gap = [
        (0.0, 100.0, 200.0, 100.0, 100.0),
        (1.0, 100.0, 200.0, 100.0, 100.0),
        (5.5, 100.0, 200.0, 100.0, 100.0),  # 4.5s gap between t=1.0 and t=5.5
    ]
    test_tracks.append(("Large Time Gap", make_track(7, obs_gap), "UNKNOWN"))

    # Run analysis
    raw_track_list = [t[1] for t in test_tracks]
    analyzed_tracks = analyze_motion(raw_track_list)

    # Print summary table and execute asserts
    print(f"{'Test Case':<18} | {'Expected':<12} | {'Actual':<12} | {'Growth Rate/s':<14} | {'Lateral Spd/s':<14} | {'Status'}")
    print("-" * 88)
    for idx, (name, orig_trk, expected_label) in enumerate(test_tracks):
        res = analyzed_tracks[idx]["motion"]
        actual_label = res["label"]
        growth = f"{res['growth_rate_per_s']:+.3f}"
        lateral = f"{res['lateral_speed_per_s']:+.3f}"
        status = "PASS" if actual_label == expected_label else "FAIL"
        print(f"{name:<18} | {expected_label:<12} | {actual_label:<12} | {growth:<14} | {lateral:<14} | {status}")
        assert actual_label == expected_label, f"Assertion failed for {name}: expected {expected_label}, got {actual_label}"

    print("=" * 88)
    print("ALL MOTION ASSERTIONS PASSED SUCCESSFULLY.\n")
