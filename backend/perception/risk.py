import os
import sys
import copy
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path when running file directly
sys.path.insert(0, os.path.abspath("."))

from backend.spatial_intelligence.zones import classify_spatial_zone
from backend.spatial_intelligence.proximity import estimate_proximity

# Configure module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# RISK ENGINE WEIGHTS & THRESHOLDS
# =============================================================================
# Total weights sum to 1.0. All parameters are named constants for explainability.

WEIGHT_PROXIMITY: float = 0.40
PROXIMITY_SCORES: Dict[str, float] = {
    "NEAR": 1.00,      
    "MID": 0.50,       
    "FAR": 0.10,       
    "UNKNOWN": 0.20,   
}

WEIGHT_ZONE: float = 0.25
ZONE_SCORES: Dict[str, float] = {
    "CENTER": 1.00,    
    "LEFT": 0.50,      
    "RIGHT": 0.50,     
}

WEIGHT_SIZE_AND_POSITION: float = 0.15
MAX_AREA_FRACTION_REF: float = 0.25  

WEIGHT_CLASS_DANGER: float = 0.12
CLASS_DANGER_SCORES: Dict[str, float] = {
    "person": 1.00,        
    "chair": 0.80,         
    "couch": 0.80,         
    "bed": 0.80,           
    "dining table": 0.80,  
    "suitcase": 0.70,      
    "backpack": 0.70,      
    "toilet": 0.60,        
    "refrigerator": 0.60,  
    "tv": 0.60,            
    "potted plant": 0.50,  
    "bottle": 0.40,        
    "laptop": 0.40,        
}
DEFAULT_CLASS_SCORE: float = 0.50

WEIGHT_MOTION: float = 0.08
MOTION_SCORES: Dict[str, float] = {
    "APPROACHING": 1.00,   
    "CROSSING": 0.60,      
    "STATIC": 0.20,        
    "RECEDING": 0.00,      
    "UNKNOWN": 0.20,       
}

# Risk Level Classification Thresholds
RISK_THRESHOLD_HIGH: float = 0.65
RISK_THRESHOLD_MEDIUM: float = 0.45  


def compute_object_risk(
    obj: Dict[str, Any],
    depth_map: np.ndarray,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """
    Evaluates spatial intelligence, computes risk score (0 to 1), and assigns risk level.
    """
    bbox = obj["bbox"]
    x1, y1, x2, y2 = bbox
    label = obj.get("label", "unknown")
    motion_info = obj.get("motion", {})
    motion_label = motion_info.get("label", "UNKNOWN")

    touches_border = (
        x1 <= 2 or y1 <= 2 or x2 >= (image_width - 2) or y2 >= (image_height - 2)
    )
    if touches_border and motion_label == "RECEDING":
        motion_label = "STATIC"

    # 1. Spatial Zone & Proximity (Call EY's modules read-only)
    # UPDATED: We now pass image_height to classify_spatial_zone for perspective weighting
    zone = classify_spatial_zone(bbox, image_width, image_height)
    
    prox_result = estimate_proximity(depth_map, bbox)
    proximity = prox_result.get("proximity", "UNKNOWN")
    raw_depth = prox_result.get("raw_depth")

    # 2. Compute Individual Score Components
    score_prox = PROXIMITY_SCORES.get(proximity, PROXIMITY_SCORES["UNKNOWN"])
    score_zone = ZONE_SCORES.get(zone, 0.50)

    # Size and position cue
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    box_area = box_w * box_h
    frame_area = max(1.0, float(image_width * image_height))
    area_fraction = box_area / frame_area
    norm_area = min(1.0, area_fraction / MAX_AREA_FRACTION_REF)
    norm_y2 = min(1.0, max(0.0, y2 / float(image_height)))
    score_size_pos = 0.50 * norm_area + 0.50 * norm_y2

    score_class = CLASS_DANGER_SCORES.get(label, DEFAULT_CLASS_SCORE)
    score_motion = MOTION_SCORES.get(motion_label, MOTION_SCORES["UNKNOWN"])

    # 3. Weighted Sum Formula
    c_prox = round(WEIGHT_PROXIMITY * score_prox, 3)
    c_zone = round(WEIGHT_ZONE * score_zone, 3)
    c_size_pos = round(WEIGHT_SIZE_AND_POSITION * score_size_pos, 3)
    c_class = round(WEIGHT_CLASS_DANGER * score_class, 3)
    c_motion = round(WEIGHT_MOTION * score_motion, 3)

    risk_score = round(float(np.clip(c_prox + c_zone + c_size_pos + c_class + c_motion, 0.0, 1.0)), 3)

    # 4. Assign Risk Level
    if risk_score >= RISK_THRESHOLD_HIGH:
        risk_level = "HIGH"
    elif risk_score >= RISK_THRESHOLD_MEDIUM:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # 5. Formulate Human-Explainable Reasons
    reasons = []
    if zone == "CENTER":
        reasons.append("CENTER zone")
    if proximity == "NEAR":
        reasons.append("NEAR proximity")
    elif proximity == "MID":
        reasons.append("MID proximity")
    if area_fraction >= 0.10 or norm_y2 >= 0.80:
        reasons.append("large box low in frame")
    if motion_label == "APPROACHING":
        reasons.append("APPROACHING motion")
    elif motion_label == "CROSSING":
        reasons.append("CROSSING path")
    if score_class >= 0.80:
        reasons.append(f"high-hazard class: {label}")

    return {
        "track_id": obj.get("track_id", 0),
        "label": label,
        "confidence": obj.get("confidence", 0.0),
        "class_id": obj.get("class_id", -1),
        "bbox": [round(float(c), 1) for c in bbox],
        "zone": zone,
        "proximity": proximity,
        "raw_depth": round(float(raw_depth), 2) if raw_depth is not None else None,
        "motion_label": motion_label,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": reasons,
        "components": {
            "proximity": c_prox,
            "zone": c_zone,
            "size_position": c_size_pos,
            "class_danger": c_class,
            "motion": c_motion,
        },
    }


def build_advice_inputs(
    ranked_objects: List[Dict[str, Any]],
    free_space_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Adapter bridging Risk Engine outputs to EY's advice.py.
    """
    if free_space_result is not None:
        adapted_free_space = copy.deepcopy(free_space_result)
    else:
        # UPDATED: Using center_clear instead of is_clear
        adapted_free_space = {"center_clear": True, "left_clear": True, "right_clear": True}

    adapted_objects: List[Dict[str, Any]] = []
    has_high_risk_in_center = False

    for obj in ranked_objects:
        if obj["risk_level"] in ("MEDIUM", "HIGH"):
            obj_copy = copy.deepcopy(obj)
            if obj["risk_level"] == "HIGH":
                obj_copy["proximity"] = "NEAR"
            adapted_objects.append(obj_copy)

            if obj["risk_level"] == "HIGH" and obj["zone"] == "CENTER":
                has_high_risk_in_center = True

    # If a HIGH-risk obstacle blocks the center walking path, force path clearance to False
    if has_high_risk_in_center:
        # UPDATED: Using center_clear instead of is_clear
        adapted_free_space["center_clear"] = False

    return {
        "objects": adapted_objects,
        "free_space_result": adapted_free_space,
    }


def assess_risk(
    tracked_objects: List[Dict[str, Any]],
    depth_map: np.ndarray,
    image_width: int,
    image_height: int,
    free_space_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main risk assessment entry point.
    """
    if not tracked_objects:
        empty_fs = (
            copy.deepcopy(free_space_result)
            if free_space_result is not None
            # UPDATED: Using center_clear instead of is_clear
            else {"center_clear": True, "left_clear": True, "right_clear": True}
        )
        return {
            "ranked_objects": [],
            "top_threat": None,
            "advice_inputs": {
                "objects": [],
                "free_space_result": empty_fs,
            },
        }

    ranked: List[Dict[str, Any]] = []
    for obj in tracked_objects:
        evaluated = compute_object_risk(obj, depth_map, image_width, image_height)
        ranked.append(evaluated)

    ranked.sort(key=lambda x: x["risk_score"], reverse=True)

    top_threat = ranked[0] if ranked else None
    advice_inputs = build_advice_inputs(ranked, free_space_result)

    return {
        "ranked_objects": ranked,
        "top_threat": top_threat,
        "advice_inputs": advice_inputs,
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING RISK ENGINE TEST SUITE WITH ASSERTIONS")
    print("=" * 80)

    w, h = 1000, 1000
    synth_depth = np.full((h, w), 0.5, dtype=np.float32)

    synth_depth[500:950, 350:650] = 4.0
    synth_depth[100:300, 50:250] = 0.8
    synth_depth[400:600, 400:600] = 2.0

    # UPDATED: Test dictionary now uses center_clear, left_clear, right_clear
    fs_initial_clear = {"center_clear": True, "left_clear": True, "right_clear": True}
    
    # Test 1: NEAR + CENTER + APPROACHING + big box low in frame = HIGH
    obj_urgent = {
        "track_id": 1,
        "bbox": [350.0, 500.0, 650.0, 950.0],
        "label": "person",
        "confidence": 0.95,
        "class_id": 0,
        "motion": {"label": "APPROACHING", "trend_value": 1.5},
    }
    res1 = assess_risk([obj_urgent], synth_depth, w, h, fs_initial_clear)

    assert len(res1["ranked_objects"]) == 1
    top1 = res1["top_threat"]
    assert top1 is not None
    assert top1["risk_level"] == "HIGH", f"Expected HIGH risk, got {top1['risk_level']} (score: {top1['risk_score']})"
    assert top1["zone"] == "CENTER"
    assert top1["proximity"] == "NEAR"
    
    # UPDATED: Assertion checks center_clear
    assert res1["advice_inputs"]["free_space_result"]["center_clear"] is False, "Adapter failed to flip center_clear to False for HIGH-risk center obstacle"
    
    assert len(res1["advice_inputs"]["objects"]) == 1
    assert res1["advice_inputs"]["objects"][0]["proximity"] == "NEAR"
    print(f"[PASS] Test 1: Urgent threat -> Score={top1['risk_score']}, Level={top1['risk_level']}. Adapter flipped center_clear=False.")

    # Test 2: FAR + LEFT + STATIC + small box = LOW and excluded from advice
    obj_low = {
        "track_id": 2,
        "bbox": [50.0, 100.0, 120.0, 180.0],
        "label": "chair",
        "confidence": 0.85,
        "class_id": 56,
        "motion": {"label": "STATIC", "trend_value": 0.0},
    }
    res2 = assess_risk([obj_low], synth_depth, w, h, fs_initial_clear)
    top2 = res2["top_threat"]
    assert top2 is not None
    assert top2["risk_level"] == "LOW", f"Expected LOW risk, got {top2['risk_level']} (score: {top2['risk_score']})"
    assert top2["zone"] == "LEFT"
    assert top2["proximity"] == "FAR"
    assert len(res2["advice_inputs"]["objects"]) == 0, "LOW risk object should be excluded from advice_inputs"
    print(f"[PASS] Test 2: Low-risk peripheral item -> Score={top2['risk_score']}, Level={top2['risk_level']}. Excluded from advice.")

    # Test 3: Motion weight sensitivity
    obj_base_unknown = {
        "track_id": 3,
        "bbox": [400.0, 400.0, 600.0, 600.0],
        "label": "chair",
        "confidence": 0.85,
        "class_id": 56,
        "motion": {"label": "UNKNOWN", "trend_value": 0.0},
    }
    obj_base_appr = copy.deepcopy(obj_base_unknown)
    obj_base_appr["motion"] = {"label": "APPROACHING", "trend_value": 1.2}

    res_unk = assess_risk([obj_base_unknown], synth_depth, w, h, fs_initial_clear)["ranked_objects"][0]
    res_appr = assess_risk([obj_base_appr], synth_depth, w, h, fs_initial_clear)["ranked_objects"][0]
    diff = abs(res_appr["risk_score"] - res_unk["risk_score"])
    assert diff <= 0.10, f"Motion weight is too high! Difference was {diff:.3f}"
    assert res_appr["risk_score"] > res_unk["risk_score"], "Approaching object should score higher than unknown"
    print(f"[PASS] Test 3: Motion weight is restrained (Score diff = {diff:.3f} <= 0.10).")

    # Test 4: Class danger ranking
    obj_person = copy.deepcopy(obj_base_unknown)
    obj_person["label"] = "person"
    obj_person["class_id"] = 0

    obj_bottle = copy.deepcopy(obj_base_unknown)
    obj_bottle["label"] = "bottle"
    obj_bottle["class_id"] = 39

    res_person = assess_risk([obj_person], synth_depth, w, h, fs_initial_clear)["ranked_objects"][0]
    res_bottle = assess_risk([obj_bottle], synth_depth, w, h, fs_initial_clear)["ranked_objects"][0]
    assert res_person["risk_score"] > res_bottle["risk_score"], f"Person ({res_person['risk_score']}) should score above bottle ({res_bottle['risk_score']})"
    print(f"[PASS] Test 4: Class hazard hierarchy confirmed (Person: {res_person['risk_score']} > Bottle: {res_bottle['risk_score']}).")

    # Test 5: Empty input handling
    res_empty = assess_risk([], synth_depth, w, h, fs_initial_clear)
    assert res_empty["ranked_objects"] == []
    assert res_empty["top_threat"] is None
    assert res_empty["advice_inputs"]["objects"] == []
    
    # UPDATED: Assertion checks center_clear
    assert res_empty["advice_inputs"]["free_space_result"]["center_clear"] is True

    obj_zero = {
        "track_id": 99,
        "bbox": [50.0, 50.0, 50.0, 50.0],
        "label": "chair",
        "confidence": 0.50,
        "class_id": 56,
        "motion": {"label": "UNKNOWN"},
    }
    res_zero = assess_risk([obj_zero], synth_depth, w, h, fs_initial_clear)
    assert len(res_zero["ranked_objects"]) == 1
    assert res_zero["ranked_objects"][0]["proximity"] == "UNKNOWN"
    print("[PASS] Test 5: Gracefully handled empty input list and zero-area bounding box.")

    # Test 6: Sorting is descending by risk_score
    res_multi = assess_risk([obj_low, obj_urgent, obj_bottle], synth_depth, w, h, fs_initial_clear)
    scores = [o["risk_score"] for o in res_multi["ranked_objects"]]
    assert scores == sorted(scores, reverse=True), f"Objects were not sorted descending by risk score: {scores}"
    assert res_multi["top_threat"]["track_id"] == obj_urgent["track_id"]
    print(f"[PASS] Test 6: Output list strictly sorted descending by risk score ({scores}).")

    # Test 7: Real image values
    real_w, real_h = 2000, 2400
    real_synth_depth = np.zeros((real_h, real_w), dtype=np.float32)
    real_synth_depth[1617:1948, 934:1171] = 2.24
    real_synth_depth[1746:2126, 285:492] = 2.92
    real_synth_depth[1314:1512, 582:794] = 1.41
    real_synth_depth[1450:1566, 1085:1159] = 0.66

    real_objects = [
        {"track_id": 1, "bbox": [934.6, 1617.8, 1170.8, 1947.9], "label": "potted plant", "confidence": 0.86, "class_id": 58, "motion": {"label": "STATIC", "trend_value": 0.0}},
        {"track_id": 2, "bbox": [285.1, 1746.0, 491.3, 2125.7], "label": "chair", "confidence": 0.77, "class_id": 56, "motion": {"label": "STATIC", "trend_value": 0.0}},
        {"track_id": 3, "bbox": [582.3, 1314.8, 793.8, 1511.4], "label": "potted plant", "confidence": 0.36, "class_id": 58, "motion": {"label": "STATIC", "trend_value": 0.0}},
        {"track_id": 4, "bbox": [1085.2, 1450.9, 1158.3, 1565.8], "label": "potted plant", "confidence": 0.35, "class_id": 58, "motion": {"label": "STATIC", "trend_value": 0.0}},
    ]

    res_real = assess_risk(real_objects, real_synth_depth, real_w, real_h, fs_initial_clear)
    ranked_real = res_real["ranked_objects"]

    print("\n" + "=" * 115)
    print("SCORE CONTRIBUTION BREAKDOWN FOR TEST_IMAGE.JPG OBJECTS")
    print("=" * 115)
    print(f"{'Label':<14} | {'Zone':<8} | {'Prox':<6} | {'Total':<6} | {'Level':<7} | {'Prox(40%)':<9} | {'Zone(25%)':<9} | {'Size(15%)':<9} | {'Class(12%)':<10} | {'Mot(8%)':<7} | {'Dominant'}")
    print("-" * 115)
    for ro in ranked_real:
        c = ro["components"]
        dominant_comp = max(c.items(), key=lambda kv: kv[1])[0]
        print(f"{ro['label']:<14} | {ro['zone']:<8} | {ro['proximity']:<6} | {ro['risk_score']:<6.3f} | {ro['risk_level']:<7} | "
              f"{c['proximity']:<9.3f} | {c['zone']:<9.3f} | {c['size_position']:<9.3f} | {c['class_danger']:<10.3f} | {c['motion']:<7.3f} | {dominant_comp}")
    print("=" * 115)

    small_plant_eval = [o for o in ranked_real if o["track_id"] == 4][0]
    assert small_plant_eval["risk_level"] == "LOW", (
        f"Expected small plant to be LOW, got {small_plant_eval['risk_level']} (score: {small_plant_eval['risk_score']})"
    )

    large_plant_eval = [o for o in ranked_real if o["track_id"] == 1][0]
    assert large_plant_eval["risk_level"] in ("MEDIUM", "HIGH"), (
        f"Expected large plant to be MEDIUM or HIGH, got {large_plant_eval['risk_level']} (score: {large_plant_eval['risk_score']})"
    )
    print("[PASS] Test 7: Real test image assertions verified (Small plant = LOW, Large plant = MEDIUM).")

    print("\n" + "=" * 80)
    print("ALL RISK ENGINE ASSERTIONS PASSED SUCCESSFULLY.\n")