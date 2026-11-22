import os
import sys
import time
import logging
from typing import Dict, Any
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from backend.perception.detector import Detector
from backend.perception.tracker import Tracker
from backend.perception.motion import analyze_motion
from backend.perception.risk import assess_risk
from backend.spatial_intelligence.depth import DepthEstimator
from backend.spatial_intelligence.free_space import analyze_free_space
from backend.spatial_intelligence.advice import generate_advice

logger = logging.getLogger(__name__)
MAX_SIDE: int = 640

def to_plain_python(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)): return int(obj)
    elif isinstance(obj, (np.floating, float)): return float(obj)
    elif isinstance(obj, (np.bool_, bool)): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: to_plain_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [to_plain_python(x) for x in obj]
    return obj

class Pipeline:
    def __init__(self, conf_threshold: float = 0.30):
        logger.info("Initializing Pipeline models...")
        self.detector = Detector(conf_threshold=conf_threshold)
        self.depth_estimator = DepthEstimator()
        self.tracker = Tracker()
        self.tracker.reset()
        
        # Caching for frame skipping
        self.last_depth_map = None
        self.frame_count = 0
        self.depth_skip_interval = 3 # Run depth every 3rd frame
        
        logger.info("Pipeline models successfully loaded.")

    def reset(self):
        self.tracker.reset()
        self.last_depth_map = None
        self.frame_count = 0

    def process(self, image: Image.Image, timestamp: float) -> Dict[str, Any]:
        total_start = time.perf_counter()
        w, h = image.size

        # 1. Detection & Tracking (Fast, runs every frame)
        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        tracked_objects = self.tracker.update(detections, timestamp=timestamp)
        objects_with_motion = analyze_motion(tracked_objects)
        track_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Depth Estimation (Slow, runs every N frames)
        t0 = time.perf_counter()
        self.frame_count += 1
        
        if self.frame_count % self.depth_skip_interval == 0 or self.last_depth_map is None:
            depth_map = self.depth_estimator.predict(image)
            self.last_depth_map = depth_map
            depth_ran = True
        else:
            depth_map = self.last_depth_map
            depth_ran = False
            
        depth_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Spatial & Risk Logic
        t0 = time.perf_counter()
        free_space_result = analyze_free_space(depth_map)

        risk_results = assess_risk(
            tracked_objects=objects_with_motion,
            depth_map=depth_map,
            image_width=w,
            image_height=h,
            free_space_result=free_space_result,
        )

        ranked_objects = risk_results["ranked_objects"]
        top_threat = risk_results["top_threat"]
        advice_inputs = risk_results["advice_inputs"]

        final_advice = generate_advice(
            advice_inputs["free_space_result"],
            advice_inputs["objects"],
        )
        risk_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Clean payload
        clean_objects = []
        for obj in ranked_objects:
            clean_objects.append({
                "track_id": int(obj["track_id"]),
                "label": str(obj["label"]),
                "confidence": float(obj["confidence"]),
                "bbox": [float(c) for c in obj["bbox"]],
                "zone": str(obj["zone"]),
                "proximity": str(obj["proximity"]),
                "motion_label": str(obj["motion_label"]),
                "risk_score": float(obj["risk_score"]),
                "risk_level": str(obj["risk_level"]),
                "reasons": list(obj["reasons"]),
            })

        response_payload = {
            "advice": str(final_advice),
            "top_threat": clean_objects[0] if clean_objects else None,
            "objects": clean_objects,
            "path_clear": bool(advice_inputs["free_space_result"].get("center_clear", True)),
            "image": {"width": int(w), "height": int(h)},
            "timing_ms": {
                "detect": round(float(detect_ms), 1),
                "track": round(float(track_ms), 1),
                "depth": round(float(depth_ms), 1),
                "depth_ran": bool(depth_ran),
                "risk": round(float(risk_ms), 1),
                "total": round(float(total_ms), 1),
            },
        }

        return to_plain_python(response_payload)