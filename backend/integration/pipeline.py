import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

# Ensure Hugging Face uses cached offline weights without waiting for network timeout
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.perception.detector import Detector
from backend.perception.tracker import Tracker
from backend.perception.motion import analyze_motion
from backend.perception.risk import assess_risk
from backend.spatial_intelligence.depth import DepthEstimator
from backend.spatial_intelligence.free_space import analyze_free_space
from backend.spatial_intelligence.advice import generate_advice

# Configure module-level logger
logger = logging.getLogger(__name__)

# Maximum pixel dimension (longest side) for inference.
# Resizing large camera frames (e.g. 2000x2400) to <= 640px reduces CPU inference time
# from ~15 seconds to ~1 second, which is critical for real-time mobile guidance.
MAX_SIDE: int = 640


def to_plain_python(obj: Any) -> Any:
    """
    Recursively converts NumPy scalars, arrays, and booleans into standard Python types
    to guarantee flawless JSON serialization across all endpoints.
    """
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_plain_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_plain_python(x) for x in obj]
    return obj


class Pipeline:
    """
    Core AI orchestration pipeline. Loads and holds the Detector, Tracker, and
    DepthEstimator models once in memory. Executes detection, tracking, motion,
    depth, free space, risk assessment, and speech advice generation in order.
    """
    def __init__(self, conf_threshold: float = 0.30):
        logger.info("Initializing NaviSense Pipeline models (YOLO + Depth Anything V2 + Tracker)...")
        self.detector = Detector(conf_threshold=conf_threshold)
        self.depth_estimator = DepthEstimator()
        self.tracker = Tracker()
        self.tracker.reset()
        logger.info("Pipeline models successfully loaded into memory.")

    def reset(self):
        """Resets the persistent multi-object tracker."""
        self.tracker.reset()
        logger.info("Pipeline tracker has been reset.")

    def process(self, image: Image.Image, timestamp: float) -> Dict[str, Any]:
        """
        Processes a single preprocessed PIL image through the complete perception and
        spatial intelligence stack.
        
        Args:
            image: Preprocessed RGB PIL Image (already resized so longest side <= MAX_SIDE)
            timestamp: Float timestamp recorded at request arrival
            
        Returns:
            Dictionary with JSON-serializable types:
                - advice: spoken advice string from EY's generate_advice
                - top_threat: highest risk object dict or None
                - objects: list of ranked obstacles with spatial/risk metadata
                - path_clear: boolean indicating if walking trajectory is clear
                - image: dict with width and height of processed frame
                - timing_ms: per-stage latency measurements
        """
        total_start = time.perf_counter()
        w, h = image.size

        # -------------------------------------------------------------
        # 1. Object Detection (YOLOv8n)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 2. Tracking + Motion Analysis
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        tracked_objects = self.tracker.update(detections, timestamp=timestamp)
        objects_with_motion = analyze_motion(tracked_objects)
        track_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 3. Monocular Depth Estimation (Depth Anything V2)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        depth_map = self.depth_estimator.predict(image)
        depth_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # 4. Spatial Free Space + Risk Scoring + Advice Generation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        # Call EY's free space analyzer on the floor region
        free_space_result = analyze_free_space(depth_map)

        # Run Risk Engine (bridges spatial intelligence with motion & class)
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

        # Call EY's rule-based advice generator using the risk-adapted inputs
        final_advice = generate_advice(
            advice_inputs["free_space_result"],
            advice_inputs["objects"],
        )
        risk_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Filter ranked objects to clean public API fields
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

        clean_top_threat = None
        if top_threat:
            clean_top_threat = {
                "track_id": int(top_threat["track_id"]),
                "label": str(top_threat["label"]),
                "confidence": float(top_threat["confidence"]),
                "bbox": [float(c) for c in top_threat["bbox"]],
                "zone": str(top_threat["zone"]),
                "proximity": str(top_threat["proximity"]),
                "motion_label": str(top_threat["motion_label"]),
                "risk_score": float(top_threat["risk_score"]),
                "risk_level": str(top_threat["risk_level"]),
                "reasons": list(top_threat["reasons"]),
            }

        response_payload = {
            "advice": str(final_advice),
            "top_threat": clean_top_threat,
            "objects": clean_objects,
            "path_clear": bool(advice_inputs["free_space_result"].get("is_clear", True)),
            "image": {"width": int(w), "height": int(h)},
            "timing_ms": {
                "detect": round(float(detect_ms), 1),
                "track": round(float(track_ms), 1),
                "depth": round(float(depth_ms), 1),
                "risk": round(float(risk_ms), 1),
                "total": round(float(total_ms), 1),
            },
        }

        # Final guarantee of JSON compatibility
        return to_plain_python(response_payload)


if __name__ == "__main__":
    test_image_path = "datasets/test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"Error: image not found at {test_image_path}")
        exit(1)

    print("\n" + "=" * 80)
    print("RUNNING PIPELINE TEST ON TEST_IMAGE.JPG")
    print("=" * 80)

    raw_image = Image.open(test_image_path).convert("RGB")
    # Resize so longest side <= MAX_SIDE (640)
    w, h = raw_image.size
    scale = min(1.0, float(MAX_SIDE) / max(w, h))
    target_size = (int(round(w * scale)), int(round(h * scale)))
    resized_image = raw_image.resize(target_size, Image.Resampling.BILINEAR)
    print(f"Original image: {w}x{h} -> Resized for CPU pipeline: {target_size[0]}x{target_size[1]}")

    pipeline = Pipeline()
    res = pipeline.process(resized_image, timestamp=time.time())

    print("\nPipeline Result:")
    print(f"  Advice: \"{res['advice']}\"")
    print(f"  Path Clear: {res['path_clear']}")
    print(f"  Objects Detected: {len(res['objects'])}")
    if res['top_threat']:
        print(f"  Top Threat: {res['top_threat']['label']} (Risk: {res['top_threat']['risk_score']:.3f}, Level: {res['top_threat']['risk_level']})")
    print(f"  Timing (ms): {res['timing_ms']}")
    print("\n[PASS] Pipeline test completed successfully.\n")
