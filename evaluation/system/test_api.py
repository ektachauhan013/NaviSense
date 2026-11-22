import os
import sys
import io
import json
import logging
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from PIL import Image
from fastapi.testclient import TestClient
from backend.api.app import app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_api_tests():
    print("\n" + "=" * 80)
    print("RUNNING API TEST SUITE (FastAPI TestClient)")
    print("=" * 80)

    test_image_path = "datasets/test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"Error: test image not found at {test_image_path}")
        return

    with open(test_image_path, "rb") as f:
        image_bytes = f.read()

    # Use TestClient as a context manager so lifespan runs (loading models into app.state)
    with TestClient(app) as client:
        # -------------------------------------------------------------
        # Test 1: GET /health
        # -------------------------------------------------------------
        resp_health = client.get("/health")
        assert resp_health.status_code == 200, f"Expected 200, got {resp_health.status_code}"
        health_data = resp_health.json()
        assert health_data["status"] == "ok"
        assert health_data["models_loaded"] is True
        print("[PASS] Test 1: GET /health returned 200 OK with models_loaded=True.", flush=True)

        # -------------------------------------------------------------
        # Test 2: POST /analyze-frame contract & JSON serializability
        # -------------------------------------------------------------
        print("Running Test 2: POST /analyze-frame...", flush=True)
        files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
        resp_analyze = client.post("/analyze-frame", files=files)
        assert resp_analyze.status_code == 200, f"Expected 200, got {resp_analyze.status_code}: {resp_analyze.text}"
        data = resp_analyze.json()
        assert data["status"] == "ok"

        required_keys = ["status", "advice", "top_threat", "objects", "path_clear", "image", "timing_ms"]
        for k in required_keys:
            assert k in data, f"Missing required response key: '{k}'"

        # Verify JSON serializability of response dict
        serialized = json.dumps(data)
        assert len(serialized) > 0
        print("[PASS] Test 2: POST /analyze-frame returned 200 with all schema keys (JSON validated).", flush=True)

        # -------------------------------------------------------------
        # Test 3: Track ID persistence across 3 consecutive requests
        # -------------------------------------------------------------
        print("Running Test 3: Track ID persistence across 3 consecutive requests...", flush=True)
        client.post("/reset")
        id_runs = []
        for run_idx in range(3):
            files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
            resp = client.post("/analyze-frame", files=files)
            assert resp.status_code == 200
            run_data = resp.json()
            track_ids = [obj["track_id"] for obj in run_data["objects"]]
            id_runs.append(track_ids)

        assert id_runs[0] == id_runs[1] == id_runs[2], (
            f"Track IDs diverged across 3 runs with identical image: {id_runs}"
        )
        print(f"[PASS] Test 3: Track IDs persisted across 3 requests: {id_runs[0]}", flush=True)

        # -------------------------------------------------------------
        # Test 4: POST /reset clears history and restarts track IDs
        # -------------------------------------------------------------
        print("Running Test 4: POST /reset...", flush=True)
        resp_reset = client.post("/reset")
        assert resp_reset.status_code == 200
        assert resp_reset.json() == {"status": "reset"}

        files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
        resp_after_reset = client.post("/analyze-frame", files=files)
        assert resp_after_reset.status_code == 200
        post_reset_ids = [obj["track_id"] for obj in resp_after_reset.json()["objects"]]
        assert post_reset_ids == id_runs[0], (
            f"Track IDs did not restart at initial indices after reset: {post_reset_ids}"
        )
        print(f"[PASS] Test 4: POST /reset successfully cleared tracker; IDs restarted at: {post_reset_ids}", flush=True)

        # -------------------------------------------------------------
        # Test 5: Invalid file upload returns HTTP 400
        # -------------------------------------------------------------
        print("Running Test 5: Invalid file upload...", flush=True)
        text_bytes = b"Hello, this is a plain text file, not an image!"
        files_bad = {"file": ("notes.txt", text_bytes, "text/plain")}
        resp_bad = client.post("/analyze-frame", files=files_bad)
        assert resp_bad.status_code == 400, f"Expected HTTP 400 for text upload, got {resp_bad.status_code}"
        print(f"[PASS] Test 5: Non-image upload rejected with HTTP 400: '{resp_bad.json()['detail']}'", flush=True)

        # -------------------------------------------------------------
        # Test 6: Rotated / Transposed Image Handling
        # -------------------------------------------------------------
        print("Running Test 6: Rotated image handling...", flush=True)
        pil_img = Image.open(io.BytesIO(image_bytes))
        rotated_img = pil_img.transpose(Image.Transpose.ROTATE_90)
        rotated_buf = io.BytesIO()
        rotated_img.save(rotated_buf, format="JPEG")
        rotated_bytes = rotated_buf.getvalue()

        files_rot = {"file": ("rotated.jpg", rotated_bytes, "image/jpeg")}
        resp_rot = client.post("/analyze-frame", files=files_rot)
        assert resp_rot.status_code == 200, f"Expected 200 for rotated image, got {resp_rot.status_code}"
        rot_data = resp_rot.json()
        assert rot_data["status"] == "ok"
        print(f"[PASS] Test 6: 90-degree rotated image processed without crashing (Resized: {rot_data['image']['width']}x{rot_data['image']['height']}).", flush=True)

        # -------------------------------------------------------------
        # Test 7: Benchmark average per-stage timing over 5 requests
        # -------------------------------------------------------------
        print("\n" + "=" * 80, flush=True)
        print("BENCHMARKING LATENCY OVER 5 REQUESTS (CPU)", flush=True)
        print("=" * 80, flush=True)

        timing_samples: Dict[str, List[float]] = {
            "detect": [],
            "track": [],
            "depth": [],
            "risk": [],
            "total": [],
        }

        for i in range(1, 6):
            files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
            resp = client.post("/analyze-frame", files=files)
            assert resp.status_code == 200
            t_dict = resp.json()["timing_ms"]
            for stage in timing_samples:
                timing_samples[stage].append(t_dict[stage])
            print(f"Request #{i}: Total = {t_dict['total']:6.1f} ms | Detect = {t_dict['detect']:5.1f} ms | Track = {t_dict['track']:4.1f} ms | Depth = {t_dict['depth']:5.1f} ms | Risk = {t_dict['risk']:4.1f} ms", flush=True)

        print("-" * 80, flush=True)
        print("AVERAGE STAGE TIMINGS OVER 5 SAMPLES:", flush=True)
        for stage, samples in timing_samples.items():
            avg_time = sum(samples) / len(samples)
            print(f"  {stage.capitalize():<8}: {avg_time:6.1f} ms (min: {min(samples):6.1f} ms, max: {max(samples):6.1f} ms)", flush=True)
        print("=" * 80, flush=True)
        print("\nALL API TESTS PASSED SUCCESSFULLY.\n", flush=True)


if __name__ == "__main__":
    run_api_tests()
