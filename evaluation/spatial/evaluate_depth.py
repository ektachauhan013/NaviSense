import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from backend.spatial_intelligence.depth import DepthEstimator
from backend.spatial_intelligence.free_space import analyze_free_space
from backend.spatial_intelligence.proximity import estimate_proximity

def run_evaluation():
    print("=" * 60)
    print("EY WORKSTREAM EVALUATION: DEPTH & SPATIAL INTELLIGENCE")
    print("=" * 60)

    image_path = "datasets/test_image.jpg"
    image = Image.open(image_path).convert("RGB")
    
    # 1. Benchmark
    print("\n[1] Depth Model Benchmark")
    estimator = DepthEstimator()
    
    # Test at pipeline resolution (640px)
    resized_img = image.resize((480, 640), Image.Resampling.BILINEAR)
    
    t0 = time.perf_counter()
    depth_map = estimator.predict(resized_img)
    t1 = time.perf_counter()
    
    print(f"  - Image Size: {resized_img.size}")
    print(f"  - Inference Time: {(t1-t0)*1000:.1f} ms")
    print(f"  - Depth Range: Min {depth_map.min():.2f}, Max {depth_map.max():.2f}")

    # 2. Free Space Corridors
    print("\n[2] Free Space Analysis")
    fs = analyze_free_space(depth_map)
    print(f"  - Center Path Clear: {fs['center_clear']} (Depth: {fs['depths']['center']:.2f})")
    print(f"  - Left Path Clear:   {fs['left_clear']} (Depth: {fs['depths']['left']:.2f})")
    print(f"  - Right Path Clear:  {fs['right_clear']} (Depth: {fs['depths']['right']:.2f})")

    # 3. Proximity Aggregation Test
    print("\n[3] Proximity Test (Ground-Contact Method)")
    h, w = depth_map.shape
    # Mocking a tall person on the left
    mock_bbox = [w*0.1, h*0.2, w*0.3, h*0.9]
    prox = estimate_proximity(depth_map, mock_bbox)
    print(f"  - Mock Object: Person on Left")
    print(f"  - Calculated Proximity: {prox['proximity']} (Depth: {prox['raw_depth']:.2f})")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")

if __name__ == "__main__":
    run_evaluation()