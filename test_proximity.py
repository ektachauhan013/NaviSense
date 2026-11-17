from PIL import Image
from backend.spatial_intelligence.depth import DepthEstimator
from backend.spatial_intelligence.proximity import estimate_proximity
from backend.spatial_intelligence.zones import classify_spatial_zone
from backend.spatial_intelligence.free_space import analyze_free_space
from backend.spatial_intelligence.advice import generate_advice

# 1. Setup
image_path = "datasets/test_image.jpg"
image = Image.open(image_path).convert("RGB")
estimator = DepthEstimator()
depth_map = estimator.predict(image)
h, w = depth_map.shape

# 2. Free Space
space_result = analyze_free_space(depth_map)

# 3. Mock Objects (Simulating EC's YOLO Detector)
# We pass the bounding boxes to our functions, create dictionaries, and put them in a list.
detected_objects = []

# Left object (Chair)
bbox_left = [w * 0.05, h * 0.25, w * 0.25, h * 0.75]
prox_left = estimate_proximity(depth_map, bbox_left)
zone_left = classify_spatial_zone(bbox_left, w)
detected_objects.append({"proximity": prox_left["proximity"], "zone": zone_left})

# Right object (Stairs)
bbox_right = [w * 0.75, h * 0.25, w * 0.95, h * 0.75]
prox_right = estimate_proximity(depth_map, bbox_right)
zone_right = classify_spatial_zone(bbox_right, w)
detected_objects.append({"proximity": prox_right["proximity"], "zone": zone_right})

# 4. Generate Advice
print("\n--- FINAL NAVISENSE EY OUTPUT ---")
final_advice = generate_advice(space_result, detected_objects)
print(f"Voice Output: \"{final_advice}\"")