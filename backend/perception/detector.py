import logging
from typing import List, Dict, Any, Union, Set, Optional
from PIL import Image
import numpy as np

# Configure module-level logger to trace detection events and timing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of COCO classes relevant to indoor obstacles for a visually impaired user.
# Filtering prevents alerting on irrelevant background objects (e.g., 'tie', 'frisbee')
# and keeps the spoken advice focused on navigable hazards.
DEFAULT_INDOOR_CLASSES: Set[str] = {
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "refrigerator",
    "potted plant",
    "suitcase",
    "backpack",
    "bottle",
}

class Detector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        allowed_classes: Optional[Set[str]] = None,
    ):
        """
        Loads the YOLOv8 model into memory once at startup.
        Loading on every frame would cause unacceptable multi-second latency on a laptop CPU.
        """
        # Lazy import so module can still be inspected even if ultralytics is not yet installed
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is not installed. "
                "Install it using: pip install ultralytics torch torchvision"
            ) from exc

        logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.allowed_classes = allowed_classes if allowed_classes is not None else DEFAULT_INDOOR_CLASSES
        logger.info(
            f"Detector initialized. Confidence threshold: {self.conf_threshold}, "
            f"Allowed classes count: {len(self.allowed_classes)}"
        )

    def detect(
        self,
        frame: Union[Image.Image, np.ndarray],
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs object detection on a single frame (PIL Image or NumPy array).
        
        Returns:
            List of dictionaries matching the spatial intelligence pipeline contract:
            [
                {
                    "bbox": [x1, y1, x2, y2],  # float coordinates in original frame pixels
                    "label": str,               # class label (e.g. 'chair')
                    "confidence": float,        # detection score 0.0 to 1.0
                    "class_id": int             # COCO integer class id
                },
                ...
            ]
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Determine original dimensions to verify bounding boxes remain in original pixel space
        if isinstance(frame, Image.Image):
            orig_w, orig_h = frame.size
        elif isinstance(frame, np.ndarray):
            orig_h, orig_w = frame.shape[:2]
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}. Expected PIL.Image or np.ndarray.")

        # Run inference. verbose=False silences per-frame console spam from YOLO
        results = self.model(frame, conf=threshold, verbose=False)

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract names mapping from model (id -> label string)
        names = self.model.names

        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = names.get(cls_id, str(cls_id))

            # Filter out non-indoor objects to keep advice relevant and avoid false alarms
            if self.allowed_classes and label not in self.allowed_classes:
                continue

            # Ultralytics boxes.xyxy automatically maps back to input frame pixel coordinates.
            # Convert tensors to standard Python floats and clamp within image boundaries.
            coords = box.xyxy[0].tolist()
            x1 = max(0.0, min(float(coords[0]), float(orig_w)))
            y1 = max(0.0, min(float(coords[1]), float(orig_h)))
            x2 = max(0.0, min(float(coords[2]), float(orig_w)))
            y2 = max(0.0, min(float(coords[3]), float(orig_h)))

            # Validate that the bounding box has non-zero area
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": round(confidence, 3),
                "class_id": cls_id,
            })

        logger.info(f"Detected {len(detections)} indoor obstacle(s) at conf >= {threshold}")
        return detections


if __name__ == "__main__":
    import os
    from PIL import ImageDraw

    print("\n--- DEFAULT INDOOR CLASSES ---")
    print(sorted(list(DEFAULT_INDOOR_CLASSES)))

    test_image_path = "datasets/test_image.jpg"
    if not os.path.exists(test_image_path):
        logger.error(f"Test image not found at '{test_image_path}'")
        exit(1)

    image = Image.open(test_image_path).convert("RGB")
    orig_w, orig_h = image.size
    print(f"\nImage loaded: {test_image_path} | Size: Width={orig_w}, Height={orig_h}")

    try:
        detector = Detector(conf_threshold=0.30)
        results = detector.detect(image)

        print("\n--- DETECTOR OUTPUT ---")
        for idx, det in enumerate(results, start=1):
            bbox_str = f"[{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
            print(f"#{idx}: {det['label']:15} | conf: {det['confidence']:.2f} | bbox: {bbox_str}")
        print(f"Total relevant detections: {len(results)}\n")

        # Draw boxes and save to evaluation/perception/detector_test.jpg
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        # Try to load a larger default font (supported in Pillow 10+)
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=44)
        except Exception:
            font = None

        for det in results:
            b = det["bbox"]
            label_text = f" {det['label']} ({det['confidence']:.2f}) "
            
            # Draw bounding box outline with prominent thickness
            draw.rectangle(b, outline="red", width=8)

            # Calculate text banner dimensions
            if font and hasattr(draw, "textbbox"):
                tb = draw.textbbox((0, 0), label_text, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
            else:
                tw = len(label_text) * 12
                th = 25

            # Position banner above box, or just inside if at very top of image
            banner_y1 = max(0.0, b[1] - th - 10.0)
            banner_y2 = banner_y1 + th + 10.0
            banner_x1 = b[0]
            banner_x2 = b[0] + tw + 10.0

            # Draw filled red background banner and bold white text
            draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2], fill="red")
            draw.text((banner_x1 + 5, banner_y1 + 5), label_text, fill="white", font=font)

        out_dir = "evaluation/perception"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "detector_test.jpg")
        draw_image.save(out_path, quality=90)
        print(f"Annotated test image saved to: {out_path}")

    except ImportError as e:
        logger.error(f"Dependency error: {e}")