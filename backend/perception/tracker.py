import collections
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# Configurable constants for tracking
# N: maximum number of historical observations per track (used by motion.py)
MAX_HISTORY: int = 10

# Maximum consecutive frames a track can be missing before being dropped
MAX_MISSED: int = 5

# Primary matching threshold: minimum Intersection-over-Union to associate detection with track
IOU_THRESHOLD: float = 0.30

# Fallback matching threshold: maximum centroid distance normalized by average bbox size.
# At 1 to 2 fps on a walking phone camera, bounding boxes can move significantly, causing IoU to drop.
# Normalized centroid distance prevents track fragmentation when walking past obstacles.
CENTROID_DIST_THRESHOLD: float = 1.0


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_normalized_centroid_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Computes Euclidean distance between centroids, normalized by the average dimension
    of the two bounding boxes. This makes distance scale-invariant (a 50px shift is small
    for a large sofa, but large for a bottle).
    """
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    w1 = max(1.0, bbox1[2] - bbox1[0])
    h1 = max(1.0, bbox1[3] - bbox1[1])

    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    w2 = max(1.0, bbox2[2] - bbox2[0])
    h2 = max(1.0, bbox2[3] - bbox2[1])

    avg_w = (w1 + w2) / 2.0
    avg_h = (h1 + h2) / 2.0

    dx = (cx1 - cx2) / avg_w
    dy = (cy1 - cy2) / avg_h
    return float(np.sqrt(dx * dx + dy * dy))


class Track:
    """
    Maintains the state and short-term trajectory history for a single tracked obstacle.
    """
    def __init__(
        self,
        track_id: int,
        detection: Dict[str, Any],
        timestamp: float,
        max_history: int = MAX_HISTORY,
    ):
        self.track_id: int = track_id
        self.label: str = detection["label"]
        self.class_id: int = detection["class_id"]
        self.confidence: float = detection["confidence"]
        self.bbox: List[float] = list(detection["bbox"])
        self.missed_frames: int = 0
        self.total_observations: int = 1
        self.max_history: int = max_history

        # Deque automatically evicts observations older than max_history
        self.history: collections.deque = collections.deque(maxlen=max_history)

        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        area = max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

        self.history.append({
            "timestamp": timestamp,
            "center": (cx, cy),
            "area": area,
            "bbox": list(self.bbox),
        })

    def update(self, detection: Dict[str, Any], timestamp: float):
        """
        Updates the track with a newly matched detection and records history.
        """
        self.bbox = list(detection["bbox"])
        self.confidence = detection["confidence"]
        self.missed_frames = 0
        self.total_observations += 1

        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        area = max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

        self.history.append({
            "timestamp": timestamp,
            "center": (cx, cy),
            "area": area,
            "bbox": list(self.bbox),
        })

    def mark_missed(self):
        """Increments consecutive missed frame count when not detected."""
        self.missed_frames += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the track into the dictionary structure expected by downstream modules.
        """
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "label": self.label,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "history": list(self.history),
            "observation_count": len(self.history),
        }


class Tracker:
    """
    Lightweight, CPU-friendly multi-object tracker that persists across server requests.
    Associates detections across frames using class-matching, IoU, and centroid fallbacks.
    """
    def __init__(
        self,
        max_history: int = MAX_HISTORY,
        max_missed: int = MAX_MISSED,
        iou_threshold: float = IOU_THRESHOLD,
        centroid_dist_threshold: float = CENTROID_DIST_THRESHOLD,
    ):
        self.max_history = max_history
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.centroid_dist_threshold = centroid_dist_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_track_id: int = 1
        logger.info(
            f"Tracker initialized. max_history={max_history}, max_missed={max_missed}, "
            f"iou_threshold={iou_threshold}, centroid_dist_threshold={centroid_dist_threshold}"
        )

    def reset(self):
        """Clears all tracks and resets the ID counter to start fresh."""
        self.tracks.clear()
        self.next_track_id = 1
        logger.info("Tracker reset. All active tracks cleared.")

    def update(
        self,
        detections: List[Dict[str, Any]],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """
        Updates tracking state with detections from the current frame at the given timestamp.
        
        Args:
            detections: List of dicts from Detector ({'bbox', 'label', 'confidence', 'class_id'})
            timestamp: Explicit float timestamp provided by the caller (e.g. time.time())
            
        Returns:
            List of currently visible tracked objects with IDs and history.
        """
        active_track_ids = list(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(active_track_ids)
        matched_pairs: List[Tuple[int, int]] = []  # (track_id, det_index)

        # -------------------------------------------------------------
        # Stage 1: Same-Class Matching with IoU
        # -------------------------------------------------------------
        iou_candidates = []
        for t_id in unmatched_tracks:
            track = self.tracks[t_id]
            for d_idx in unmatched_dets:
                det = detections[d_idx]
                # Enforce same-class matching
                if det["label"] != track.label:
                    continue
                iou = compute_iou(track.bbox, det["bbox"])
                if iou >= self.iou_threshold:
                    iou_candidates.append((iou, t_id, d_idx))

        # Sort candidate matches by highest IoU first (greedy assignment)
        iou_candidates.sort(key=lambda x: x[0], reverse=True)
        for iou, t_id, d_idx in iou_candidates:
            if t_id in unmatched_tracks and d_idx in unmatched_dets:
                matched_pairs.append((t_id, d_idx))
                unmatched_tracks.remove(t_id)
                unmatched_dets.remove(d_idx)

        # -------------------------------------------------------------
        # Stage 2: Fallback to Normalized Centroid Distance for Same-Class
        # -------------------------------------------------------------
        dist_candidates = []
        for t_id in unmatched_tracks:
            track = self.tracks[t_id]
            for d_idx in unmatched_dets:
                det = detections[d_idx]
                if det["label"] != track.label:
                    continue
                dist = compute_normalized_centroid_distance(track.bbox, det["bbox"])
                if dist <= self.centroid_dist_threshold:
                    dist_candidates.append((dist, t_id, d_idx))

        # Sort candidate matches by smallest distance first
        dist_candidates.sort(key=lambda x: x[0])
        for dist, t_id, d_idx in dist_candidates:
            if t_id in unmatched_tracks and d_idx in unmatched_dets:
                matched_pairs.append((t_id, d_idx))
                unmatched_tracks.remove(t_id)
                unmatched_dets.remove(d_idx)

        # -------------------------------------------------------------
        # Stage 3: Update Matched Tracks
        # -------------------------------------------------------------
        for t_id, d_idx in matched_pairs:
            self.tracks[t_id].update(detections[d_idx], timestamp)

        # -------------------------------------------------------------
        # Stage 4: Mark Unmatched Tracks as Missed
        # -------------------------------------------------------------
        for t_id in unmatched_tracks:
            self.tracks[t_id].mark_missed()

        # -------------------------------------------------------------
        # Stage 5: Spawn New Tracks for Unmatched Detections
        # -------------------------------------------------------------
        for d_idx in unmatched_dets:
            new_track = Track(
                track_id=self.next_track_id,
                detection=detections[d_idx],
                timestamp=timestamp,
                max_history=self.max_history,
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # -------------------------------------------------------------
        # Stage 6: Drop Expired Tracks (missed > MAX_MISSED)
        # -------------------------------------------------------------
        expired_ids = [
            t_id for t_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed
        ]
        for t_id in expired_ids:
            del self.tracks[t_id]

        # -------------------------------------------------------------
        # Return currently visible tracks (missed_frames == 0)
        # -------------------------------------------------------------
        current_tracked_objects = [
            track.to_dict()
            for track in self.tracks.values()
            if track.missed_frames == 0
        ]

        # Sort by track_id for consistent output ordering
        current_tracked_objects.sort(key=lambda x: x["track_id"])
        return current_tracked_objects


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING EXTENDED TRACKER TEST SUITE WITH ASSERTIONS")
    print("=" * 80)

    tracker = Tracker()
    tracker.reset()

    # -------------------------------------------------------------------------
    # Test 0: First frame empty and middle frame empty
    # -------------------------------------------------------------------------
    tracks_empty = tracker.update([], timestamp=0.0)
    assert len(tracks_empty) == 0, f"Expected 0 tracks for empty initial frame, got {len(tracks_empty)}"
    print("[PASS] Test 0A: Very first frame is empty -> returns 0 tracks.")

    # -------------------------------------------------------------------------
    # Test Scenario 1: Multi-frame tracking with still, moving, growing, blinking,
    # and two close objects
    # -------------------------------------------------------------------------
    # Frame 1 (t=1.0): Initial objects
    f1_dets = [
        {"bbox": [100.0, 100.0, 200.0, 200.0], "label": "chair", "confidence": 0.90, "class_id": 56},   # still chair
        {"bbox": [300.0, 200.0, 360.0, 400.0], "label": "person", "confidence": 0.85, "class_id": 0},   # moving person
        {"bbox": [500.0, 500.0, 550.0, 550.0], "label": "chair", "confidence": 0.80, "class_id": 56},   # growing chair
        {"bbox": [700.0, 700.0, 740.0, 800.0], "label": "bottle", "confidence": 0.75, "class_id": 39},  # blinking bottle
        {"bbox": [800.0, 300.0, 880.0, 420.0], "label": "chair", "confidence": 0.88, "class_id": 56},   # close chair A
        {"bbox": [890.0, 300.0, 970.0, 420.0], "label": "chair", "confidence": 0.87, "class_id": 56},   # close chair B
    ]
    t1 = tracker.update(f1_dets, timestamp=1.0)
    assert len(t1) == 6, f"Expected 6 tracks, got {len(t1)}"
    id_still = t1[0]["track_id"]
    id_person = t1[1]["track_id"]
    id_growing = t1[2]["track_id"]
    id_bottle = t1[3]["track_id"]
    id_close_a = t1[4]["track_id"]
    id_close_b = t1[5]["track_id"]

    # Frame 2 (t=2.0):
    # - still stays
    # - person shifts right
    # - growing expands from 50x50 to 65x65
    # - bottle missing
    # - close chairs shift slightly without swapping
    f2_dets = [
        {"bbox": [100.0, 100.0, 200.0, 200.0], "label": "chair", "confidence": 0.91, "class_id": 56},
        {"bbox": [325.0, 200.0, 385.0, 400.0], "label": "person", "confidence": 0.86, "class_id": 0},
        {"bbox": [492.0, 492.0, 557.0, 557.0], "label": "chair", "confidence": 0.82, "class_id": 56},
        {"bbox": [805.0, 302.0, 885.0, 422.0], "label": "chair", "confidence": 0.89, "class_id": 56},
        {"bbox": [895.0, 302.0, 975.0, 422.0], "label": "chair", "confidence": 0.86, "class_id": 56},
    ]
    t2 = tracker.update(f2_dets, timestamp=2.0)
    assert len(t2) == 5, f"Expected 5 tracks, got {len(t2)}"
    t2_by_id = {t["track_id"]: t for t in t2}
    assert id_still in t2_by_id and t2_by_id[id_still]["observation_count"] == 2
    assert id_person in t2_by_id and t2_by_id[id_person]["observation_count"] == 2
    assert id_growing in t2_by_id and t2_by_id[id_growing]["history"][-1]["area"] == 4225.0
    assert id_close_a in t2_by_id and id_close_b in t2_by_id

    # Frame 3 (t=3.0): bottle returns! Must recover id_bottle
    f3_dets = [
        {"bbox": [100.0, 100.0, 200.0, 200.0], "label": "chair", "confidence": 0.90, "class_id": 56},
        {"bbox": [350.0, 200.0, 410.0, 400.0], "label": "person", "confidence": 0.87, "class_id": 0},
        {"bbox": [485.0, 485.0, 565.0, 565.0], "label": "chair", "confidence": 0.84, "class_id": 56},
        {"bbox": [702.0, 702.0, 742.0, 802.0], "label": "bottle", "confidence": 0.76, "class_id": 39},
        {"bbox": [808.0, 305.0, 888.0, 425.0], "label": "chair", "confidence": 0.88, "class_id": 56},
        {"bbox": [898.0, 305.0, 978.0, 425.0], "label": "chair", "confidence": 0.87, "class_id": 56},
    ]
    t3 = tracker.update(f3_dets, timestamp=3.0)
    t3_by_id = {t["track_id"]: t for t in t3}
    assert id_bottle in t3_by_id, f"Blinking bottle failed to recover Track ID {id_bottle}"
    assert t3_by_id[id_bottle]["observation_count"] == 2
    print("[PASS] Test 1: Standard multi-object tracking, blinking recovery, and close objects.")

    # Test 0B: Frame with empty detection list in the middle
    t_empty_mid = tracker.update([], timestamp=4.0)
    assert len(t_empty_mid) == 0, f"Expected 0 active tracks on empty frame, got {len(t_empty_mid)}"
    print("[PASS] Test 0B: Empty detection list in the middle -> returns 0 active tracks.")

    # -------------------------------------------------------------------------
    # Test 2: Large jump (IoU = 0) handled by Centroid-Distance Fallback
    # -------------------------------------------------------------------------
    tracker.reset()
    # Initial box at [100, 100, 200, 200] (width=100, height=100, center=(150, 150))
    t_jump1 = tracker.update([{"bbox": [100.0, 100.0, 200.0, 200.0], "label": "chair", "confidence": 0.9, "class_id": 56}], 1.0)
    jump_id = t_jump1[0]["track_id"]

    # In next frame, object shifts right to [200, 100, 300, 200] (center=(250, 150))
    # Intersection is 0 width -> IoU = 0.0 (fails IOU_THRESHOLD=0.30)
    # Centroid distance: dx = (250 - 150) / 100 = 1.0 <= CENTROID_DIST_THRESHOLD (1.0)
    iou_val = compute_iou([100.0, 100.0, 200.0, 200.0], [200.0, 100.0, 300.0, 200.0])
    assert iou_val == 0.0, f"Expected IoU to be 0.0, got {iou_val}"

    t_jump2 = tracker.update([{"bbox": [200.0, 100.0, 300.0, 200.0], "label": "chair", "confidence": 0.9, "class_id": 56}], 2.0)
    assert len(t_jump2) == 1, f"Expected 1 track, got {len(t_jump2)}"
    assert t_jump2[0]["track_id"] == jump_id, f"Expected track ID {jump_id} via centroid fallback, got {t_jump2[0]['track_id']}"
    assert t_jump2[0]["observation_count"] == 2
    print(f"[PASS] Test 2: Zero-IoU jump matched successfully via centroid-distance fallback (ID {jump_id} kept).")

    # -------------------------------------------------------------------------
    # Test 3: Object missing for > MAX_MISSED frames returns with a NEW id
    # -------------------------------------------------------------------------
    tracker.reset()
    t_drop1 = tracker.update([{"bbox": [100.0, 100.0, 200.0, 200.0], "label": "chair", "confidence": 0.9, "class_id": 56}], 1.0)
    initial_drop_id = t_drop1[0]["track_id"]

    # Miss for MAX_MISSED + 1 (6 frames)
    for frame_i in range(MAX_MISSED + 1):
        tracker.update([], timestamp=2.0 + frame_i)

    # Reappear at same spot
    t_drop_reappear = tracker.update([{"bbox": [100.0, 100.0, 200.0, 200.0], "label": "chair", "confidence": 0.9, "class_id": 56}], 10.0)
    assert len(t_drop_reappear) == 1
    new_drop_id = t_drop_reappear[0]["track_id"]
    assert new_drop_id != initial_drop_id, f"Expected new ID after expiration, but old ID {initial_drop_id} was reused."
    assert t_drop_reappear[0]["observation_count"] == 1
    print(f"[PASS] Test 3: Track expired after {MAX_MISSED + 1} missed frames; reappeared with NEW ID {new_drop_id} (old was {initial_drop_id}).")

    # -------------------------------------------------------------------------
    # Test 4: Crossing Paths Analysis (Two same-class objects)
    # -------------------------------------------------------------------------
    tracker.reset()
    # Chair A moves Left-to-Right: starts at cx=150, moves +80 px/frame
    # Chair B moves Right-to-Left: starts at cx=350, moves -80 px/frame
    crossing_history = []
    
    # Frame 0: A=[100..200] (cx=150), B=[300..400] (cx=350)
    res0 = tracker.update([
        {"bbox": [100.0, 200.0, 200.0, 300.0], "label": "chair", "confidence": 0.9, "class_id": 56},
        {"bbox": [300.0, 200.0, 400.0, 300.0], "label": "chair", "confidence": 0.9, "class_id": 56},
    ], 0.0)
    id_a = res0[0]["track_id"]  # starts on left
    id_b = res0[1]["track_id"]  # starts on right

    # Frame 1: A moves to [180..280] (cx=230), B moves to [220..320] (cx=270)
    res1 = tracker.update([
        {"bbox": [180.0, 200.0, 280.0, 300.0], "label": "chair", "confidence": 0.9, "class_id": 56},
        {"bbox": [220.0, 200.0, 320.0, 300.0], "label": "chair", "confidence": 0.9, "class_id": 56},
    ], 1.0)

    # Frame 2: Crossing point. A moves to [260..360] (cx=310), B moves to [140..240] (cx=190)
    res2 = tracker.update([
        {"bbox": [140.0, 200.0, 240.0, 300.0], "label": "chair", "confidence": 0.9, "class_id": 56},  # object now on left
        {"bbox": [260.0, 200.0, 360.0, 300.0], "label": "chair", "confidence": 0.9, "class_id": 56},  # object now on right
    ], 2.0)
    
    # Find which ID is currently on the left (cx < 250) and which on the right (cx > 250)
    res2_left_id = [t["track_id"] for t in res2 if t["bbox"][0] < 200][0]
    res2_right_id = [t["track_id"] for t in res2 if t["bbox"][0] > 200][0]

    print("\n--- Crossing Paths Test Diagnostic ---")
    print(f"Initial positions: Track {id_a} on Left (cx=150), Track {id_b} on Right (cx=350)")
    print(f"After crossing:    Object on Left has Track ID {res2_left_id}, Object on Right has Track ID {res2_right_id}")

    swapped = (res2_left_id == id_a and res2_right_id == id_b)
    if swapped:
        print("Outcome: Identity SWAP occurred during crossing. (Object A was heading right but retained ID on left).")
        print("Explanation: This is a known, expected limitation of pure geometry-based (IoU + Centroid) trackers.")
        print("Without visual appearance embeddings (Re-ID), when two same-class objects cross paths,")
        print("each track greedily associates with whichever detection is geographically closest to its previous position,")
        print("causing the tracks to bounce back rather than pass through each other.")
    else:
        print("Outcome: Tracks maintained their true identities through crossing.")

    print("\n" + "=" * 80)
    print("ALL TRACKER ASSERTIONS PASSED SUCCESSFULLY.")
    print("=" * 80 + "\n")
