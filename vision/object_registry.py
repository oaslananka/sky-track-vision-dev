from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autonomy.contracts import Detection
from vision.reid import cosine_similarity, extract_hsv_embedding

_VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})


@dataclass(slots=True)
class RegistryObject:
    object_id: int
    class_name: str
    first_seen_ns: int
    last_seen_ns: int
    hits: int = 0
    best_confidence: float = 0.0
    mean_confidence: float = 0.0
    last_center: tuple[float, float] = (0.0, 0.0)
    last_area: float = 0.0
    raw_track_ids: set[int] = field(default_factory=set)
    embedding: np.ndarray | None = None


@dataclass(slots=True)
class RegistrySummary:
    total_objects: int
    active_objects: int
    unique_object_counts: dict[str, int]
    unique_vehicle_count: int
    active_object_views: list[dict[str, Any]]
    merge_count: int


class MultiObjectRegistry:
    """Mission-wide object memory with track-id merge logic.

    ByteTrack IDs are treated as hints, not final identity. When a new raw track
    appears, the registry attempts to merge it into a previously observed object
    using class match, recency, center distance, and appearance similarity.
    """

    def __init__(
        self,
        *,
        active_window_s: float = 1.5,
        merge_window_s: float = 8.0,
        max_center_distance_px: float = 160.0,
        similarity_threshold: float = 0.58,
        min_hits: int = 1,
        max_active_views: int = 8,
    ) -> None:
        self._active_window_ns = int(active_window_s * 1_000_000_000)
        self._merge_window_ns = int(merge_window_s * 1_000_000_000)
        self._max_center_distance_px = max_center_distance_px
        self._similarity_threshold = similarity_threshold
        self._min_hits = min_hits
        self._max_active_views = max_active_views
        self._objects: dict[int, RegistryObject] = {}
        self._track_to_object: dict[int, int] = {}
        self._next_object_id = 1
        self._merge_count = 0

    def update(
        self,
        detections: list[Detection],
        *,
        frame: np.ndarray | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                continue
            class_name = detection.class_name.strip().lower()
            object_id = self._track_to_object.get(track_id)
            embedding = extract_hsv_embedding(frame, detection.bbox) if frame is not None else None
            if object_id is None:
                object_id = self._resolve_object_id(
                    detection,
                    class_name=class_name,
                    timestamp_ns=timestamp_ns,
                    embedding=embedding,
                )
                self._track_to_object[track_id] = object_id
            obj = self._objects[object_id]
            self._update_object(
                obj,
                detection,
                class_name=class_name,
                timestamp_ns=timestamp_ns,
                embedding=embedding,
            )

    def summary(self, *, timestamp_ns: int | None = None) -> RegistrySummary:
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        eligible = [obj for obj in self._objects.values() if obj.hits >= self._min_hits]
        counts = Counter(obj.class_name for obj in eligible)
        active = [
            obj for obj in eligible if (timestamp_ns - obj.last_seen_ns) <= self._active_window_ns
        ]
        active.sort(key=lambda obj: obj.last_seen_ns, reverse=True)
        active_views = [
            {
                "object_id": obj.object_id,
                "class_name": obj.class_name,
                "hits": obj.hits,
                "raw_track_ids": sorted(obj.raw_track_ids),
                "last_center": [round(obj.last_center[0], 2), round(obj.last_center[1], 2)],
                "best_confidence": round(obj.best_confidence, 3),
                "age_s": round((timestamp_ns - obj.first_seen_ns) / 1_000_000_000, 3),
                "last_seen_s_ago": round((timestamp_ns - obj.last_seen_ns) / 1_000_000_000, 3),
            }
            for obj in active[: self._max_active_views]
        ]
        vehicle_count = sum(1 for obj in eligible if obj.class_name in _VEHICLE_CLASSES)
        return RegistrySummary(
            total_objects=len(eligible),
            active_objects=len(active),
            unique_object_counts=dict(sorted(counts.items())),
            unique_vehicle_count=vehicle_count,
            active_object_views=active_views,
            merge_count=self._merge_count,
        )

    def _resolve_object_id(
        self,
        detection: Detection,
        *,
        class_name: str,
        timestamp_ns: int,
        embedding: np.ndarray | None,
    ) -> int:
        best_candidate: tuple[float, int] | None = None
        for object_id, obj in self._objects.items():
            if obj.class_name != class_name:
                continue
            age_ns = timestamp_ns - obj.last_seen_ns
            if age_ns > self._merge_window_ns:
                continue
            distance = self._center_distance(detection.center, obj.last_center)
            if distance > self._max_center_distance_px:
                continue
            area_ratio = max(detection.area, obj.last_area, 1.0) / max(
                min(detection.area, obj.last_area),
                1.0,
            )
            if area_ratio > 2.5:
                continue
            similarity = cosine_similarity(embedding, obj.embedding)
            very_close_recent = distance <= (self._max_center_distance_px * 0.15) and age_ns <= min(
                self._merge_window_ns, 1_000_000_000
            )
            plausible_revisit = (
                similarity >= (self._similarity_threshold - 0.08)
                and distance <= (self._max_center_distance_px * 0.65)
                and area_ratio <= 1.8
                and age_ns <= min(self._merge_window_ns, 3_000_000_000)
            )
            if (
                similarity < self._similarity_threshold
                and not very_close_recent
                and not plausible_revisit
            ):
                continue
            score = similarity - (distance / max(self._max_center_distance_px, 1.0))
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, object_id)
        if best_candidate is not None:
            self._merge_count += 1
            return best_candidate[1]
        object_id = self._next_object_id
        self._next_object_id += 1
        self._objects[object_id] = RegistryObject(
            object_id=object_id,
            class_name=class_name,
            first_seen_ns=timestamp_ns,
            last_seen_ns=timestamp_ns,
        )
        return object_id

    def _update_object(
        self,
        obj: RegistryObject,
        detection: Detection,
        *,
        class_name: str,
        timestamp_ns: int,
        embedding: np.ndarray | None,
    ) -> None:
        obj.class_name = class_name
        obj.last_seen_ns = timestamp_ns
        obj.hits += 1
        obj.best_confidence = max(obj.best_confidence, detection.confidence)
        obj.mean_confidence = (
            detection.confidence
            if obj.hits == 1
            else ((obj.mean_confidence * (obj.hits - 1)) + detection.confidence) / obj.hits
        )
        obj.last_center = detection.center
        obj.last_area = detection.area
        if detection.track_id is not None:
            obj.raw_track_ids.add(detection.track_id)
        if embedding is not None:
            if obj.embedding is None:
                obj.embedding = embedding
            else:
                obj.embedding = ((obj.embedding * 0.7) + (embedding * 0.3)).astype(np.float32)
                norm = float(np.linalg.norm(obj.embedding))
                if norm > 1e-8:
                    obj.embedding /= norm

    @staticmethod
    def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))
