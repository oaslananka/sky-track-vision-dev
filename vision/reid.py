"""Target re-identification using appearance embeddings.

Phase 2: Maintains a stored embedding of the locked target. When the tracker
considers switching lock to a new track ID, the cosine similarity of the
candidate's appearance vs. the stored embedding must exceed a threshold.
This prevents following the wrong object after occlusion.

Uses a lightweight HSV-based histogram approach as a baseline.  The Hue
channel is weighted more heavily for better light-invariance; the Saturation
channel provides complementary information.  For higher accuracy, swap in
CLIP or OSNet embedding extraction.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from config.runtime_logging import log_event

logger = logging.getLogger("skytrackvision.vision.reid")


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Return cosine similarity for two normalized appearance embeddings."""
    if a is None or b is None:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_hsv_embedding(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Extract a light-invariant HSV histogram embedding for a detection crop."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_hist, _ = np.histogram(hsv_crop[:, :, 0].ravel(), bins=36, range=(0, 180))
    h_hist = h_hist.astype(np.float32) * 2.0
    s_hist, _ = np.histogram(hsv_crop[:, :, 1].ravel(), bins=16, range=(0, 256))
    s_hist = s_hist.astype(np.float32)

    combined = np.concatenate([h_hist, s_hist])
    norm = float(np.linalg.norm(combined))
    if norm <= 1e-8:
        return None
    combined /= norm
    return combined


class AppearanceStore:
    """Store and compare appearance features for target re-identification."""

    def __init__(self, similarity_threshold: float = 0.55, bins: int = 32) -> None:
        self._similarity_threshold = similarity_threshold
        self._bins = bins
        self._locked_embedding: np.ndarray | None = None
        self._locked_track_id: int | None = None

    @property
    def has_lock(self) -> bool:
        return self._locked_embedding is not None

    @property
    def locked_track_id(self) -> int | None:
        return self._locked_track_id

    def lock_appearance(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        track_id: int,
    ) -> None:
        """Extract and store the appearance embedding when target is first locked."""
        embedding = self._extract_embedding(frame, bbox)
        if embedding is not None:
            self._locked_embedding = embedding
            self._locked_track_id = track_id
            log_event(
                logger,
                logging.DEBUG,
                "reid.lock",
                "Appearance locked for target",
                track_id=track_id,
            )

    def match_score(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Compute cosine similarity between locked and candidate appearance."""
        if self._locked_embedding is None:
            return 1.0  # No lock → accept everything

        candidate = self._extract_embedding(frame, bbox)
        return cosine_similarity(self._locked_embedding, candidate)

    def should_accept_switch(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        new_track_id: int,
        frames_since_seen: int = 0,
    ) -> bool:
        """Check if a track ID switch should be accepted based on appearance similarity."""
        if self._locked_embedding is None:
            return True  # No stored appearance, accept any

        if new_track_id == self._locked_track_id:
            return True  # Same ID, no switch

        score = self.match_score(frame, bbox)
        # Adaptive threshold: lower bar after long occlusions
        adaptive_threshold = max(0.35, self._similarity_threshold - frames_since_seen * 0.03)
        accepted = score >= adaptive_threshold

        log_event(
            logger,
            logging.INFO,
            "reid.switch_check",
            "Track ID switch evaluation",
            current_id=self._locked_track_id,
            candidate_id=new_track_id,
            similarity=round(score, 4),
            threshold=round(adaptive_threshold, 4),
            base_threshold=self._similarity_threshold,
            frames_since_seen=frames_since_seen,
            accepted=accepted,
        )

        if accepted:
            # Update lock to new ID (same-looking target, new tracker assignment)
            self._locked_track_id = new_track_id

        return accepted

    def reset(self) -> None:
        """Clear stored appearance."""
        self._locked_embedding = None
        self._locked_track_id = None

    def _extract_embedding(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """Extract an HSV-based histogram feature vector from the target crop.

        Uses Hue (36 bins, 2× weight) + Saturation (16 bins) for better
        light-invariance compared to raw BGR histograms.  For production-grade
        Re-ID, replace with CLIP (ViT-B/32) or OSNet feature extraction.
        """
        return extract_hsv_embedding(frame, bbox)
