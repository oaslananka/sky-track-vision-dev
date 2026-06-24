from __future__ import annotations

from collections import Counter

from autonomy.contracts import Detection, SceneInsight


class SceneReasoner:
    """Summarize raw detections into mission-friendly scene state."""

    def analyze(self, detections: list[Detection]) -> SceneInsight:
        if not detections:
            return SceneInsight(
                class_counts={},
                dominant_class=None,
                scene_state="empty",
                activity_score=0.0,
                summary_text="No tracked activity in frame.",
            )
        counts = Counter(detection.class_name for detection in detections)
        dominant_class, dominant_count = counts.most_common(1)[0]
        activity_score = min(1.0, len(detections) / 10.0)
        if dominant_class == "person" and dominant_count >= 3:
            scene_state = "high_pedestrian_density"
        elif len(detections) >= 5:
            scene_state = "high_activity"
        else:
            scene_state = "low_activity"
        return SceneInsight(
            class_counts=dict(counts),
            dominant_class=dominant_class,
            scene_state=scene_state,
            activity_score=activity_score,
            summary_text=f"{len(detections)} objects detected; dominant={dominant_class}",
        )
