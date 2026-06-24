from __future__ import annotations

_COMPATIBILITY: dict[str, tuple[str, ...]] = {
    "vehicle": ("car", "truck", "bus", "motorcycle", "bicycle"),
    # Keep truck targeting strict enough to avoid locking generic cars.
    # Bus remains compatible with truck due to frequent confusion in distance views.
    "truck": ("truck", "bus"),
    "bus": ("bus", "truck"),
    "car": ("car",),
}


def is_priority_compatible(priority_class: str | None, detected_class: str | None) -> bool:
    if not priority_class or not detected_class:
        return False
    priority = priority_class.strip().lower()
    detected = detected_class.strip().lower()
    if priority == detected:
        return True
    return detected in _COMPATIBILITY.get(priority, ())


def classes_for_priority(priority_class: str | None) -> tuple[str, ...]:
    if not priority_class:
        return ()
    priority = priority_class.strip().lower()
    compatible = _COMPATIBILITY.get(priority)
    if compatible:
        return compatible
    return (priority,)
