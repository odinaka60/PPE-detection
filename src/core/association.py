from typing import Dict, List, Set, Tuple

Box = Tuple[int, int, int, int]


def containment(inner: Box, outer: Box) -> float:
    """Fraction of the `inner` box (a PPE item) that lies inside `outer` (a person).
    """
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / inner_area if inner_area else 0.0


def associate_ppe_to_persons(
    persons: List[dict],
    ppe_items: List[dict],
    min_containment: float = 0.5,
) -> Dict[int, Set[str]]:
    """Assign each PPE detection to the single person it overlaps most.

    Returns {track_id: set of PPE class names worn by that person}.
    A PPE item below the containment threshold for every person is dropped,
    so a stray detection in the background can't be credited to anyone.
    """
    worn: Dict[int, Set[str]] = {p["id"]: set() for p in persons}

    for item in ppe_items:
        best_id, best_score = None, min_containment
        for p in persons:
            score = containment(item["box"], p["box"])
            if score > best_score:
                best_id, best_score = p["id"], score
        if best_id is not None:
            worn[best_id].add(item["class"])

    return worn
