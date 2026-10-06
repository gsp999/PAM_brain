"""Select overlap candidates and identify connected image groups."""
from __future__ import annotations

import cv2
import numpy as np
from .registration import registration_view


def appearance_vector(image: np.ndarray) -> np.ndarray:
    """Compact blurred image descriptor for retrieving similar fields of view."""
    view = registration_view(image)
    small = cv2.resize(cv2.GaussianBlur(view, (9, 9), 0), (24, 24), interpolation=cv2.INTER_AREA)
    vector = small.astype(np.float32).ravel()
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def candidate_pairs(frames: list[np.ndarray], *, temporal_window: int = 5,
                    appearance_neighbors: int = 6) -> list[tuple[int, int]]:
    count = len(frames)
    pairs = {(i - 1, i) for i in range(1, count)}
    # Gaps of a few acquisitions are common during a rapid view change.
    for i in range(count):
        for gap in (2, 3, 5):
            if gap <= temporal_window and i + gap < count:
                pairs.add((i, i + gap))
    if appearance_neighbors <= 0:
        return sorted(pairs, key=lambda p: (p[1]-p[0], p[0]))
    descriptors = np.stack([appearance_vector(frame) for frame in frames])
    similarity = descriptors @ descriptors.T
    for i in range(count):
        similarity[i, max(0, i-temporal_window):min(count, i+temporal_window+1)] = -2
        choices = np.argpartition(similarity[i], -min(appearance_neighbors, count))[-min(appearance_neighbors, count):]
        for j in choices:
            if similarity[i, j] >= 0.65:
                pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs, key=lambda p: (p[1]-p[0], p[0]))


def connected_components(count: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(count))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda group: group[0])
