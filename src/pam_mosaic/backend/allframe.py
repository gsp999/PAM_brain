"""Optional offline factors; retain every frame and every original graph edge."""
from collections import Counter
import cv2
import numpy as np
from .poses import Edge, pose_to_matrix, optimize_poses
from .factors import geometry_measurement, intensity_measurement, edge_points, homogeneous, spread
from ..frontend.registration import register, sift_inliers, _overlap, _warped_correlation
from ..frontend.view import registration_view
from ..frontend.candidates import appearance_vector

FACTOR_KINDS = ('geometry', 'intensity', 'loop')


def parse_factors(value):
    """Canonical order, explicit opt-in; reject typos before processing data."""
    if isinstance(value, str):
        value = value.strip()
        if value == 'none':
            return ()
        if value == 'all':
            return FACTOR_KINDS
        value = tuple(part.strip() for part in value.split(','))
    selected = set(value)
    if not selected.issubset(FACTOR_KINDS):
        raise ValueError('backend factors must be none, all, or geometry,intensity,loop')
    return tuple(kind for kind in FACTOR_KINDS if kind in selected)


def refine_all_frames(frames, baseline, edges, *, factors=FACTOR_KINDS,
                      sequence_indices=None, excluded_pairs=()):
    """Frames are matching views, baseline is the original full-graph optimum.

    Frozen local maps exclude the current frame. New measurements are relative
    edges, not a simultaneous photometric BA. No cross-component reconnection.
    sequence_indices preserves acquisition-order gaps inside a connected group.
    excluded_pairs supports reproducible diagnostic holdouts (local node IDs).
    """
    factors = parse_factors(factors)
    count = len(frames)
    sequence = np.arange(count) if sequence_indices is None else np.asarray(sequence_indices)
    if baseline.shape != (count, 3) or sequence.shape != (count,):
        raise ValueError('Frames, poses and acquisition-order indices must agree')
    if count > 1 and np.any(np.diff(sequence) <= 0):
        raise ValueError('Acquisition-order indices must increase strictly')
    summary = {'algorithm': 'allframe_factors_v1', 'keyframes_used': False,
               'all_original_edges_preserved': True, 'frames': count,
               'original_edges': len(edges), 'enabled_factors': list(factors),
               'extra_factors': {}, 'measurements': [], 'extra_edges': []}
    if not factors or count < 2:
        return baseline.copy(), summary
    views = [registration_view(frame) for frame in frames]
    features = None
    if 'intensity' in factors or 'loop' in factors:
        sift = cv2.SIFT_create(nfeatures=1200)
        features = [sift.detectAndCompute(view, None) for view in views]
    points = [edge_points(view) for view in views] if 'geometry' in factors else None
    poses = {i: pose_to_matrix(p) for i, p in enumerate(baseline)}
    centers = np.array([poses[i][:2, :2] @ np.array([f.shape[1]//2, f.shape[0]//2])
                        + poses[i][:2, 2] for i, f in enumerate(frames)])
    appearances = np.stack([appearance_vector(f) for f in frames]) if 'loop' in factors else None
    descriptors = appearances @ appearances.T if appearances is not None else None
    pairset = {tuple(sorted((e.reference, e.moving))) for e in edges}
    forbidden = {tuple(sorted(pair)) for pair in excluded_pairs}
    extra, audits = [], summary['measurements']
    for k in range(count):
        if 'geometry' in factors or 'intensity' in factors:
            choices = sorted((j for j in range(count) if j != k and abs(sequence[j]-sequence[k]) <= 30
                              and tuple(sorted((j, k))) not in forbidden),
                             key=lambda j: float(np.linalg.norm(centers[j]-centers[k])))[:10]
            neighbors = [j for j in choices if _overlap((np.linalg.inv(poses[j])@poses[k])[:2],
                                                       frames[j].shape, frames[k].shape) > .3][:6]
            if len(neighbors) >= 2:
                reference = neighbors[0]
                for kind in factors:
                    if kind == 'loop':
                        continue
                    if kind == 'geometry':
                        measured, audit = geometry_measurement(k, neighbors, points, poses, reference)
                    else:
                        measured, audit, _ = intensity_measurement(k, neighbors, frames, features, poses, reference)
                    audit.update(frame=k, reference=reference)
                    audits.append(audit)
                    if measured is not None:
                        matrix, information = measured
                        extra.append(Edge(reference, k, matrix[:2], .35, kind, information))
        if 'loop' in factors:
            checked = 0
            for j in np.argsort(-descriptors[k]):
                j = int(j)
                pair = tuple(sorted((j, k)))
                if sequence[j] >= sequence[k]-30 or pair in pairset or pair in forbidden or descriptors[k, j] < .55:
                    continue
                checked += 1
                if checked > 3:
                    break
                match = register(frames[j], frames[k], min_inliers=25,
                                 reference_features=features[j], moving_features=features[k])
                audit = {'frame': k, 'reference': j, 'kind': 'loop',
                         'accepted': False, 'reason': 'pair_validation'}
                if match.valid and match.inliers >= 30 and match.median_error <= 2.5:
                    correspondence = sift_inliers(frames[j], frames[k], 25,
                                                 reference_features=features[j], moving_features=features[k])
                    backward = register(frames[k], frames[j], min_inliers=25,
                                        reference_features=features[k], moving_features=features[j])
                    if correspondence is not None and min(spread(correspondence[0]), spread(correspondence[1])) >= 20 and backward.valid:
                        cycle = homogeneous(match.matrix) @ homogeneous(backward.matrix)
                        overlap, corr = _warped_correlation(frames[j], frames[k], match.matrix)
                        accepted = bool(np.linalg.norm(cycle[:2, 2]) < 2
                                        and abs(np.arctan2(cycle[1, 0], cycle[0, 0])) < np.deg2rad(.5)
                                        and overlap > .3 and np.isfinite(corr) and corr > .55)
                        audit.update(accepted=accepted, reason='accepted' if accepted else 'consistency')
                        if accepted:
                            extra.append(Edge(j, k, match.matrix, .7, 'loop'))
                            pairset.add(pair)
                audits.append(audit)
        if (k+1) % 100 == 0:
            print(f'backend factors {k+1}/{count}: added {len(extra)}', flush=True)
    summary['extra_factors'] = dict(Counter(e.kind for e in extra))
    summary['extra_edges'] = [dict(reference=e.reference, moving=e.moving, kind=e.kind,
                                   matrix=e.matrix.tolist(), weight=e.weight,
                                   sqrt_information=e.sqrt_information.tolist() if e.sqrt_information is not None else None)
                              for e in extra]
    optimized = optimize_poses(baseline, list(edges)+extra) if extra else baseline.copy()
    return optimized, summary
