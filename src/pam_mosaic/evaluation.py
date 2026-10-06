"""Common-support image checks, independent of optimizer residual weighting."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from .registration import registration_view


def pair_scores(reference, moving, matrices):
    reference, moving = registration_view(reference), registration_view(moving)
    h, w = reference.shape
    mask = np.ones((h, w), np.uint8)
    for matrix in matrices:
        mask &= cv2.warpAffine(np.ones(moving.shape, np.uint8), matrix[:2], (w, h), flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8)).astype(bool)
    if mask.sum() < 1000:
        return None
    ref_edges, mov_edges = cv2.Canny(reference, 35, 90)>0, cv2.Canny(moving, 35, 90)>0
    yr, xr = np.nonzero(ref_edges & mask)
    ym, xm = np.nonzero(mov_edges)
    if min(len(xr), len(xm)) < 30:
        return None
    ref_points, mov_points = np.column_stack((xr, yr)), np.column_stack((xm, ym))
    ref_field, mov_field = distance_transform_edt(~ref_edges), distance_transform_edt(~mov_edges)
    result = []
    for matrix in matrices:
        matrix = matrix[:2]
        warped = cv2.warpAffine(moving, matrix, (w, h))
        if min(reference[mask].std(), warped[mask].std()) < 1:
            return None
        correlation = float(np.corrcoef(reference[mask].astype(float), warped[mask].astype(float))[0, 1])
        forward = mov_points @ matrix[:, :2].T + matrix[:, 2]
        valid = map_coordinates(mask.astype(float), forward[:, ::-1].T, order=0, mode='constant', cval=0) > .5
        backward = (ref_points-matrix[:, 2]) @ matrix[:, :2]
        values = np.r_[map_coordinates(ref_field, forward[valid, ::-1].T, order=1, mode='constant', cval=20),
                       map_coordinates(mov_field, backward[:, ::-1].T, order=1, mode='constant', cval=20)]
        result.append({'correlation': correlation, 'edge_q75_px': float(np.percentile(values, 75)),
                       'common_pixels': int(mask.sum())})
    return result
