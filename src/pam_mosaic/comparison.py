"""Read-only result comparison: fixed-input pixel audit and display-only alignment."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from .dataset import index_dataset
from .reconstruction import reconstruct
from .roundtrip import calibrate, fuse_if_aligned
from .poses import pose_to_matrix
from .registration import registration_view, register


def pose_lookup(root):
    result = {}
    report = json.loads((root/'report.json').read_text())
    for component in report['components']:
        path = Path(component['path'])
        poses = np.load(path/'poses_xytheta.npy')
        for index, pose in zip(component['file_indices'], poses):
            result[index] = (str(path), pose_to_matrix(pose))
    return result


def pair_matrix(lookup, a, b):
    if a not in lookup or b not in lookup or lookup[a][0] != lookup[b][0]:
        return None
    return (np.linalg.inv(lookup[a][1]) @ lookup[b][1])[:2]


def compare(root, data, backend='backend'):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    roots = {'current': Path('outputs/moderate_denoise_full'),
             'new_frontend': root/'frontend', 'new_backend': root/backend}
    lookups = {key: pose_lookup(value) for key, value in roots.items()}
    common = sorted(set.intersection(*(set(x) for x in lookups.values())))
    pairs = [(a, a+1) for a in common if all(pair_matrix(x, a, a+1) is not None for x in lookups.values())]
    # Uniform, deterministic sampling chosen without looking at measured errors.
    pairs = [pairs[i] for i in np.linspace(0, len(pairs)-1, min(100, len(pairs))).astype(int)] if pairs else []
    records = index_dataset(data)
    calibration = calibrate([r.path for r in records[:5]])
    required = {k for pair in pairs for k in pair}
    frames, decisions = {}, {}
    for record in records:
        if record.index in required:
            frames[record.index], decision = fuse_if_aligned(
                *reconstruct(record.path, calibration), suppress_bright_points=True)
            decisions[record.index] = decision.selected
    rows = []
    for a, b in pairs:
        reference, moving = registration_view(frames[a]), registration_view(frames[b])
        matrices = {key: pair_matrix(lookup, a, b) for key, lookup in lookups.items()}
        support = np.ones(reference.shape, np.uint8)
        for matrix in matrices.values():
            support &= cv2.warpAffine(np.ones(moving.shape, np.uint8), matrix, (500, 500), flags=cv2.INTER_NEAREST)
        support = cv2.erode(support, np.ones((11, 11), np.uint8)).astype(bool)
        if support.sum() < 1000:
            continue
        edge_ref = (cv2.Canny(reference, 35, 90) > 0) & support
        if edge_ref.sum() < 30:
            continue
        field_ref = distance_transform_edt(~edge_ref)
        row = {'reference': a, 'moving': b, 'common_pixels': int(support.sum())}
        for name, matrix in matrices.items():
            warped = cv2.warpAffine(moving, matrix, (500, 500))
            edge_mov = (cv2.Canny(warped, 35, 90) > 0) & support
            if edge_mov.sum() < 30:
                continue
            distances = np.r_[field_ref[edge_mov], distance_transform_edt(~edge_mov)[edge_ref]]
            corr = np.corrcoef(reference[support].astype(float), warped[support].astype(float))[0, 1]
            row[name] = {'correlation': float(corr) if np.isfinite(corr) else None,
                         'symmetric_edge_q75_px': float(np.percentile(distances, 75))}
        rows.append(row)
    metrics = {}
    complete = [r for r in rows if all(name in r and r[name]['correlation'] is not None for name in roots)]
    for name in roots:
        metrics[name] = {field: float(np.median([r[name][field] for r in complete]))
                         for field in ('correlation', 'symmetric_edge_q75_px')} if complete else {}
    changed = [np.linalg.norm(lookups['new_backend'][k][1][:2, 2]-lookups['new_frontend'][k][1][:2, 2]) for k in common]
    report = {'paired_audit_count': len(complete), 'sample_policy': '100_uniform_common_adjacent_pairs',
              'interpretation': 'same reconstructed inputs and intersection support; internal consistency, not anatomical ground truth',
              'median_metrics': metrics, 'pair_metrics': rows,
              'backend_translation_change_px': {'median': float(np.median(changed)), 'max': float(np.max(changed))}}
    source_decisions = {item['file_index']: item['selected'] for item in json.loads((root/'report.json').read_text())['roundtrip_decisions']}
    report['reconstruction_selection_check'] = {
        'checked_files': len(decisions),
        'different_from_new_full_run': sum(v != source_decisions[k] for k, v in decisions.items())}
    destination = root/('comparison' if backend == 'backend' else 'comparison_'+backend)
    destination.mkdir(exist_ok=True)
    (destination/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    # Align for display only. Neither pose graph nor floating-point result changes.
    images = {'baseline': Path('outputs/lm_stitched/mosaic_001'),
              'current': Path('outputs/moderate_denoise_stitched/mosaic_001'),
              'new_frontend': root/'frontend_stitched/mosaic_001',
              'new_backend': root/(backend+'_stitched')/'mosaic_001'}
    anchor = np.load(images['current']/'mosaic_float64.npy').astype(np.float32)
    small_anchor = cv2.resize(anchor, None, fx=.35, fy=.35)
    panels, alignment, full_views = [], {}, {}
    for name, path in images.items():
        image = np.load(path/'mosaic_float64.npy').astype(np.float32)
        small = cv2.resize(image, None, fx=.35, fy=.35)
        match = register(small_anchor, small, min_inliers=20, refine_geometry=False)
        if name == 'current':
            small = small_anchor
            full_views[name] = anchor
        elif match.valid:
            small = cv2.warpAffine(small, match.matrix, (small_anchor.shape[1], small_anchor.shape[0]))
            full_matrix = match.matrix.copy()
            full_matrix[:, 2] /= .35
            full_views[name] = cv2.warpAffine(image, full_matrix, (anchor.shape[1], anchor.shape[0]))
        else:
            small = cv2.resize(small, (small_anchor.shape[1], small_anchor.shape[0]))
        alignment[name] = {'valid': bool(match.valid), 'inliers': match.inliers}
        preview = cv2.applyColorMap(registration_view(small), cv2.COLORMAP_HOT)
        preview = cv2.resize(preview, (650, round(650*preview.shape[0]/preview.shape[1])))
        preview = cv2.copyMakeBorder(preview, 35, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(preview, name + ('' if match.valid else ' [unaligned]'), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        panels.append(preview)
    cv2.imwrite(str(destination/'overview.png'), np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
    detail_rows = []
    for fraction_x, fraction_y in ((.4, .5), (.65, .35), (.65, .65)):
        x, y = int(anchor.shape[1]*fraction_x), int(anchor.shape[0]*fraction_y)
        crops = []
        for name in ('current', 'new_frontend', 'new_backend'):
            if name not in full_views:
                break
            crop = full_views[name][y-180:y+180, x-180:x+180]
            preview = cv2.applyColorMap(registration_view(crop), cv2.COLORMAP_HOT)
            preview = cv2.copyMakeBorder(preview, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(preview, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            crops.append(preview)
        if len(crops) == 3:
            detail_rows.append(np.hstack(crops))
    if detail_rows:
        cv2.imwrite(str(destination/'details.png'), np.vstack(detail_rows))
    (destination/'display_alignment.json').write_text(json.dumps(alignment, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/paatam_2d'))
    parser.add_argument('--data', type=Path, default=Path('PAM_Skull_FYH_20260929'))
    parser.add_argument('--backend', default='backend')
    args = parser.parse_args()
    result = compare(args.root, args.data, args.backend)
    print(json.dumps({k: v for k, v in result.items() if k != 'pair_metrics'}, indent=2))
