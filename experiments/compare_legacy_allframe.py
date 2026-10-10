"""Audit the raw-data rerun and render old/new full-frame backend comparisons."""
from pathlib import Path
from collections import Counter
import json
import cv2
import numpy as np
import scipy
from PIL import Image, ImageDraw
from pam_mosaic.backend.fusion import _canvas, _canvas_varied
from pam_mosaic.backend.poses import pose_to_matrix


def read(path):
    return json.loads(path.read_text())


def main():
    cv2.setNumThreads(4)
    root = Path(__file__).resolve().parents[1]/'outputs'
    old_full = root/'stable_rerun_20261006_full'
    new_full = root/'legacy_allframe_factors_full_20261010'
    old_final = root/'tile_edge_refinement_20261006'
    new_final = root/'legacy_allframe_factors_stitched_20261010'
    output = root/'legacy_allframe_factors_comparison_20261010'
    output.mkdir(exist_ok=True)
    old, new = read(old_full/'report.json'), read(new_full/'report.json')
    assert not new['background_correction'] and new['background_applied_frames'] == 0
    assert new['cache_frame_stage'] == 'after_reader'
    assert new['intensity_range'] == [.02, .25]
    assert [c['file_indices'] for c in old['components']] == [c['file_indices'] for c in new['components']]
    old_cache = sorted((old_full/'.run_cache').glob('frame_*.npy'))
    new_cache = sorted((new_full/'.run_cache').glob('frame_*.npy'))
    assert len(old_cache) == len(new_cache) == 1301
    cache_error = max(float(np.max(np.abs(np.load(a)-np.load(b)))) for a, b in zip(old_cache, new_cache))
    assert cache_error == 0, cache_error
    assert old['registrations'] == new['registrations'], 'Frontend changed; not a backend-only comparison'
    counts = Counter()
    translation, angles = [], []
    baseline_error = 0.
    for component in new['components']:
        name = Path(component['path']).name
        before = np.load(old_full/name/'poses_xytheta.npy')
        saved = np.load(new_full/name/'poses_baseline_xytheta.npy')
        baseline_error = max(baseline_error, float(np.max(np.abs(before-saved))))
        after = np.load(new_full/name/'poses_xytheta.npy')
        translation.extend(np.linalg.norm(after[:, :2]-before[:, :2], axis=1).tolist())
        angles.extend(np.abs(np.rad2deg(after[:, 2]-before[:, 2])).tolist())
        counts.update(component['extra_factors'])
    assert baseline_error < 1e-7, baseline_error
    finals = [read(p/'report.json') for p in [old_final, new_final]]
    assert len(finals[0]['groups']) == len(finals[1]['groups']) == 1
    grays, anchors = [], []
    anchor_name = Path(finals[0]['groups'][0]['tiles'][0]).name
    for full, final, report in zip([old_full, new_full], [old_final, new_final], finals):
        group = report['groups'][0]
        image = np.load(final/'mosaic_001/mosaic_float64.npy')
        weight = np.load(final/'mosaic_001/weight_map.npy')
        assert np.isfinite(image).all()
        gray = np.rint(255*np.clip((image-.02)/.23, 0, 1)).astype(np.uint8)
        gray[weight <= 0] = 0
        grays.append(gray)
        names = [Path(p).name for p in group['tiles']]
        poses = np.load(final/'mosaic_001/tile_poses_xytheta.npy')
        shapes = [np.load(full/name/'mosaic_float64.npy', mmap_mode='r').shape for name in names]
        canvas = _canvas_varied(poses, shapes)[2]
        index = names.index(anchor_name)
        frame_poses = np.load(full/anchor_name/'poses_xytheta.npy')
        component_canvas = _canvas(frame_poses, (500, 500))[2]
        anchors.append(canvas @ pose_to_matrix(poses[index]) @ component_canvas @ pose_to_matrix(frame_poses[0]))
    labels = ['Original backend / background OFF', 'All-frame factors / background OFF']
    for i, gray in enumerate(grays):
        cv2.imwrite(str(output/f'{i}_gray.png'), gray)
        cv2.imwrite(str(output/f'{i}_hot.png'), cv2.applyColorMap(gray, cv2.COLORMAP_HOT))
    for palette in ['hot', 'gray']:
        canvas = Image.new('RGB', (1400, 760), 'white')
        for i in range(2):
            im = Image.open(output/f'{i}_{palette}.png').convert('RGB')
            im.thumbnail((700, 710), Image.Resampling.LANCZOS)
            ImageDraw.Draw(canvas).text((700*i+10, 12), labels[i], fill='black')
            canvas.paste(im, (700*i+(700-im.width)//2, 40))
        canvas.save(output/f'comparison_{palette}.png')
    # Align only the origin via the same raw anchor frame, no image-fit or local warp.
    new_to_old = anchors[0] @ np.linalg.inv(anchors[1])
    rois = [(1400, 450, 400, 350), (850, 1200, 400, 350), (1750, 1800, 400, 350)]
    canvas = Image.new('RGB', (1000, 1440), 'white')
    for row, (x, y, width, height) in enumerate(rois):
        crop = np.array([[1., 0, -x], [0, 1, -y], [0, 0, 1]])
        for i, gray in enumerate(grays):
            matrix = crop if i == 0 else crop @ new_to_old
            pixels = cv2.warpAffine(gray, matrix[:2], (width, height))
            im = Image.fromarray(pixels).convert('RGB').resize((500, 438), Image.Resampling.NEAREST)
            ImageDraw.Draw(canvas).text((i*500+8, row*480+8), f'{labels[i]} / ROI {row+1}', fill='black')
            canvas.paste(im, (i*500, row*480+35))
    canvas.save(output/'vessel_rois.png')
    summary = {'background_correction': False, 'window': [.02, .25], 'gamma': 1,
               'runtime': {'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__, 'opencv_threads': cv2.getNumThreads()},
               'raw_full_rerun': True, 'processed_frames': 1301, 'components': len(new['components']),
               'cache_max_abs_difference': cache_error, 'frontend_exactly_equal': True,
               'baseline_pose_max_abs_difference': baseline_error,
               'extra_factors': dict(counts), 'old_final': finals[0]['groups'], 'new_final': finals[1]['groups'],
               'pose_translation_change_median_px': float(np.median(translation)),
               'pose_translation_change_max_px': float(np.max(translation)),
               'pose_rotation_change_max_deg': float(np.max(angles)),
               'roi_xywh': rois, 'roi_new_to_old_anchor_transform': new_to_old.tolist(),
               'notes': 'Same raw-data reconstruction, original frontend edges, display window and feather fusion. '
                        'All frame factors added within components; tile registration rerun on updated components. '
                        'Full previews preserve own canvases; ROIs align the same anchor frame only, not fitted to images.'}
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
