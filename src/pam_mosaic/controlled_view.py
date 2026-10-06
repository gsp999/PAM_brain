"""Scientific comparisons using shared pose gauge, with no display registration."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .fusion import _canvas, _canvas_varied
from .poses import matrix_to_pose, pose_to_matrix
from .registration import registration_view


def _panel(view, label, width=500, height=420):
    ratio = min(width/view.shape[1], height/view.shape[0])
    resized = cv2.resize(view, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    result = np.zeros((height+32, width, 3), np.uint8)
    x, y = (width-resized.shape[1])//2, 32+(height-resized.shape[0])//2
    result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    cv2.putText(result, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1)
    return result


def render(root, cache):
    manifest = json.loads((cache/'manifest.json').read_text())
    frames = {r['index']: r['path'] for r in manifest['frames']}
    summary = json.loads((root/'summary.json').read_text())
    overview_rows, overlay_rows = [], []
    for scenario in summary['scenarios']:
        reports = {mode: json.loads((root/scenario/mode/'report.json').read_text()) for mode in ('legacy','line','backend')}
        components = {mode: max(r['components'], key=lambda c:c['frames']) for mode,r in reports.items()}
        common_gauge = len({c['first_file'] for c in components.values()}) == 1
        if common_gauge:
            images, placements = [], []
            for component in components.values():
                poses = np.load(Path(component['path'])/'poses_xytheta.npy')
                _, _, offset = _canvas(poses, (500,500))
                placements.append(matrix_to_pose(np.linalg.inv(offset)))
                images.append(np.load(Path(component['path'])/'mosaic_float64.npy'))
            placements = np.asarray(placements)
            width,height,offset = _canvas_varied(placements,[i.shape for i in images])
            panels = []
            for mode,image,placement in zip(components,images,placements):
                image = cv2.warpAffine(image, (offset @ pose_to_matrix(placement))[:2], (width,height))
                preview = cv2.applyColorMap(registration_view(image), cv2.COLORMAP_HOT)
                panels.append(_panel(preview, f'{scenario}: {mode}'))
            overview_rows.append(np.hstack(panels))
        evaluation = json.loads((root/scenario/'evaluation.json').read_text())
        # Explicitly select the worst correlation regression to expose failure,
        # not the best-looking case. This is a diagnostic, not another metric.
        pair = min(evaluation['pairs'], key=lambda r:r['backend']['correlation']-r['legacy']['correlation'])
        a,b = pair['reference'], pair['moving']
        reference,moving = (registration_view(np.load(frames[k])) for k in (a,b))
        panels = []
        for mode,report in reports.items():
            component = next(c for c in report['components'] if a in c['file_indices'] and b in c['file_indices'])
            poses = np.load(Path(component['path'])/'poses_xytheta.npy')
            ia,ib = (component['file_indices'].index(k) for k in (a,b))
            matrix = (np.linalg.inv(pose_to_matrix(poses[ia])) @ pose_to_matrix(poses[ib]))[:2]
            warped = cv2.warpAffine(moving,matrix,(500,500))
            # BGR: fixed reference red, projected observation cyan.
            overlay = np.stack((warped,warped,reference),axis=2)
            panels.append(_panel(overlay, f'{a}-{b}: {mode}',height=500))
        overlay_rows.append(np.hstack(panels))
    destination = root/'comparison'
    destination.mkdir(exist_ok=True)
    if overview_rows:
        cv2.imwrite(str(destination/'overview.png'),np.vstack(overview_rows))
    if overlay_rows:
        cv2.imwrite(str(destination/'worst_pairs.png'),np.vstack(overlay_rows))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/controlled_v2/segments_dense'))
    parser.add_argument('--cache',type=Path,default=Path('outputs/controlled_v2/inputs'))
    args=parser.parse_args()
    render(args.root,args.cache)
