"""Compare completed rerun with both preserved references; no pose changes."""
import json
from pathlib import Path
import cv2
import numpy as np
from pam_mosaic.registration import register, registration_view
from pam_mosaic.audit import audit_tile_seams
from pam_mosaic.fusion import _canvas
from pam_mosaic.poses import pose_to_matrix


def angles(source, stitched):
    report=json.loads((source/'report.json').read_text())
    by_path={str(Path(c['path']).resolve()):c for c in report['components']}
    group=json.loads((stitched/'report.json').read_text())['groups'][0]
    tiles=np.load(Path(group['path'])/'tile_poses_xytheta.npy')
    result={}
    for tile,path in zip(tiles,group['tiles']):
        component=by_path[str(Path(path).resolve())]
        poses=np.load(Path(path)/'poses_xytheta.npy')
        _,_,offset=_canvas(poses,(500,500))
        for index,pose in zip(component['file_indices'],poses):
            transform=pose_to_matrix(tile) @ offset @ pose_to_matrix(pose)
            result[index]=float(np.degrees(np.arctan2(transform[1,0],transform[0,0])))
    return result


def main():
    cv2.setNumThreads(2)
    roots={'baseline':(Path('outputs/lm_full'),Path('outputs/lm_stitched')),
           'previous':(Path('outputs/moderate_denoise_full'),Path('outputs/moderate_denoise_stitched')),
           'rerun':(Path('outputs/frontend_only_full'),Path('outputs/frontend_only_stitched'))}
    destination=roots['rerun'][1]/'comparison';destination.mkdir(exist_ok=True)
    summary,images,world_angles={},{},{}
    for name,(source,stitched) in roots.items():
        report=json.loads((source/'report.json').read_text()); tiles=json.loads((stitched/'report.json').read_text())
        group=tiles['groups'][0];path=Path(group['path'])
        image=np.load(path/'mosaic_float64.npy').astype(np.float32)
        weights=np.load(path/'weight_map.npy',mmap_mode='r')
        summary[name]={'input_files':sum(c['frames'] for c in report['components']),
                       'components':len(report['components']),
                       'adjacent_success':sum(r['valid'] and r['moving']==r['reference']+1 for r in report['registrations']),
                       'stitched_files':group['source_files'],'selected_tiles':len(group['tiles']),
                       'dimensions_xy':list(image.shape[::-1]),'valid_pixels':int((weights>0).sum())}
        images[name]=image;world_angles[name]=angles(source,stitched)
    common=sorted(set(world_angles['previous']) & set(world_angles['rerun']))
    delta=np.array([world_angles['rerun'][k]-world_angles['previous'][k] for k in common])
    delta=(delta+180)%360-180
    offset=float(np.median(delta));residual=np.abs((delta-offset+180)%360-180)
    summary['rotation_comparison']={'common_files':len(common),'global_offset_deg':offset,
                                    'after_offset_abs_p50_p95_max_deg':np.percentile(residual,[50,95,100]).tolist()}
    summary['coverage_change']={'gained_files':sorted(set(world_angles['rerun'])-set(world_angles['previous'])),
                                'lost_files':sorted(set(world_angles['previous'])-set(world_angles['rerun']))}
    anchor=images['previous'];scale=.35
    small_ref=cv2.resize(anchor,None,fx=scale,fy=scale)
    panels=[];full_views={};alignment={}
    for name,image in images.items():
        small=cv2.resize(image,None,fx=scale,fy=scale)
        if name=='previous':
            matrix=np.eye(2,3);valid=True
        else:
            matched=register(small_ref,small,min_inliers=20,refine_geometry=False)
            matrix,valid=matched.matrix,matched.valid
        alignment[name]={'display_alignment_valid':bool(valid),'matrix_at_scale035':matrix.tolist()}
        if valid:
            small=cv2.warpAffine(small,matrix,(small_ref.shape[1],small_ref.shape[0]))
            original=matrix.copy();original[:,2]/=scale
            full_views[name]=cv2.warpAffine(image,original,(anchor.shape[1],anchor.shape[0]))
        else:
            small=cv2.resize(small,(small_ref.shape[1],small_ref.shape[0]))
        preview=cv2.applyColorMap(registration_view(small),cv2.COLORMAP_HOT)
        preview=cv2.resize(preview,(550,round(550*preview.shape[0]/preview.shape[1])))
        preview=cv2.copyMakeBorder(preview,32,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(preview,name+('' if valid else ' [unaligned]'),(8,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
        panels.append(preview)
    cv2.imwrite(str(destination/'overview.png'),np.hstack(panels))
    rows=[]
    if len(full_views)==3:
        for fx,fy in ((.4,.5),(.65,.35),(.65,.65)):
            x,y=int(anchor.shape[1]*fx),int(anchor.shape[0]*fy);row=[]
            for name,image in full_views.items():
                preview=cv2.applyColorMap(registration_view(image[y-180:y+180,x-180:x+180]),cv2.COLORMAP_HOT)
                preview=cv2.copyMakeBorder(preview,32,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(preview,name,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);row.append(preview)
            rows.append(np.hstack(row))
        cv2.imwrite(str(destination/'details.png'),np.vstack(rows))
    summary['display_only_alignment']=alignment
    for name in ('previous','rerun'):
        source,stitched=roots[name]
        audit=audit_tile_seams(source,stitched,destination/('seams_'+name))
        summary[name]['seam_risk_counts']=audit['risk_counts']
    summary['interpretation']='Image-based heuristic comparisons, not anatomical truth; display alignment does not modify saved poses or arrays.'
    (destination/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
