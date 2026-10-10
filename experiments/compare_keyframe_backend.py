"""Inspect full PAM512 baseline versus optional backend, plus sequential control."""
from pathlib import Path
import json
from collections import Counter
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from retired_backend.keyframes import BackendConfig,run_keyframe_backend
from retired_backend.poses import Edge,pose_to_matrix
from pam_mosaic.backend.fusion import fuse,_canvas
from pam_mosaic.frontend.view import matching_image
from pam_mosaic.frontend.registration import sift_inliers


def main():
    cv2.setNumThreads(4)
    root=Path(__file__).resolve().parents[1]/'outputs'
    base=root/'pam512_backend_baseline_20261010'
    new=root/'pam512_backend_factors_20261010'
    output=root/'pam512_backend_comparison_20261010'
    output.mkdir(exist_ok=True)
    br=json.loads((base/'report.json').read_text())
    nr=json.loads((new/'report.json').read_text())
    assert br['registrations']==nr['registrations']
    assert len(br['components'])==len(nr['components'])==1
    frames=[np.load(f) for f in sorted((base/'.run_cache').glob('frame_*.npy'))]
    other=sorted((new/'.run_cache').glob('frame_*.npy'))
    assert all(np.array_equal(a,np.load(b)) for a,b in zip(frames,other))
    indices=br['components'][0]['file_indices'];lookup={f:i for i,f in enumerate(indices)}
    edges=[Edge(lookup[d['reference']],lookup[d['moving']],np.asarray(d['matrix']),min(2.,d['inliers']/30))
           for d in br['registrations'] if d['valid']]
    matching=[matching_image(f,tuple(br['intensity_range'])) for f in frames]
    start=time.perf_counter()
    runs,audit=run_keyframe_backend(matching,edges,BackendConfig(()))
    assert len(runs)==1
    control=output/'sequential_control'
    fuse(frames,runs[0]['poses'],control,display_lo=0,display_hi=.06)
    np.save(control/'poses_xytheta.npy',runs[0]['poses'])
    (control/'backend.json').write_text(json.dumps({k:v for k,v in runs[0].items() if k!='poses'},indent=2))
    paths=[base/'component_001',control,new/'component_001']
    labels=['Original pose graph','Sequential keyframes, odometry only','Keyframes + geometry/intensity/loops']
    poses=[np.load(p/'poses_xytheta.npy') for p in paths]
    images=[];masks=[];offsets=[]
    for p,position in zip(paths,poses):
        data=np.load(p/'mosaic_float64.npy');weight=np.load(p/'weight_map.npy')
        assert np.isfinite(data).all() and np.isfinite(position).all()
        gray=np.rint(255*np.clip(data/.06,0,1)).astype(np.uint8);gray[weight<=0]=0
        images.append(gray);masks.append(weight>0)
        offsets.append(_canvas(position,frames[0].shape)[2])
    for palette in ['gray','hot']:
        canvas=Image.new('RGB',(1500,800),'white')
        for i,gray in enumerate(images):
            rgb=cv2.cvtColor(cv2.applyColorMap(gray,cv2.COLORMAP_HOT),cv2.COLOR_BGR2RGB) if palette=='hot' else gray
            im=Image.fromarray(rgb).convert('RGB');im.thumbnail((500,750),Image.Resampling.LANCZOS)
            ImageDraw.Draw(canvas).text((i*500+8,10),labels[i],fill='black')
            canvas.paste(im,(i*500+(500-im.width)//2,40))
        canvas.save(output/f'comparison_{palette}.png')
    a=np.array([[.0336202331845055,.686132032823773,-33.71458966275943],[-.513889681240328,.0250391580511848,644.378393234798],[0,0,1.]])
    flip=np.array([[-1,0,images[0].shape[1]-1],[0,1,0],[0,0,1.]])
    crop=np.array([[1,0,-329],[0,1,-238],[0,0,1.]])
    canvas=Image.new('RGB',(1812,484),'white')
    for i,(gray,offset) in enumerate(zip(images,offsets)):
        matrix=(crop@a@flip@offsets[0]@np.linalg.inv(offset))[:2]
        roi=cv2.warpAffine(gray,matrix,(302,217))
        im=Image.fromarray(roi).convert('RGB').resize((604,434),Image.Resampling.NEAREST)
        ImageDraw.Draw(canvas).text((i*604+8,12),labels[i],fill='black')
        canvas.paste(im,(i*604,50))
    canvas.save(output/'vessel_roi.png')
    # Pairs absent from explicit frontend/factor edge lists. Their image features
    # may still enter local maps, so these are diagnostics, not independent truth.
    used={(lookup[d['reference']],lookup[d['moving']]) for d in br['registrations']}
    newmeta=json.loads((new/'component_001/backend.json').read_text())
    used.update((e['reference'],e['moving']) for e in newmeta['factors'])
    candidates=[(i,i+7) for i in range(len(frames)-7) if (i,i+7) not in used]
    choices=[candidates[i] for i in np.linspace(0,len(candidates)-1,min(40,len(candidates)),dtype=int)]
    measurements=[];errors=[[],[],[]]
    for i,j in choices:
        correspondence=sift_inliers(matching[i],matching[j],20)
        if correspondence is None:
            continue
        src,dst,_=correspondence
        per_method=[]
        for m,position in enumerate(poses):
            ti,tj=pose_to_matrix(position[i]),pose_to_matrix(position[j])
            pi=dst@ti[:2,:2].T+ti[:2,2];pj=src@tj[:2,:2].T+tj[:2,2]
            error=float(np.median(np.linalg.norm(pi-pj,axis=1)))
            errors[m].append(error);per_method.append(error)
        measurements.append({'files':[indices[i],indices[j]],'median_errors_px':per_method})
    audit=json.loads((new/'backend_group_001.json').read_text())
    counts=Counter((m['kind'],m['accepted']) for m in audit['measurements'])
    summary={'frames':len(frames),'same_frontend_measurements':True,'same_preprocessed_arrays':True,
             'labels':labels,'sizes_xy':[[im.shape[1],im.shape[0]] for im in images],
             'keyframes':len(newmeta['keyframes']),
             'factor_counts':dict(Counter(e['kind'] for e in newmeta['factors'])),
             'measurement_counts':{str(k):v for k,v in counts.items()},
             'global_loop_updates':audit['global_loop_updates'],
             'diagnostic_pair_count':len(measurements),
             'median_pair_error_px':[float(np.median(e)) for e in errors],
             'p90_pair_error_px':[float(np.percentile(e,90)) for e in errors],
             'diagnostic_pairs':measurements,'control_and_comparison_seconds':time.perf_counter()-start,
             'notes':'Same preprocessing, frontend measurements, feather and display window 0..0.06. New backend is sequential, baseline graph is offline. Middle panel isolates loss from keyframe/odometry simplification. Pair checks are not anatomical ground truth and can share features with training; not independent validation.'}
    (output/'report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='diagnostic_pairs'},indent=2),flush=True)


if __name__=='__main__':
    main()
