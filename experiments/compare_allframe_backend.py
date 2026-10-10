"""Keep every original node/edge; test extra image-derived factors offline."""
from pathlib import Path
from collections import Counter
import json
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
from retired_backend.poses import Edge,pose_to_matrix,optimize_poses
from retired_backend.factors import (geometry_measurement,intensity_measurement,
    edge_points,homogeneous,spread)
from pam_mosaic.frontend.registration import register,sift_inliers,_overlap,_warped_correlation
from pam_mosaic.frontend.view import matching_image,registration_view
from pam_mosaic.frontend.candidates import appearance_vector
from pam_mosaic.backend.fusion import fuse,_canvas


def main():
    cv2.setNumThreads(4)
    root=Path(__file__).resolve().parents[1]/'outputs'
    source=root/'pam512_backend_baseline_20261010'
    output=root/'pam512_allframe_backend_20261010'
    output.mkdir(exist_ok=True)
    report=json.loads((source/'report.json').read_text())
    indices=report['components'][0]['file_indices'];lookup={v:i for i,v in enumerate(indices)}
    baseline=np.load(source/'component_001/poses_xytheta.npy')
    frames=[np.load(p) for p in sorted((source/'.run_cache').glob('frame_*.npy'))]
    assert len(frames)==len(baseline)==569
    matching=[matching_image(f,(0,.06)) for f in frames]
    edges=[Edge(lookup[d['reference']],lookup[d['moving']],np.asarray(d['matrix']),
                min(2.,d['inliers']/30)) for d in report['registrations'] if d['valid']]
    pairset={(e.reference,e.moving) for e in edges}
    olddiag=json.loads((root/'pam512_backend_comparison_20261010/report.json').read_text())
    validation=[(lookup[d['files'][0]],lookup[d['files'][1]]) for d in olddiag['diagnostic_pairs']]
    forbidden={tuple(sorted(pair)) for pair in validation}
    views=[registration_view(f) for f in matching]
    sift=cv2.SIFT_create(nfeatures=1200)
    features=[sift.detectAndCompute(v,None) for v in views]
    points=[edge_points(v) for v in views]
    poses={i:pose_to_matrix(p) for i,p in enumerate(baseline)}
    appearances=np.stack([appearance_vector(f) for f in matching])
    descriptors=appearances@appearances.T
    centers=np.array([p[:2,:2]@np.array([256.,256.])+p[:2,2] for p in poses.values()])
    extra=[];audits=[];start=time.perf_counter()
    for k in range(len(frames)):
        choices=sorted((j for j in range(len(frames)) if j!=k and abs(j-k)<=30
                       and tuple(sorted((j,k))) not in forbidden),
                       key=lambda j:float(np.linalg.norm(centers[j]-centers[k])))[:10]
        neighbors=[j for j in choices if _overlap((np.linalg.inv(poses[j])@poses[k])[:2],frames[j].shape,frames[k].shape)>.3][:6]
        if len(neighbors)>=2:
            reference=neighbors[0]
            for kind in ['geometry','intensity']:
                if kind=='geometry':
                    measured,audit=geometry_measurement(k,neighbors,points,poses,reference)
                else:
                    measured,audit,_=intensity_measurement(k,neighbors,matching,features,poses,reference)
                audit.update(frame=k,reference=reference)
                audits.append(audit)
                if measured is not None:
                    matrix,information=measured
                    extra.append(Edge(reference,k,matrix[:2],.35,kind,information))
        candidates=np.argsort(-descriptors[k])
        checked=0
        for j in candidates:
            j=int(j);pair=tuple(sorted((j,k)))
            if j>=k-30 or pair in pairset or pair in forbidden or descriptors[k,j]<.55:
                continue
            checked+=1
            if checked>3:
                break
            match=register(matching[j],matching[k],min_inliers=25,reference_features=features[j],moving_features=features[k])
            audit={'frame':k,'reference':j,'kind':'loop','accepted':False,'reason':'pair_validation'}
            if match.valid and match.inliers>=30 and match.median_error<=2.5:
                correspondence=sift_inliers(matching[j],matching[k],25,reference_features=features[j],moving_features=features[k])
                backward=register(matching[k],matching[j],min_inliers=25,reference_features=features[k],moving_features=features[j])
                if correspondence is not None and min(spread(correspondence[0]),spread(correspondence[1]))>=20 and backward.valid:
                    cycle=homogeneous(match.matrix)@homogeneous(backward.matrix)
                    overlap,corr=_warped_correlation(matching[j],matching[k],match.matrix)
                    accepted=bool(np.linalg.norm(cycle[:2,2])<2 and abs(np.arctan2(cycle[1,0],cycle[0,0]))<np.deg2rad(.5)
                                  and overlap>.3 and np.isfinite(corr) and corr>.55)
                    audit.update(accepted=accepted,reason='accepted' if accepted else 'consistency')
                    if accepted:
                        extra.append(Edge(j,k,match.matrix,.7,'loop'));pairset.add(pair)
            audits.append(audit)
        if (k+1)%100==0:
            print(f'All-frame measurements {k+1}/569, added {len(extra)}',flush=True)
    selections=[('original',()),('geometry',('geometry',)),('intensity',('intensity',)),
                ('loop',('loop',)),('all',('geometry','intensity','loop'))]
    positions={}
    for name,kinds in selections:
        selected=[e for e in extra if e.kind in kinds]
        result=optimize_poses(baseline,edges+selected) if selected else baseline.copy()
        positions[name]=result
        np.save(output/f'{name}_poses.npy',result)
        print(f'Optimized {name}: original {len(edges)} + extra {len(selected)}',flush=True)
    error={name:[] for name,_ in selections};checks=[]
    for i,j in validation:
        corr=sift_inliers(matching[i],matching[j],20,reference_features=features[i],moving_features=features[j])
        if corr is None:
            continue
        src,dst,_=corr;entry={'files':[indices[i],indices[j]]}
        for name,position in positions.items():
            a,b=pose_to_matrix(position[i]),pose_to_matrix(position[j])
            distances=np.linalg.norm(dst@a[:2,:2].T+a[:2,2]-src@b[:2,:2].T-b[:2,2],axis=1)
            value=float(np.median(distances));error[name].append(value);entry[name]=value
        checks.append(entry)
    metrics={name:{'median_px':float(np.median(values)),'p90_px':float(np.percentile(values,90)),
                   'improved_pairs':int(np.sum(np.asarray(values)<np.asarray(error['original'])))} for name,values in error.items()}
    for name in ['original','all']:
        fuse(frames,positions[name],output/name,display_lo=0,display_hi=.06)
    original=np.load(source/'component_001/mosaic_float64.npy')
    recreated=np.load(output/'original/mosaic_float64.npy')
    assert np.array_equal(original,recreated)
    grays=[];offsets=[]
    for name in ['original','all']:
        data=np.load(output/name/'mosaic_float64.npy');weight=np.load(output/name/'weight_map.npy')
        assert np.isfinite(data).all()
        gray=np.rint(255*np.clip(data/.06,0,1)).astype(np.uint8);gray[weight<=0]=0
        cv2.imwrite(str(output/name/'mosaic_gray.png'),gray)
        grays.append(gray);offsets.append(_canvas(positions[name],frames[0].shape)[2])
    labels=['Original: all frames + all original edges','All frames + original edges + extra factors']
    for palette in ['hot','gray']:
        canvas=Image.new('RGB',(1200,920),'white')
        for i,gray in enumerate(grays):
            pixels=cv2.cvtColor(cv2.applyColorMap(gray,cv2.COLORMAP_HOT),cv2.COLOR_BGR2RGB) if palette=='hot' else gray
            im=Image.fromarray(pixels).convert('RGB');im.thumbnail((600,870),Image.Resampling.LANCZOS)
            ImageDraw.Draw(canvas).text((i*600+8,12),labels[i],fill='black')
            canvas.paste(im,(i*600+(600-im.width)//2,40))
        canvas.save(output/f'comparison_{palette}.png')
    a=np.array([[.0336202331845055,.686132032823773,-33.71458966275943],[-.513889681240328,.0250391580511848,644.378393234798],[0,0,1.]])
    flip=np.array([[-1,0,grays[0].shape[1]-1],[0,1,0],[0,0,1.]])
    crop=np.array([[1,0,-329],[0,1,-238],[0,0,1.]])
    canvas=Image.new('RGB',(1208,484),'white')
    for i,(gray,offset) in enumerate(zip(grays,offsets)):
        roi=cv2.warpAffine(gray,(crop@a@flip@offsets[0]@np.linalg.inv(offset))[:2],(302,217))
        im=Image.fromarray(roi).convert('RGB').resize((604,434),Image.Resampling.NEAREST)
        ImageDraw.Draw(canvas).text((i*604+8,12),labels[i],fill='black');canvas.paste(im,(i*604,50))
    canvas.save(output/'vessel_roi.png')
    summary={'source':str(source),'frames':569,'original_edges':len(edges),'all_original_edges_preserved':True,
        'keyframes_used':False,'extra_factors':dict(Counter(e.kind for e in extra)),
        'metrics':metrics,'diagnostic_pair_count':len(checks),'diagnostic_pairs':checks,
        'measurements':audits,'elapsed_seconds':time.perf_counter()-start,
        'notes':'Same cached zero-baseline data, original all-frame poses used as initialization; every original edge retained. Same 38 diagnostic pairs, excluded as direct new map-neighbor and loop pairs, but indirect shared features mean no independent ground truth. No parameter search.'}
    (output/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['diagnostic_pairs','measurements']},indent=2))


if __name__=='__main__':
    main()
