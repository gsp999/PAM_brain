"""Optional sequential keyframe backend inside each frontend-connected group.

Only earlier frame measurements initialize poses. Ordinary frames retain their
reference-keyframe transforms. No baseline optimized pose enters this module.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from .poses import Edge, pose_to_matrix, matrix_to_pose, optimize_poses
from .window import optimize_window
from .factors import (homogeneous, transform, spread, edge_points,
                      geometry_measurement, intensity_measurement)
from pam_mosaic.frontend.registration import register, sift_inliers, _overlap, _warped_correlation
from pam_mosaic.frontend.view import registration_view
from pam_mosaic.frontend.candidates import appearance_vector


@dataclass(frozen=True)
class BackendConfig:
    factors: tuple[str,...] = ('geometry','intensity','loop')
    window_size: int = 12
    keyframe_max_gap: int = 8
    keyframe_translation_fraction: float = .10
    keyframe_angle_deg: float = 1.5
    loop_min_gap: int = 30


def run_keyframe_backend(frames, frontend_edges, config=BackendConfig()):
    if not set(config.factors)<= {'geometry','intensity','loop'}:
        raise ValueError('Unknown backend factor')
    if config.window_size<2 or config.keyframe_max_gap<1:
        raise ValueError('Invalid backend window/keyframe interval')
    n=len(frames)
    views=[registration_view(frame) for frame in frames]
    sift=cv2.SIFT_create(nfeatures=1200)
    features=[sift.detectAndCompute(view,None) for view in views]
    points=[edge_points(view) for view in views]
    appearances=np.stack([appearance_vector(f) for f in frames])
    incoming={i:[] for i in range(n)}
    for edge in frontend_edges:
        a,b=edge.reference,edge.moving
        matrix=homogeneous(edge.matrix)
        if a>b:
            a,b=b,a;matrix=np.linalg.inv(matrix)
        incoming[b].append((a,matrix,edge.weight))
    poses={}
    segments=[]
    all_audits=[]
    keyframes=[];keyposes=[];edges=[];refs={};relatives={};members=[]
    global_updates=0

    def update_frames():
        for f in members:
            poses[f]=pose_to_matrix(keyposes[refs[f]])@relatives[f]

    def finish():
        nonlocal keyposes
        if not members:
            return
        # Final global pass propagates historical corrections to all keyframes.
        keyposes=list(optimize_poses(np.asarray(keyposes),edges))
        update_frames()
        segments.append({'frame_indices':members.copy(),
                         'poses':np.asarray([matrix_to_pose(poses[i]) for i in members]),
                         'keyframes':keyframes.copy(),
                         'keyframe_poses':np.asarray(keyposes).tolist(),
                         'ordinary_frame_references':[keyframes[refs[i]] for i in members],
                         'factors':[{'reference':keyframes[e.reference],'moving':keyframes[e.moving],
                                     'kind':e.kind,'weight':e.weight,'matrix':e.matrix.tolist(),
                                     'sqrt_information':None if e.sqrt_information is None else e.sqrt_information.tolist()}
                                    for e in edges]})

    for k in range(n):
        if k==0:
            match=None
        else:
            available=[e for e in incoming[k] if e[0] in poses and e[0] in members]
            available.sort(key=lambda e: (k-e[0],-e[2]))
            match=available[0] if available else None
            # If the precomputed frontend has no past link, try recent keyframes.
            if match is None:
                for j in reversed(keyframes[-3:]):
                    measured=register(frames[j],frames[k],reference_features=features[j],moving_features=features[k])
                    if measured.valid:
                        match=(j,homogeneous(measured.matrix),min(2.,measured.inliers/30));break
        if match is None:
            if members:
                finish()
                all_audits.append({'frame':k,'kind':'tracking','accepted':False,'reason':'new_submap_no_past_match'})
            keyframes=[k];keyposes=[np.zeros(3)];edges=[];members=[k]
            refs={k:0};relatives={k:np.eye(3)};poses[k]=np.eye(3)
            continue
        parent,relative,tracking_weight=match
        poses[k]=poses[parent]@relative
        last=keyframes[-1]
        relative_to_key=np.linalg.inv(poses[last])@poses[k]
        motion=matrix_to_pose(relative_to_key)
        threshold=config.keyframe_translation_fraction*min(frames[k].shape)
        is_key=(k-last>=config.keyframe_max_gap or np.linalg.norm(motion[:2])>=threshold
                or abs(np.degrees(motion[2]))>=config.keyframe_angle_deg
                or tracking_weight<.5 or k==n-1)
        members.append(k)
        refs[k]=len(keyframes)-1;relatives[k]=relative_to_key
        if not is_key:
            continue
        reference=len(keyframes)-1
        current=len(keyframes)
        keyframes.append(k);keyposes.append(matrix_to_pose(poses[k]))
        refs[k]=current;relatives[k]=np.eye(3)
        # One odometry edge per keyframe interval, no duplicate per-frame edges.
        edges.append(Edge(reference,current,relative_to_key[:2],min(1.,tracking_weight),'odometry'))
        local=[]
        for j in keyframes[:-1]:
            if k-j>config.loop_min_gap:
                continue
            relative_j=np.linalg.inv(poses[j])@poses[k]
            if _overlap(relative_j[:2],frames[j].shape,frames[k].shape)>.25:
                local.append(j)
        local=local[-6:]
        for kind in ('intensity','geometry'):
            if kind not in config.factors:
                continue
            if kind=='intensity':
                measured,audit,_=intensity_measurement(k,local,frames,features,poses,last)
            else:
                measured,audit=geometry_measurement(k,local,points,poses,last)
            audit.update(frame=k,reference=last)
            all_audits.append(audit)
            if measured is not None:
                matrix,information=measured
                edges.append(Edge(reference,current,matrix[:2],.35,kind,information))
        accepted_loop=False
        if 'loop' in config.factors:
            historical=[j for j in keyframes[:-1] if k-j>config.loop_min_gap]
            ranked=sorted(historical,key=lambda j:float(appearances[j]@appearances[k]),reverse=True)[:3]
            spatial=sorted(historical,key=lambda j:float(np.linalg.norm(poses[j][:2,2]-poses[k][:2,2])))[:2]
            for j in dict.fromkeys(ranked+spatial):
                if float(appearances[j]@appearances[k])<.55:
                    continue
                forward=register(frames[j],frames[k],min_inliers=25,reference_features=features[j],moving_features=features[k])
                audit={'frame':k,'reference':j,'kind':'loop','accepted':False,'reason':'pair_validation'}
                if forward.valid and forward.inliers>=30 and forward.median_error<=2.5:
                    correspondence=sift_inliers(frames[j],frames[k],25,reference_features=features[j],moving_features=features[k])
                    if correspondence is None or min(spread(correspondence[0]),spread(correspondence[1]))<20:
                        audit['reason']='poor_feature_distribution'
                        all_audits.append(audit)
                        continue
                    backward=register(frames[k],frames[j],min_inliers=25,reference_features=features[k],moving_features=features[j])
                    if backward.valid:
                        cycle=homogeneous(forward.matrix)@homogeneous(backward.matrix)
                        discrepancy=matrix_to_pose(cycle)
                        overlap,correlation=_warped_correlation(frames[j],frames[k],forward.matrix)
                        accepted=bool(np.linalg.norm(discrepancy[:2])<2 and abs(discrepancy[2])<np.deg2rad(.5)
                                      and overlap>.3 and np.isfinite(correlation) and correlation>.55)
                        audit.update(accepted=accepted,reason='accepted' if accepted else 'bidirectional_or_correlation',
                                     inliers=forward.inliers,median_error=forward.median_error,
                                     correlation=float(correlation) if np.isfinite(correlation) else None)
                        if accepted:
                            edges.append(Edge(keyframes.index(j),current,forward.matrix,.7,'loop'))
                            accepted_loop=True
                all_audits.append(audit)
        if accepted_loop:
            keyposes=list(optimize_poses(np.asarray(keyposes),edges));global_updates+=1
        else:
            keyposes=list(optimize_window(np.asarray(keyposes),edges,config.window_size))
        update_frames()
        if len(keyframes)%10==0:
            print(f'backend frame {k+1}/{n}, keyframes {len(keyframes)}, factors {len(edges)}',flush=True)
    finish()
    audit={'configuration':asdict(config),'initialization':'past frontend measurements only; no offline optimized poses',
           'window_prior':'Schur prior rebuilt from archived history at window linearization; not constant-memory streaming',
           'global_loop_updates':global_updates,'measurements':all_audits}
    return segments,audit
