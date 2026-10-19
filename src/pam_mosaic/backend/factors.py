"""Image-derived 2D local-map and multi-view measurements (not pose priors)."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from ..frontend.registration import sift_inliers, _huber_residual
from .poses import matrix_to_pose, pose_to_matrix


def transform(points, matrix):
    return points@matrix[:2,:2].T+matrix[:2,2]


def homogeneous(matrix):
    return np.vstack((matrix,[0.,0.,1.])) if matrix.shape==(2,3) else matrix


def edge_points(view, limit=700):
    ys,xs=np.nonzero(cv2.Canny(view,35,90))
    points=np.column_stack((xs,ys)).astype(float)
    if len(points)>limit:
        points=points[np.linspace(0,len(points)-1,limit,dtype=int)]
    return points


def spread(points):
    return float(np.sqrt(max(0,np.linalg.eigvalsh(np.cov(points.T))[0]))) if len(points)>2 else 0.


def solve_measurement(initial, residual, kind, minimum_count):
    """Scale angle to pixel-like units before conditioning and LM."""
    p=matrix_to_pose(initial); p[2]*=100
    def matrix(x):
        physical=x.copy();physical[2]/=100
        return pose_to_matrix(physical)
    function=lambda x: residual(matrix(x))
    if len(function(p))<minimum_count:
        return None,{'kind':kind,'accepted':False,'reason':'insufficient_support'}
    fit=least_squares(lambda x:_huber_residual(function(x),2.),p,method='lm',max_nfev=60)
    result=matrix(fit.x)
    before=function(p);after=function(fit.x)
    # Numeric Jacobian of unrobustified residual for observability, not damping.
    jac=np.column_stack([(function(fit.x+np.eye(3)[i]*1e-3)-function(fit.x-np.eye(3)[i]*1e-3))/2e-3 for i in range(3)])
    h=jac.T@jac/max(1,len(after))
    eig=np.linalg.eigvalsh(h)
    ratio=float(eig[0]/max(eig[-1],1e-12))
    shift=float(np.linalg.norm(fit.x[:2]-p[:2]))
    angle=float(abs(fit.x[2]-p[2])/100*180/np.pi)
    accepted=bool(np.isfinite(result).all() and ratio>1e-5 and shift<=10 and angle<=1.5
                  and np.mean(_huber_residual(after,2.)**2)<=np.mean(_huber_residual(before,2.)**2)+1e-8)
    sigma=max(.75,float(1.4826*np.median(np.abs(after-np.median(after)))))
    # Normalize sample count; conservative information cap for correlated data.
    covariance=np.linalg.pinv(h+np.eye(3)*1e-9)*sigma**2
    transport=np.eye(3);transport[:2,:2]=result[:2,:2].T
    covariance=transport@covariance@transport.T
    confidence=np.clip(1/np.sqrt(np.maximum(np.diag(covariance),1e-6)),.05,1.)
    audit={'kind':kind,'accepted':accepted,'reason':'accepted' if accepted else 'degenerate_or_excessive_update',
           'support':len(after),'median_before':float(np.median(np.abs(before))),
           'median_after':float(np.median(np.abs(after))),'conditioning_ratio':ratio,
           'shift_px':shift,'angle_deg':angle,'sqrt_information_diagonal':confidence.tolist()}
    return (result,np.diag(confidence)) if accepted else None,audit


def intensity_measurement(current, neighbors, frames, features, poses, reference):
    source=[];targets=[];contributors=[];heldout=[]
    for j in neighbors:
        match=sift_inliers(frames[j],frames[current],reference_features=features[j],moving_features=features[current])
        if match is None:
            continue
        src,dst,_=match
        if spread(src)<15:
            continue
        target=transform(dst,poses[j])
        # Spatial hash keeps the same current feature out of train across all views.
        keys=np.rint(src*4).astype(int)
        validation=(keys[:,0]*73856093+keys[:,1]*19349663)%5==0
        heldout.append((j,src[validation],dst[validation]))
        source.extend(src[~validation]);targets.extend(target[~validation]);contributors.append(j)
    if len(contributors)<2 or len(source)<24:
        return None,{'kind':'intensity','accepted':False,'reason':'need_two_views_and_24_points'},heldout
    # Collapse repeated observations of the same feature instead of multiplying weight.
    source=np.asarray(source);targets=np.asarray(targets)
    _,ids=np.unique(np.rint(source*4).astype(int),axis=0,return_inverse=True)
    count=np.bincount(ids)
    src=np.column_stack([np.bincount(ids,weights=source[:,d])/count for d in range(2)])
    dst=np.column_stack([np.bincount(ids,weights=targets[:,d])/count for d in range(2)])
    initial=poses[current]
    good=np.linalg.norm(transform(src,initial)-dst,axis=1)<8
    src,dst=src[good],dst[good]
    measurement,audit=solve_measurement(initial,lambda m:(transform(src,m)-dst).ravel(),'intensity',48)
    audit['map_frames']=contributors
    measurement=world_to_relative(measurement,audit,poses[reference])
    return measurement,audit,heldout


def world_to_relative(measurement, audit, reference_pose):
    audit['registration_frame']='component_world'
    audit['information_frame']='moving_frame_log_error'
    if measurement is None:
        return None
    matrix,information=measurement
    audit['world_pose_xytheta']=matrix_to_pose(matrix).tolist()
    return np.linalg.inv(reference_pose)@matrix,information


def geometry_measurement(current, neighbors, points, poses, reference):
    available=[transform(points[j],poses[j]) for j in neighbors if len(points[j])>20]
    if len(available)<2 or len(points[current])<50:
        return None,{'kind':'geometry','accepted':False,'reason':'insufficient_map'}
    local=np.concatenate(available)
    _,unique=np.unique(np.rint(local/2).astype(int),axis=0,return_index=True)
    local=local[unique]
    tree=cKDTree(local)
    initial=poses[current]
    query=transform(points[current],initial)
    distances,indices=tree.query(query,k=5)
    patches=local[indices];centers=patches.mean(axis=1)
    centered=patches-centers[:,None,:]
    covariance=np.einsum('nki,nkj->nij',centered,centered)/5
    values,vectors=np.linalg.eigh(covariance)
    valid=(distances[:,0]<4)&(distances[:,-1]<12)&(values[:,0]<.2*np.maximum(values[:,1],1e-9))&(values[:,1]>1)
    src=points[current][valid];dst=centers[valid];normal=vectors[valid,:,0]
    # Fixed point-to-line associations within one measurement; frozen for this offline factor-building pass.
    residual=lambda m:np.sum((transform(src,m)-dst)*normal,axis=1)
    measurement,audit=solve_measurement(initial,residual,'geometry',80)
    audit['map_frames']=neighbors
    if measurement is not None:
        m,_=measurement
        nearest=tree.query(transform(src,m),k=1)[0]
        if float(np.median(nearest))>3:
            measurement=None;audit.update(accepted=False,reason='poor_map_overlap')
    measurement=world_to_relative(measurement,audit,poses[reference])
    return measurement,audit
