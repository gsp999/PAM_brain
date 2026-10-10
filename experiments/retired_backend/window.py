"""Fixed-lag LM with a Schur prior; archive the full graph for loop corrections.

The prior is rebuilt from archived factors at each new window linearization.
This offline implementation bounds nonlinear active states, not total memory.
"""
import numpy as np
from scipy.optimize import least_squares
from .poses import _lm_residual_jacobian
from pam_mosaic.frontend.registration import _huber_residual


def schur_prior(hessian, gradient, eliminated):
    """Eliminate the first `eliminated` variables using a PSD pseudoinverse."""
    if eliminated == 0:
        return hessian.copy(), gradient.copy()
    a=hessian[:eliminated,:eliminated]
    b=hessian[:eliminated,eliminated:]
    inverse=np.linalg.pinv(a, rcond=1e-10, hermitian=True)
    h=hessian[eliminated:,eliminated:]-b.T@inverse@b
    g=gradient[eliminated:]-b.T@inverse@gradient[:eliminated]
    return (h+h.T)/2,g


def optimize_window(start, edges, window_size=12):
    n=len(start)
    if n<2:
        return start.copy()
    first=max(1,n-window_size)
    local=[e for e in edges if e.reference>=first and e.moving>=first]
    history=[e for e in edges if e.reference<first or e.moving<first]
    r,j=_lm_residual_jacobian(start,history)
    robust=np.minimum(1.,3/np.maximum(np.abs(r),1e-12))
    j=j.toarray()
    h=j.T@(robust[:,None]*j)
    g=j.T@(robust*r)
    prior_h,prior_g=schur_prior(h,g,3*(first-1))
    values,vectors=np.linalg.eigh(prior_h)
    keep=values>max(1e-10,float(values.max(initial=0))*1e-10)
    root=np.sqrt(values[keep])[:,None]*vectors[:,keep].T
    bias=(vectors[:,keep].T@prior_g)/np.sqrt(values[keep])
    origin=start[first:].ravel().copy()
    def residual(x):
        p=start.copy(); p[first:]=x.reshape(-1,3)
        rr,_=_lm_residual_jacobian(p,local)
        delta=x-origin
        delta[2::3]=(delta[2::3]+np.pi)%(2*np.pi)-np.pi
        return np.concatenate((_huber_residual(rr,3.),root@delta+bias))
    def jacobian(x):
        p=start.copy();p[first:]=x.reshape(-1,3)
        rr,jj=_lm_residual_jacobian(p,local)
        magnitude=np.abs(rr)
        derivative=np.ones_like(rr)
        outside=magnitude>3
        derivative[outside]=3/np.sqrt(6*magnitude[outside]-9)
        return np.vstack((derivative[:,None]*jj[:,3*(first-1):].toarray(),root))
    # A rank-deficient graph is retained unchanged rather than inventing a prior.
    if len(residual(origin))<len(origin):
        return start.copy()
    fit=least_squares(residual,origin,jac=jacobian,method='lm',max_nfev=35)
    result=start.copy()
    if np.isfinite(fit.x).all() and np.sum(residual(fit.x)**2)<=np.sum(residual(origin)**2):
        result[first:]=fit.x.reshape(-1,3)
    return result
