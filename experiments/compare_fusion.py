"""Compare three fusion methods on cached frames with identical fixed poses.

Run: PYTHONPATH=src python experiments/compare_fusion.py
No registration, background estimation or source data is changed.
"""
from pathlib import Path
import json
import time
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
from pam_mosaic.backend.fusion import _canvas
from pam_mosaic.backend.poses import pose_to_matrix


def pyramid(image, levels):
    result = [image]
    for _ in range(levels):
        result.append(cv2.pyrDown(result[-1]))
    return result


def main():
    cv2.setNumThreads(4)
    root = Path(__file__).resolve().parents[1]
    source = root/'outputs/pam512_background_full_20261008'
    output = root/'outputs/pam512_fusion_comparison_20261010'
    output.mkdir(exist_ok=True)
    report = json.loads((source/'report.json').read_text())
    assert len(report['components']) == 1
    files = sorted((source/'.run_cache').glob('frame_*.npy'))
    poses = np.load(source/'component_001/poses_xytheta.npy')
    assert len(files) == len(poses) == 569
    shape = np.load(files[0]).shape
    width, height, offset = _canvas(poses, shape)
    h, w = shape
    edge = np.minimum(distance_transform_edt(np.pad(np.ones(shape), 1))[1:-1,1:-1],35).astype(np.float32)
    feather = np.maximum(edge,1)/35
    sums = np.zeros((height,width),np.float64)
    weights = np.zeros((height,width),np.float32)
    levels = 5
    templates = pyramid(weights, levels)
    bands = [np.zeros_like(a) for a in templates]
    bandweights = [np.zeros_like(a) for a in templates]
    seam = np.zeros_like(weights)
    seamvalid = np.zeros_like(weights,dtype=bool)
    finder = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    elapsed = {'feather':0., 'multiband':0., 'graphcut_feather':0.}
    lo, hi = report['intensity_range']
    start = time.perf_counter()
    for index, (file, pose) in enumerate(zip(files,poses)):
        frame = np.load(file)
        matrix = (offset@pose_to_matrix(pose))[:2].astype(np.float32)
        warped = cv2.warpAffine(frame,matrix,(width,height),flags=cv2.INTER_LINEAR,borderValue=0)
        weight = cv2.warpAffine(feather,matrix,(width,height),flags=cv2.INTER_LINEAR,borderValue=0)
        tick = time.perf_counter()
        sums += warped*weight
        weights += weight
        elapsed['feather'] += time.perf_counter()-tick
        tick = time.perf_counter()
        # Extend the source at its edge to avoid introducing a black halo in
        # image pyramids. Gaussian weight masks still restrict its contribution.
        extended = cv2.warpAffine(frame,matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        gp = pyramid(extended,levels)
        wp = pyramid(weight,levels)
        for level in range(levels):
            lap = gp[level]-cv2.pyrUp(gp[level+1],dstsize=(gp[level].shape[1],gp[level].shape[0]))
            bands[level] += lap*wp[level]
            bandweights[level] += wp[level]
        bands[-1] += gp[-1]*wp[-1]
        bandweights[-1] += wp[-1]
        elapsed['multiband'] += time.perf_counter()-tick
        tick = time.perf_counter()
        ys,xs = np.nonzero(weight>0)
        y0,y1=max(0,ys.min()-8),min(height,ys.max()+9)
        x0,x1=max(0,xs.min()-8),min(width,xs.max()+9)
        region = np.s_[y0:y1,x0:x1]
        old = seam[region]
        oldvalid = seamvalid[region].copy()
        newvalid = weight[region]>0
        incoming = warped[region]
        if not np.any(oldvalid & newvalid):
            old[newvalid] = incoming[newvalid]
        else:
            # Sequential graph-cut of incoming frame against current mosaic.
            # Costs use a shared display scale; fusion retains signed floats.
            images = [cv2.UMat(np.repeat((255*np.clip((a-lo)/(hi-lo),0,1))[...,None],3,axis=2).astype(np.float32)) for a in [old,incoming]]
            masks = [cv2.UMat(a.astype(np.uint8)*255) for a in [oldvalid,newvalid]]
            finder.find(images,[(0,0),(0,0)],masks)
            choose_new = (masks[1].get()>0)&newvalid
            choose_new |= newvalid & ~oldvalid
            signed = distance_transform_edt(choose_new)-distance_transform_edt(~choose_new)
            alpha = np.clip(.5+signed/10.,0,1).astype(np.float32)
            alpha[~newvalid]=0
            alpha[newvalid & ~oldvalid]=1
            old[:] = old*(1-alpha)+incoming*alpha
        seamvalid[region] |= newvalid
        elapsed['graphcut_feather'] += time.perf_counter()-tick
        if (index+1)%50==0 or index+1==len(files):
            print(f'Fused {index+1}/{len(files)} frames',flush=True)
    valid=weights>0
    feather_result=np.divide(sums,weights,out=np.zeros_like(sums),where=valid)
    normalized=[np.divide(a,b,out=np.zeros_like(a),where=b>1e-8) for a,b in zip(bands,bandweights)]
    multiband=normalized[-1]
    for level in range(levels-1,-1,-1):
        multiband=cv2.pyrUp(multiband,dstsize=(normalized[level].shape[1],normalized[level].shape[0]))+normalized[level]
    multiband[~valid]=0
    assert np.array_equal(seamvalid,valid)
    baseline=np.load(source/'component_001/mosaic_float64.npy')
    baseline_error=float(np.max(np.abs(feather_result-baseline)))
    assert baseline_error<1e-7, baseline_error
    results={'feather':feather_result,'multiband':multiband,'graphcut_feather':seam}
    grayscale=[]
    for name,data in results.items():
        assert np.isfinite(data).all()
        np.save(output/f'{name}.npy',data)
        gray=np.rint(255*np.clip((data-lo)/(hi-lo),0,1)).astype(np.uint8)
        gray[~valid]=0
        cv2.imwrite(str(output/f'{name}_gray.png'),gray)
        cv2.imwrite(str(output/f'{name}_hot.png'),cv2.applyColorMap(gray,cv2.COLORMAP_HOT))
        grayscale.append(gray)
    np.save(output/'coverage.npy',valid)
    labels=['Feather (current)','Laplacian multiband (5 levels)','Graph-cut seam + narrow feather']
    for palette in ['gray','hot']:
        canvas=Image.new('RGB',(1500,790),'white')
        for i,name in enumerate(results):
            im=Image.open(output/f'{name}_{palette}.png').convert('RGB')
            im.thumbnail((500,740),Image.Resampling.LANCZOS)
            ImageDraw.Draw(canvas).text((500*i+10,12),labels[i],fill='black')
            canvas.paste(im,(500*i+(500-im.width)//2,40))
        canvas.save(output/f'comparison_{palette}.png')
    # Same fixed coordinate mapping as the previously inspected weak-vessel ROI.
    a=np.array([[.0336202331845055,.686132032823773,-33.71458966275943],[-.513889681240328,.0250391580511848,644.378393234798],[0,0,1.]])
    flip=np.array([[-1,0,width-1],[0,1,0],[0,0,1.]])
    crop=np.array([[1,0,-329],[0,1,-238],[0,0,1.]])
    matrix=(crop@a@flip)[:2]
    canvas=Image.new('RGB',(1812,484),'white')
    for i,gray in enumerate(grayscale):
        roi=cv2.warpAffine(gray,matrix,(302,217))
        im=Image.fromarray(roi).convert('RGB').resize((604,434),Image.Resampling.NEAREST)
        ImageDraw.Draw(canvas).text((604*i+10,12),labels[i],fill='black')
        canvas.paste(im,(604*i,50))
    canvas.save(output/'vessel_roi.png')
    (output/'report.json').write_text(json.dumps({'source':str(source),'frames':len(files),'size_xy':[width,height],
        'window':[lo,hi],'gamma':1.,'multiband_levels':levels,'seam_method':'sequential graph-cut COST_COLOR_GRAD, frame index order',
        'seam_transition_width_px':10,'elapsed_seconds_excluding_shared_warps':elapsed,
        'total_seconds':time.perf_counter()-start,'feather_baseline_max_abs_error':baseline_error,
        'notes':'Same zero-baseline cached frames and same poses. Single resampling per frame. No registration rerun. Sequential seam selection is order-dependent; no claim of anatomical improvement.'},indent=2))
    print(output,flush=True)


if __name__=='__main__':
    main()
