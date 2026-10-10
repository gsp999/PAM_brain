"""Two-stage float Laplacian fusion, fixed original no-background geometry."""
from pathlib import Path
import json
import time
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
from pam_mosaic.backend.fusion import _canvas_varied
from pam_mosaic.backend.poses import pose_to_matrix
from compare_fusion import pyramid


def blend(items, poses, shapes):
    width,height,offset=_canvas_varied(poses,shapes)
    weights=np.zeros((height,width),np.float32)
    sums=np.zeros((height,width),np.float64)
    bands=[np.zeros_like(a) for a in pyramid(weights,5)]
    ws=[np.zeros_like(a) for a in bands]
    for (image,weight),pose in zip(items,poses):
        matrix=(offset@pose_to_matrix(pose))[:2].astype(np.float32)
        warped=cv2.warpAffine(image.astype(np.float32),matrix,(width,height),flags=cv2.INTER_LINEAR,borderValue=0)
        projected=cv2.warpAffine(weight.astype(np.float32),matrix,(width,height),flags=cv2.INTER_LINEAR,borderValue=0)
        sums+=warped*projected
        weights+=projected
        valid=weight>0
        if not valid.all():
            nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
            image=image[tuple(nearest)]
        extended=cv2.warpAffine(image.astype(np.float32),matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        gp=pyramid(extended,5)
        wp=pyramid(projected,5)
        for i in range(6):
            detail=gp[i] if i==5 else gp[i]-cv2.pyrUp(gp[i+1],dstsize=(gp[i].shape[1],gp[i].shape[0]))
            bands[i]+=detail*wp[i]
            ws[i]+=wp[i]
    normalized=[np.divide(a,b,out=np.zeros_like(a),where=b>1e-8) for a,b in zip(bands,ws)]
    result=normalized[-1]
    for i in range(4,-1,-1):
        result=cv2.pyrUp(result,dstsize=(normalized[i].shape[1],normalized[i].shape[0]))+normalized[i]
    result[weights<=0]=0
    feather=np.divide(sums,weights,out=np.zeros_like(sums),where=weights>0)
    return result,weights,feather


def main():
    cv2.setNumThreads(4)
    root=Path(__file__).resolve().parents[1]/'outputs'
    source=root/'stable_rerun_20261006_full'
    old=root/'tile_edge_refinement_20261006'
    out=root/'legacy_multiband_no_background_20261010'
    out.mkdir(exist_ok=True)
    report=json.loads((source/'report.json').read_text())
    assert not report.get('background_correction',False)
    indices=[d['file_index'] for d in report['roundtrip_decisions']]
    mapping={file_index:source/f'.run_cache/frame_{i:04d}.npy' for i,file_index in enumerate(indices)}
    sample=np.load(next(iter(mapping.values())))
    weight=np.maximum(np.minimum(distance_transform_edt(np.pad(np.ones(sample.shape),1))[1:-1,1:-1],35),1).astype(np.float32)/35
    errors=[]
    start=time.perf_counter()
    for number,component in enumerate(report['components'],1):
        name=Path(component['path']).name
        poses=np.load(source/name/'poses_xytheta.npy')
        items=((np.load(mapping[i]),weight) for i in component['file_indices'])
        result,w,feather=blend(items,poses,[sample.shape]*len(poses))
        expected=np.load(source/name/'mosaic_float64.npy')
        error=float(np.max(np.abs(feather-expected)))
        assert error<1e-7,(name,error)
        errors.append(error)
        destination=out/name
        destination.mkdir(exist_ok=True)
        np.save(destination/'mosaic_float64.npy',result.astype(np.float64))
        np.save(destination/'weight_map.npy',w)
        if number%10==0 or number==len(report['components']):
            print(f'Multiband components {number}/{len(report["components"])}',flush=True)
    group=json.loads((old/'report.json').read_text())['groups'][0]
    names=[Path(p).name for p in group['tiles']]
    poses=np.load(old/'mosaic_001/tile_poses_xytheta.npy')
    images=[np.load(out/n/'mosaic_float64.npy') for n in names]
    weights=[np.load(out/n/'weight_map.npy') for n in names]
    result,w,_=blend(zip(images,weights),poses,[a.shape for a in images])
    # Verify second-stage geometry and current feather formula independently.
    original_images=[np.load(source/n/'mosaic_float64.npy') for n in names]
    _,baseline_weights,recreated=blend(zip(original_images,weights),poses,[a.shape for a in original_images])
    baseline=np.load(old/'mosaic_001/mosaic_float64.npy')
    final_error=float(np.max(np.abs(recreated-baseline)))
    assert final_error<1e-7,final_error
    assert np.array_equal(w>0,baseline_weights>0)
    assert np.isfinite(result).all()
    np.save(out/'mosaic_float64.npy',result.astype(np.float64))
    np.save(out/'weight_map.npy',w)
    grays=[]
    for name,data in [('previous_feather',baseline),('multiband',result)]:
        gray=np.rint(255*np.clip((data-.02)/.23,0,1)).astype(np.uint8)
        gray[w<=0]=0
        grays.append(gray)
        cv2.imwrite(str(out/f'{name}_gray.png'),gray)
        cv2.imwrite(str(out/f'{name}_hot.png'),cv2.applyColorMap(gray,cv2.COLORMAP_HOT))
    labels=['Previous feather (background OFF)','Laplacian multiband (background OFF)']
    for palette in ['gray','hot']:
        canvas=Image.new('RGB',(1400,760),'white')
        for i,name in enumerate(['previous_feather','multiband']):
            image=Image.open(out/f'{name}_{palette}.png').convert('RGB')
            image.thumbnail((700,710),Image.Resampling.LANCZOS)
            ImageDraw.Draw(canvas).text((i*700+12,12),labels[i],fill='black')
            canvas.paste(image,(i*700+(700-image.width)//2,40))
        canvas.save(out/f'comparison_{palette}.png')
    # Three fixed-coordinate crops span upper, middle and lower vessel regions.
    crops=[(1400,450,400,350),(850,1200,400,350),(1750,1800,400,350)]
    canvas=Image.new('RGB',(1000,3*480),'white')
    for row,(x,y,cw,ch) in enumerate(crops):
        for i,gray in enumerate(grays):
            image=Image.fromarray(gray[y:y+ch,x:x+cw]).convert('RGB').resize((500,438),Image.Resampling.NEAREST)
            ImageDraw.Draw(canvas).text((i*500+8,row*480+8),f'{labels[i]} / ROI {row+1}',fill='black')
            canvas.paste(image,(i*500,row*480+35))
    canvas.save(out/'vessel_rois.png')
    (out/'report.json').write_text(json.dumps({'source':str(source),'baseline':str(old),'background_correction':False,
        'processed_frames':len(indices),'components':len(report['components']),'selected_tiles':len(names),'final_frames':group['source_files'],
        'size_xy':[result.shape[1],result.shape[0]],'window':[.02,.25],'gamma':1,'levels':5,
        'first_stage_baseline_max_abs_error':max(errors),'final_baseline_max_abs_error':final_error,
        'elapsed_seconds':time.perf_counter()-start,'roi_xywh':crops,
        'notes':'Both fusion stages changed; all frame and tile poses fixed to original no-background run. No registration rerun. Same coverage and window.'},indent=2))
    print(out,flush=True)


if __name__=='__main__':
    main()
