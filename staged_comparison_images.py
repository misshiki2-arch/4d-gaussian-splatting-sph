"""Corresponding A/B/GT fixed views; no quality threshold or test selection."""
from pathlib import Path
from formal_checkpoint_identity import file_identity, require
from staged_comparison_checkpoint import _read_record
import staged_comparison_support as s


def compare(worker, destination, a_worker, a_run, dataset):
    import numpy as np
    from PIL import Image
    destination=Path(destination);destination.mkdir()
    n=worker['completed'];require([v['k'] for v in worker['trace']['fixed']]==[0,n//2,n], 'B_boundaries')
    require([v['label'] for v in a_worker['fixed']]==[0,n//2,n], 'A_boundaries')
    require(a_worker['stage']['name']==worker['stage'] and a_worker['calls']['optimizer_calls']==n-2,'A_stage')
    pictures={};boundaries=[];b_run=Path(worker['expected']['path']).parent
    for aa,bb in zip(a_worker['fixed'],worker['trace']['fixed']):
        k=bb['k'];rows=[];area=destination/str(k);area.mkdir()
        require([v['frame'][1] for v in aa['views']]==[v['frame'][1] for v in bb['views']]==list(s.VIEWS),'comparison_populations')
        for a,b in zip(aa['views'],bb['views']):
            require(a['frame']==b['frame'] and a['raw_time']==b['raw_time'],'frame_time')
            index=a['frame'][1];mask=Path(dataset)/a['frame'][2].replace('images/','masks/',1)
            require(file_identity(mask)==worker['consumed'][str(mask)]==a_worker['consumed'][str(mask)],'mask_identity')
            with Image.open(mask) as im:fg=np.asarray(im.convert('L'))>=128
            images={};metrics={}
            for side,view,root in (('A',a,Path(a_run)),('B',b,b_run)):
                images[side]={}
                for kind in ('gt','render','alpha'):
                    p=root/'fixed'/str(k)/(str(index)+'-'+kind+'.png')
                    require(file_identity(p)==view['images'][kind],'comparison_png')
                    with Image.open(p) as im:images[side][kind]=np.asarray(im.convert('RGB')).copy()
                rgb,gt,alpha=[images[side][v] for v in ('render','gt','alpha')]
                require(rgb.shape==gt.shape==alpha.shape and rgb.shape[:2]==fg.shape,'image_shape')
                error=np.abs(rgb.astype('float64')-gt)/255
                mean=lambda v:float(v.mean()) if v.size else None
                metrics[side]=dict(l1=mean(error),foreground_l1=mean(error[fg]),background_l1=mean(error[~fg]),
                    foreground_alpha=mean(alpha[fg]/255),background_alpha=mean(alpha[~fg]/255))
            require(np.array_equal(images['A']['gt'],images['B']['gt']),'same_GT')
            pixels=np.argwhere(fg)
            crop=None if not len(pixels) else [max(0,int(pixels[:,1].min())-16),max(0,int(pixels[:,0].min())-16),
                min(fg.shape[1],int(pixels[:,1].max())+17),min(fg.shape[0],int(pixels[:,0].max())+17)]
            for kind,box in (('full',None),('crop',crop)):
                if kind=='crop' and box is None:continue
                parts=[Image.fromarray(v) for v in (images['A']['gt'],images['A']['render'],images['B']['render'])]
                if box is not None:parts=[im.crop(box) for im in parts]
                width,height=parts[0].size;canvas=Image.new('RGB',(3*width,height))
                for j,im in enumerate(parts):canvas.paste(im,(j*width,0))
                p=area/(str(index)+'-'+kind+'.png')
                with p.open('xb') as stream:canvas.save(stream,format='PNG')
                pictures[str(p.relative_to(destination))]=file_identity(p)
            rows.append(dict(frame=a['frame'],time=a['raw_time'],crop=crop,png8=metrics,
                float_metrics=dict(A=a['metrics'],B=b['metrics'])))
        boundaries.append(dict(k=k,views=rows,fixed9=[r for r in rows if r['frame'][1] in s.NINE],
            historical5=[r for r in rows if r['frame'][1] in s.HISTORY]))
    value=dict(stage=worker['stage'],status='recorded_requires_image_review',boundaries=boundaries,pictures=pictures,
        regular_train5_test166=dict(A=a_worker['evaluations'],B=worker['trace']['evaluations']),
        optimizer_calls=dict(A=n-2,B=n),test_used_for_selection=False,step18_accepted=False,gate_b_passed=False)
    s.write_json(destination/'comparison.json',value)
    return value
