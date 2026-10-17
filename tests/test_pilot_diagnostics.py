"""Focused metadata/math fixtures. Actual CPU-consumer cases live in pilot suite."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import pilot_contract as c


class DiagnosticTests(unittest.TestCase):
    def test_reviewed_source_delta_only_and_candidate_binds_actual_bytes(self):
        files=c.strict_json((c.BASELINE/'inputs.json').read_bytes(),64*1024**2)['files']
        c.verify_comparison_source(files)
        source=c.strict_json(c.SOURCE_CHANGE.read_bytes());identity=c.file_identity
        for name in ('formal_inputs.py','tests/test_formal_connection.py','train.py','scene/gaussian_model.py'):
            def changed(path):return dict(size=-1,sha256='fixture') if path==c.ROOT/name else identity(path)
            with self.subTest(name=name),patch.object(c,'file_identity',side_effect=changed),self.assertRaises(ValueError):
                c.verify_comparison_source(files)
        for mode in ('extra','before','after','diff'):
            bad=copy.deepcopy(source)
            if mode=='extra':bad['files']['train.py']=bad['files']['formal_inputs.py']
            else:bad['files']['formal_inputs.py'][mode]={}
            with self.subTest(mode=mode),patch.object(c,'strict_json',return_value=bad),self.assertRaises(ValueError):
                c.verify_comparison_source(files)
        actual=c.candidate()
        self.assertEqual(actual['files']['formal_inputs.py'],identity(c.ROOT/'formal_inputs.py'))
        self.assertEqual(actual['comparisons']['records'][str(c.SOURCE_CHANGE)],identity(c.SOURCE_CHANGE))

    def test_fixed_three_time_gates_rotated_variance_and_coordinate_mapping(self):
        import importlib.util
        import math
        import torch
        spec=importlib.util.spec_from_file_location('pilot_compare',c.COMPARE)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        # Four independent centers. qL quarter-turn swaps time and a spatial
        # axis: temporal scale is 4, but rotated sigma is 1 for every point.
        capture=[None]*19
        capture[1]=torch.zeros((4,3),dtype=torch.float64)
        capture[4]=torch.zeros((4,3),dtype=torch.float64)
        capture[14]=torch.full((4,1),math.log(4.),dtype=torch.float64)
        capture[5]=torch.tensor([[0.,1.,0.,0.]]*4,dtype=torch.float64)
        capture[15]=torch.tensor([[1.,0.,0.,0.]]*4,dtype=torch.float64)
        capture[13]=torch.tensor([[0.],[2.],[6.],[10.]],dtype=torch.float64)
        capture[6]=torch.zeros((4,1),dtype=torch.float64)
        loaded=dict(state=dict(capture=capture),metadata=dict(k=3000),
                    config=SimpleNamespace(time_derivation=SimpleNamespace(divisor=2.)))
        representatives=[['train',i,'fixture'+str(i),stamp] for i,stamp in enumerate((0.,4.,12.))]
        result=helper.checkpoint_widths(loaded,representatives)
        self.assertEqual([r['time'] for r in result['temporal_gate']],[0.,2.,6.])
        self.assertEqual([r['temporal_gate_count'] for r in result['temporal_gate']],[2,2,1])
        for row,stamp in zip(result['temporal_gate'],(0.,2.,6.)):
            expected=sum(math.exp(-.5*(center-stamp)**2) for center in (0.,2.,6.,10.))
            self.assertAlmostEqual(row['sum_marginal'],expected,places=14)
            self.assertAlmostEqual(row['sum_opacity_marginal'],expected*.5,places=14)
        self.assertEqual(result['widths']['sigma_tt']['mean'],1.)
        with self.assertRaises(ValueError):helper.checkpoint_widths(loaded,representatives[:2])
        bad=copy.deepcopy(representatives);bad[0][0]='test'
        with self.assertRaises(ValueError):helper.checkpoint_widths(loaded,bad)

    def test_exact_typed_two_leaf_binding_and_no_third_change(self):
        raw=c.strict_json(c.CONFIG.read_bytes());value=c.comparison_binding(raw)
        self.assertEqual([d['path'] for d in value['changes']],['/initialization/time_variance_denominator','/output/directory'])
        self.assertEqual(value['diagnostics'],dict(initial=0,reset=3000,prune=3100))
        for mode in ('third','type','missing','old_coefficient','output'):
            bad=copy.deepcopy(raw)
            if mode=='third':bad['initialization']['seed']+=1
            elif mode=='type':bad['initialization']['time_variance_denominator']=33
            elif mode=='missing':bad['reporting'].pop('test_updates')
            elif mode=='old_coefficient':bad['initialization']['time_variance_denominator']=5.0
            else:bad['output']['directory']+='-other'
            with self.subTest(mode=mode),self.assertRaises(ValueError):c.comparison_binding(bad)

    def test_prune_all_memberships_and_size_disabled_and_mask_identity(self):
        import torch
        import pilot_diagnostics as d
        b=torch.arange(8)
        m=SimpleNamespace(get_opacity=torch.where((b&1)!=0,.001,.5)[:,None],
                          max_radii2D=torch.where((b&2)!=0,21.,20.),
                          get_scaling=torch.where((b&4)!=0,.2,.01)[:,None].expand(8,3))
        mask=b!=0;before=mask.clone()
        result=d.prune_summary(m,mask,.005,1.,20)
        self.assertEqual(result['bins'],[1]*8);self.assertEqual(result['union'],7)
        self.assertEqual(result['inclusive'],dict(opacity=4,screen=4,world=4))
        self.assertEqual(result['overlaps'],dict(opacity_screen=2,opacity_world=2,screen_world=2,triple=1))
        self.assertTrue(torch.equal(before,mask))
        off=d.prune_summary(m,(b&1)!=0,.005,1.,None);self.assertEqual(off['bins'],[4,4,0,0,0,0,0,0])
        empty=d.prune_summary(m,torch.zeros(8,dtype=torch.bool),0.,1.,None);self.assertEqual(empty['kept'],8)
        for bad in (~mask,mask[:,None],mask.float()):
            with self.assertRaises(ValueError):d.prune_summary(m,bad,.005,1.,20)

    def fixture(self,out):
        import numpy as np
        import pilot_diagnostics as d
        rep=['train',0,'images/f.png',0.]
        images={}
        directory=out/'diagnostics/reset-3000-before';directory.mkdir(parents=True)
        for name in ('gt','render','alpha','depth'):
            p=directory/('0-'+name+'.png');p.write_bytes(b'encoded file fixture');images[name]=c.file_identity(p)
        view=dict(frame=rep[:3],raw_time=0.,images=images,metrics=dict(l1=[0.1],ssim=[.9],psnr=[10.,10.,10.]))
        widths={name:d.summary(np.full(16,0. if name=='log_scale' else 1.)) for name in ('log_scale','scale','variance','sigma_tt','rotated_sigma')}
        diagnostics=dict(initial=dict(k=0,stage='before_first_fetch',rows=16,widths=widths),
            reset=dict(attempted_update=3000,completed=2999,stage='after_population_before_reset',after_fixed=3000,views=[view]),
            prune=dict(attempted_update=3100,stage='final_mask_after_growth',before=16,rows=16,after=2,net=-14,prune_only=True,split_parent_calls=0,
                bins=[2]*8,union=14,kept=2,inclusive=dict(opacity=8,screen=8,world=8),
                overlaps=dict(opacity_screen=4,opacity_world=4,screen_world=4,triple=2),thresholds=dict(opacity=.005,extent=1.,screen=20)))
        trace=dict(fixed=[dict(k=3000,views=[view])],population=[dict(k=3100,before=16,after=2,prune_only=True,size_threshold=20)])
        return diagnostics,dict(reset=[3000],representatives=[rep]),trace

    def test_diagnostics_saved_readback_and_missing_corrupt_invalid_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);d,s,t=self.fixture(out)
            c.write_json(out/'diagnostics.json',d);loaded=c.strict_json((out/'diagnostics.json').read_bytes())
            self.assertEqual(c.validate_diagnostics(loaded,s,t,out),d)
            for mode in ('missing','initial','finite','reset_stage','time','image','after','rows','bins','overlap','parent','threshold'):
                bad=copy.deepcopy(d)
                if mode=='missing':bad.pop('initial')
                elif mode=='initial':bad['initial']['widths']['variance']['mean']=6.6
                elif mode=='finite':bad['initial']['widths']['scale']['mean']=float('nan')
                elif mode=='reset_stage':bad['reset']['completed']=3000
                elif mode=='time':bad['reset']['views'][0]['raw_time']=.1
                elif mode=='image':bad['reset']['views'][0]['images']['render']={}
                elif mode=='after':bad['reset']['after_fixed']=None
                elif mode=='rows':bad['prune']['rows']+=1
                elif mode=='bins':bad['prune']['bins'][0]=True
                elif mode=='overlap':bad['prune']['overlaps']['triple']+=1
                elif mode=='parent':bad['prune']['split_parent_calls']=1
                else:bad['prune']['thresholds']['opacity']=0.
                with self.subTest(mode=mode),self.assertRaises((ValueError,KeyError)):c.validate_diagnostics(bad,s,t,out)
            (out/'diagnostics/reset-3000-before/0-render.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'reset_image_identity'):c.validate_diagnostics(d,s,t,out)

    def test_receipt_binds_diagnostic_worker_bytes_not_optional_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            for n in ('inputs.json','worker-result.json','reader-result.json'):c.write_json(out/n,dict(pilot_diagnostics={}))
            for mode in ('worker','reader'):
                logs={}
                for stream in ('stdout','stderr'):
                    p=out/(mode+'.'+stream+'.log');p.write_bytes(b'');logs[stream]=c.file_identity(p)
                c.write_json(out/(mode+'-exit.json'),dict(mode=mode,exit_code=0,timed_out=False,logs=logs))
            with patch.object(c,'validate_results'):
                c.finalize(out,dict(status='passed'));c.verify_completion(out)
                (out/'worker-result.json').write_bytes(b'{}')
                with self.assertRaisesRegex(ValueError,'result_evidence_identity'):c.verify_completion(out)


if __name__=='__main__':
    if sys.executable!=c.PYTHON:raise SystemExit('4dgs310 required')
    unittest.main(failfast=True)
