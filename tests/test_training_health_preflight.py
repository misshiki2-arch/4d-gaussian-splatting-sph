"""Stdlib-only entry/serialization tests. Never launch GPU run/worker/reader."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'tests/cuda/training_health_cuda_acceptance.py'
spec=importlib.util.spec_from_file_location('health_entry',PATH)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class PreflightTests(unittest.TestCase):
    def test_stdlib_candidate_preflight_and_import_denial(self):
        code="""
import sys,importlib.util,tempfile
from pathlib import Path
class Deny:
 def find_spec(self,n,*a):
  if n.split('.')[0] in ('torch','numpy','PIL','scene','train','gaussian_renderer','simple_knn'):
   raise AssertionError('heavy import before execution')
sys.meta_path.insert(0,Deny())
spec=importlib.util.spec_from_file_location('entry',sys.argv[1]);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
v=e.candidate();e.verify_candidate(v)
assert set(e.EXTRA)<=v['files'].keys()
assert {'formal_training_health.py','formal_run_diagnostic.py'}<=v['files'].keys()
with tempfile.TemporaryDirectory() as tmp:
 e.REPORTS=Path(tmp);e.preflight(v,Path(tmp)/'new-output')
 assert not (Path(tmp)/'new-output').exists()
assert 'torch' not in sys.modules
"""
        result=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(PATH)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_changed_candidate_existing_output_and_override_rejected(self):
        expected=entry.candidate()
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'REPORTS',Path(tmp)):
            out=Path(tmp)/'new'
            bad=copy.deepcopy(expected);bad['files'].pop('train.py')
            with self.assertRaises(ValueError):entry.preflight(bad,out)
            out.mkdir()
            with self.assertRaises(ValueError):entry.preflight(expected,out)
            with patch.dict(entry.os.environ,{'STEP90_CUDA_DEBUG_TEST':'1'}):
                with self.assertRaises(ValueError):entry.preflight(expected,Path(tmp)/'other')
            self.assertEqual(list(out.iterdir()),[])


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)

    def children(self):
        marker=self.out/'marker';marker.write_bytes(b'CPU result fixture, not CUDA evidence')
        row=dict(path=str(self.out/'checkpoint'),k=1,marker=entry.file_identity(marker))
        worker=dict(status='passed',cases=[dict(name=name,status='passed',outcome=dict(
            run='returned' if name in entry.CASES[:3] else 'failed_expected')) for name in entry.CASES],checkpoints=[row])
        reader=dict(status='passed',checkpoints=[dict(row,parameters=sorted(entry.GROUPS))])
        entry.write_json(self.out/'worker-result.json',worker)
        entry.write_json(self.out/'reader-result.json',reader)
        for name in ('worker','reader'):entry.write_json(self.out/(name+'-exit.json'),dict(mode=name,exit_code=0))

    def test_normalization_exclusive_all_leaf_readback(self):
        original=dict(nested=[dict(pair=(1,2),flag=True,count=1,real=1.,text='positive_infinity',
                                   reason='identical_channel')])
        before=copy.deepcopy(original)
        path=self.out/'normal.json';entry.write_json(path,original)
        self.assertEqual(original,before)
        self.assertEqual(entry.strict_json(path.read_bytes())['nested'][0]['pair'],[1,2])
        with self.assertRaises(FileExistsError):entry.write_json(path,original)
        parse=entry.strict_json
        for mode in ('missing','number','bool_type','int_type'):
            def altered(*a,**kw):
                value=parse(*a,**kw);row=value['nested'][0]
                if mode=='missing':del row['pair']
                elif mode=='number':row['pair'][1]=3
                elif mode=='bool_type':row['flag']=1
                else:row['count']=1.
                return value
            with self.subTest(mode=mode),patch.object(entry,'strict_json',side_effect=altered):
                with self.assertRaisesRegex(ValueError,'readback'):entry.write_json(self.out/(mode+'.json'),original)
        for i,bad in enumerate((float('nan'),float('inf'),-float('inf'),object(),{1:'key'})):
            with self.assertRaises((ValueError,TypeError)):entry.write_json(self.out/('bad'+str(i)),dict(value=bad))

    def test_complete_receipt_and_reader_parameter_coverage(self):
        self.children();entry.validate_results(self.out)
        entry.finalize_result(self.out,dict(status='passed',stage='completed'))
        self.assertEqual(entry.verify_completion(self.out)['status'],'checks_passed')
        original=entry.strict_json
        for mode in ('missing_parameter','duplicate_parameter','missing_row','wrong_value','exit'):
            def altered(data,*a,**kw):
                value=original(data,*a,**kw)
                if mode=='exit' and value.get('mode')=='reader':value['exit_code']=1
                if 'checkpoints' in value and 'cases' not in value:
                    if mode=='missing_parameter':value['checkpoints'][0]['parameters'].pop()
                    elif mode=='duplicate_parameter':value['checkpoints'][0]['parameters'][0]=value['checkpoints'][0]['parameters'][1]
                    elif mode=='missing_row':value['checkpoints']=[]
                    elif mode=='wrong_value':value['checkpoints'][0]['k']=2
                return value
            with self.subTest(mode=mode),patch.object(entry,'strict_json',side_effect=altered):
                with self.assertRaises(ValueError):entry.verify_completion(self.out)

    def test_result_and_receipt_failure_cannot_complete(self):
        for mode in ('result','receipt','readback'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);write=entry.write_json
                def fail(path,value):
                    if path.name==mode+'.json':raise OSError('injected finalization failure')
                    if mode=='readback' and path.name=='result.json':
                        write(path,value);raise ValueError('injected independent readback failure')
                    return write(path,value)
                with patch.object(entry,'write_json',side_effect=fail):
                    with self.assertRaises((OSError,ValueError)):entry.finalize_result(out,dict(status='passed',stage='completed'))
                self.assertFalse((out/'receipt.json').exists())
                with self.assertRaises((OSError,ValueError)):entry.verify_completion(out)

    def test_child_full_output_exit_and_failure_without_GPU(self):
        popen=entry.subprocess.Popen
        def cpu_child(command,**kw):
            self.assertEqual(command[3],'worker')
            self.assertEqual(command[0],sys.executable)
            code="import sys;print('x'*12000);print('y'*12000,file=sys.stderr);sys.exit(3)"
            return popen([sys.executable,'-B','-c',code],**kw)
        with patch.object(entry.subprocess,'Popen',side_effect=cpu_child):
            with self.assertRaisesRegex(ValueError,'child_failed'):entry.child('worker',self.out,30)
        self.assertEqual((self.out/'worker.stdout.log').read_text(),'x'*12000+'\n')
        self.assertEqual((self.out/'worker.stderr.log').read_text(),'y'*12000+'\n')
        self.assertEqual(entry.strict_json((self.out/'worker-exit.json').read_bytes())['exit_code'],3)


if __name__=='__main__':unittest.main()
