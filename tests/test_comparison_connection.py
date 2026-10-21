"""Actual CPU consumers and Adam; only GPU/native and small fixture domains isolated."""
import dataclasses
import copy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from formal_training_fixture import CPUCase
import comparison_profile as profile
import comparison_inputs as inputs


class ComparisonConnectionTests(CPUCase):
    def provenance_fixture(self):
        """Immutable synthetic origin independent of candidate mutations.

        Only native/device boundaries are simulated; record/file/source/module
        and M/T validators are real. Never rewrite the saved production record.
        """
        import staged_comparison_checkpoint as cp
        from formal_checkpoint_identity import (ROOT, KNN_UNITS, RASTER_UNITS,
                                                canonical, file_identity, byte_identity)
        if not hasattr(self, '_provenance'):
            e = self.f.checkpoint_evidence().value()
            old = copy.deepcopy(e)
            old['source'] = {k:v for k,v in old['source'].items()
                             if not k.startswith(('comparison', 'staged_comparison', 'pointops2/'))}
            old['source']['train.py'] = byte_identity(b'older Python connection')
            # Independent historical source and build directories, not ROOT.
            source = self.f.base/'historical-source'; source.mkdir()
            build = self.f.base/'historical-build'; build.mkdir()
            for name in old['source']:
                if name.startswith(('simple-knn/', 'diff-gaussian-rasterization/', 'gaussian_renderer/')):
                    target = source/name; target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((ROOT/name).read_bytes())
            old['repository'], old['build_root'] = str(source), str(build)
            for kind, units in (('knn', KNN_UNITS), ('raster', RASTER_UNITS)):
                row = old[kind]
                def relocate(path):
                    target = build/Path(path).relative_to(e['build_root'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(Path(path).read_bytes())
                    return str(target)
                for field in ('binary', 'recipe'):
                    row[field]['path'] = relocate(row[field]['path'])
                row['objects'] = {relocate(p):v for p,v in row['objects'].items()}
                recipe = Path(row['recipe']['path'])
                recipe.write_text('\n'.join('build '+str(i)+'.o: cuda_compile '+str(source/u)
                                            for i,u in enumerate(units)))
                row['recipe']['identity'] = file_identity(recipe)
            sys.modules['simple_knn._C'].__file__ = old['knn']['binary']['path']
            sys.modules['gaussian_renderer.diff_gaussian_rasterization']._C.__file__ = old['raster']['binary']['path']
            record = self.f.base/'native-record.json'
            record.write_bytes(canonical(dict(native=old)))
            anchor = self.f.base/'native-anchor.json'
            anchor.write_bytes(canonical(dict(path=str(record), identity=file_identity(record))))
            self.f.gpu.enter_context(patch.object(cp, 'NATIVE_ANCHOR', anchor))
            self._provenance = dict(native_origin=old, connection_source=e['source'],
                                    environment=e['environment'])
            self.origin_record, self.origin_anchor = record, anchor
        return copy.deepcopy(self._provenance)

    def anchor_fixture(self, value):
        """Install an independently recorded CPU origin, never production data."""
        from formal_checkpoint_identity import canonical, file_identity
        self.origin_record.write_bytes(canonical(dict(native=value['native_origin'])))
        self.origin_anchor.write_bytes(canonical(dict(path=str(self.origin_record),
                                                      identity=file_identity(self.origin_record))))

    def initial(self):
        t = self.torch
        # Independent physical16 fixture with distinguishable row/channel order.
        values = dict(xyz=t.arange(6, dtype=t.float32).reshape(2, 3)/10,
            features_dc=t.arange(6, dtype=t.float32).reshape(2, 3, 1)/100,
            features_rest=t.arange(90, dtype=t.float32).reshape(2, 3, 15)/1000,
            scaling=t.full((2, 3), -2.), rotation=t.tensor([[1., .1, .2, .3], [1., .3, .2, .1]]),
            opacity=t.full((2, 1), -1.), t=t.tensor([[.125], [32.5]]),
            scaling_t=t.tensor([[-3.], [-2.5]]), rotation_r=t.tensor([[1., 0, 0, 0], [1., .1, 0, 0]]))
        return values

    def bundle(self, stage='stabilization'):
        from staged_comparison_state import Initializer
        f = self.f
        for split in ('train', 'test'):
            name = 'transforms_'+split+'.json'
            raw = json.loads((f.source/name).read_bytes())
            for i, frame in enumerate(raw['frames']):
                frame['time'] = (0., 16.5, 33.)[i]
                f.ref['frames'][split][i][1] = frame['time']
            f.store_input(name, json.dumps(raw).encode())
        ply = f.PlyData.read(io.BytesIO((f.source/'points3d.ply').read_bytes()))
        ply['vertex'].data['time'] = [0., 1., 16.5, 33.]
        stream = io.BytesIO(); ply.write(stream); f.store_input('points3d.ply', stream.getvalue())
        with patch.object(inputs, 'OUTPUT_OWNER', f.base):
            return inputs.acquire(stage, str(f.output), Initializer.from_initial(self.initial()))

    def test_closed_four_stage_values_and_formal_separation(self):
        from formal_config import FormalConfig
        from formal_checkpoint_state import validate_state
        from formal_views import ConsumerView
        expected = [(12000,.00008,.0000008,.0001,.0005,.01,.02,1000000000,0,100,1.,0.,-1),
                    (30000,.00008,.0000008,.0002,.0005,.01,.02,14000,26000,200,.0006,.002,-1),
                    (12000,.00005,.0000005,.0003,.0003,.015,.01,200,3000,400,.0006,.05,2500000),
                    (12000,.00002,.0000002,.00015,.00015,.01,.01,2500,9000,100,.00022,.02,3200000)]
        for stage, row in zip(profile.STAGES, expected):
            c = profile.stage_config(stage, str(self.f.output)); profile.validate_config(c)
            o = ConsumerView(c, 'optimization')
            self.assertEqual((o.iterations,o.position_lr_init,o.position_lr_final,o.scaling_lr,
                o.rotation_lr,o.opacity_lr,o.lambda_opa_mask,o.densify_from_iter,o.densify_until_iter,
                o.densification_interval,o.densify_grad_threshold,o.thresh_opa_prune,o.densify_until_num_points),row)
            self.assertEqual(o.position_lr_max_steps,12000)
            self.assertEqual(c.maximum_sh.slot_count,16)
            self.assertFalse(isinstance(c,FormalConfig))
            with self.assertRaisesRegex(ValueError,'formal_state_domain'): validate_state({}, {}, c)
            with self.assertRaisesRegex(ValueError,'comparison_profile'):
                profile.validate_config(dataclasses.replace(c,maximum_sh=c.maximum_sh._replace(slot_count=48)))
        with self.assertRaises(ValueError): profile.stage_config('arbitrary', str(self.f.output))
        c = self.f.verify().config
        with self.assertRaises(ValueError): profile.validate_config(c)

    def test_real_scene_inherited_state_fresh_adam_time_and_camera(self):
        formal_bundle = self.f.verify()
        bundle = self.bundle(); initial = self.initial()
        # A PLY/KNN model must never be constructed then thrown away.
        with patch.object(self.train.GaussianModel, 'create_from_pcd', side_effect=AssertionError('discarded model')):
            prepared = self.train.prepare_comparison_runtime(bundle)
        m = prepared.gaussians
        self.assertEqual(tuple(m._features_rest.shape),(2,15,3))
        for key, attr in [('xyz','_xyz'),('opacity','_opacity'),('scaling','_scaling'),('rotation','_rotation'),
                          ('t','_t'),('scaling_t','_scaling_t'),('rotation_r','_rotation_r')]:
            self.assertTrue(self.torch.equal(getattr(m,attr),initial[key]))
        self.assertTrue(self.torch.equal(m._features_dc,initial['features_dc'].transpose(1,2)))
        self.assertTrue(self.torch.equal(m._features_rest,initial['features_rest'].transpose(1,2)))
        self.assertEqual(len(m.optimizer.state),0)
        for name in ('max_radii2D','xyz_gradient_accum','t_gradient_accum','denom'):
            self.assertFalse(getattr(m,name).any())
        self.assertTrue(all(g['params'][0].grad is None for g in m.optimizer.param_groups))
        self.assertIs(m._formal_time,bundle.config.time_derivation)
        self.assertEqual(sorted(cam.formal_frame.effective_time for cam in prepared.scene.train_cameras[1.0]),[0.,16.5,33.])
        self.assertFalse(self.f.output.exists())
        with self.assertRaisesRegex(ValueError,'formal_input_domain'):
            self.train.prepare_formal_runtime(bundle)
        with self.assertRaisesRegex(ValueError,'formal_input_domain'):
            self.train.continue_formal_training(prepared,None)
        with self.assertRaisesRegex(ValueError,'comparison_input_domain'):
            self.train.prepare_comparison_runtime(formal_bundle)

    def test_initializer_rejects_invalid_layout_value_and_parent(self):
        from staged_comparison_state import Initializer
        for mutation in ('extra','shape','nan','quaternion'):
            initial=self.initial()
            if mutation=='extra': initial['other']=0
            if mutation=='shape': initial['features_rest']=initial['features_rest'][:,:,:14]
            if mutation=='nan': initial['t'][0,0]=float('nan')
            if mutation=='quaternion': initial['rotation'][0]=0
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): Initializer.from_initial(initial)
        init=Initializer.from_initial(self.initial())
        model=self.train.GaussianModel(3,gaussian_dim=4,rot_4d=True)
        with self.assertRaisesRegex(ValueError,'comparison_initial_stage'):
            init.initialize(model,self.f.np.float32(1.3),profile.stage_config('growth',str(self.f.output)))

    def test_population_eligibility_uses_pre_topology_count(self):
        c=profile.stage_config('e2_like',str(self.f.output))
        self.assertTrue(profile.population_active(c,400,2499999))
        self.assertFalse(profile.population_active(c,400,2500000))
        self.assertFalse(profile.population_active(c,3000,100))
        self.assertTrue(profile.population_active(profile.stage_config('growth',str(self.f.output)),15000,9999999))
        self.assertFalse(profile.population_active(profile.stage_config('stabilization',str(self.f.output)),1,2))

    def test_shared_loop_exact_final_update_and_full_population_stop(self):
        # Small fixture only: the public profile never accepts custom N/schedule.
        row=list(profile._VALUES[0]);row[0]=3;row[7:10]=[0,4,1];row[12]=1
        with patch.object(profile,'_VALUES',(tuple(row),)+profile._VALUES[1:]):
            p=self.train.prepare_comparison_runtime(self.bundle())
            # The fixture's dispatch is redirected, not the consumer/update body.
            with patch.object(self.train,'continue_formal_training',self.train.continue_comparison_training):
                trace=self.exercise(p)
            self.assertEqual(trace['completed'],3)
            self.assertEqual(trace['labels'],[1,2,3])
            self.assertEqual(len(trace['fetch']),3)
            self.assertEqual(len(trace['steps']),3)
            self.assertEqual(trace['mutations'],[])
            self.assertFalse(p.gaussians.max_radii2D.any())
            self.assertFalse(p.gaussians.denom.any())
            for before, _ in trace['steps']:
                self.assertTrue(all(grad is not None for _, _, grad in before))
            self.assertTrue(any(not self.torch.equal(trace['initial'][g['name']],g['params'][0])
                                for g in p.gaussians.optimizer.param_groups))

    def test_physical16_snapshot_and_weights_only_file_roundtrip(self):
        import staged_comparison_state as state
        from test_formal_checkpoint import independent, assert_expected
        p=self.train.prepare_comparison_runtime(self.bundle());m=p.gaussians;c=p.inputs.config
        m.env_map=None
        k=c.optimization.total_updates
        m.update_learning_rate(k)
        sum(g['params'][0].sum() for g in m.optimizer.param_groups).backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        expected=independent(m)
        payload,desc=state.snapshot(m,c,k)
        meta=dict(k=k,rows=len(m._xyz),optimizer=desc)
        path=self.f.base/'comparison.pt'
        with path.open('xb') as stream:self.torch.save(payload,stream)
        with path.open('rb') as stream:loaded=state.load_state(stream,meta,c)
        result=dict(config=c,state=loaded,metadata=meta)
        assert_expected(self,expected,result)
        model=state.restore_diagnostic(result,'cpu')
        self.assertIsNone(model.optimizer)
        self.assertTrue(self.torch.equal(model._features_rest,m._features_rest))
        self.assertNotEqual(model._features_rest.data_ptr(),m._features_rest.data_ptr())

    def test_comparison_exclusive_writer_reader_parent_and_domain(self):
        import staged_comparison_checkpoint as cp
        import staged_comparison_state as state
        import formal_checkpoint
        from formal_checkpoint_identity import file_identity,canonical
        from staged_comparison_runtime import ComparisonClaim,begin_output
        from test_formal_checkpoint import independent,assert_expected
        bundle=self.bundle();p=self.train.prepare_comparison_runtime(bundle);m=p.gaussians
        self.assertTrue(bundle.initializer.consumed)
        self.assertEqual(bundle.initializer.parameters,())
        m.env_map=None
        initial=self.f.base/'initial.pt'
        with initial.open('xb') as stream:self.torch.save(self.initial(),stream)
        parent=dict(kind='initial_dict',path=str(initial),identity=file_identity(initial),stage=None,
                    completed=0,verification=None)
        native=self.provenance_fixture()
        self.f.output.parent.mkdir()
        with patch.object(inputs,'OUTPUT_OWNER',self.f.base),patch.object(cp,'INITIAL_PATH',str(initial)):
            claim=ComparisonClaim(bundle);self.addCleanup(claim.close)
            with patch.object(self.train,'TENSORBOARD_FOUND',False):begin_output(p,claim,native,parent)
            for domain in ('connection_source', 'native_origin', 'repository'):
                bad = copy.deepcopy(native)
                if domain == 'repository': bad['native_origin']['repository'] = str(self.f.base)
                else:
                    target = bad[domain] if domain == 'connection_source' else bad[domain]['source']
                    del target['train.py']
                with self.subTest(boundary='context', domain=domain), \
                     self.assertRaisesRegex(ValueError, 'source_changed|comparison_native_origin'):
                    cp.prepare_context(bundle, bad, parent)
            paths=[]
            for k in p.inputs.config.save_updates:
                m.update_learning_rate(k)
                sum(g['params'][0].sum() for g in m.optimizer.param_groups).backward()
                m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
                expected=independent(m)
                marker=p.scene.save(k)
                directory=self.f.output/'checkpoints'/('completed-'+str(k));paths.append(directory)
                loaded=cp.read_checkpoint(directory);self.assertEqual(loaded['marker_identity'],marker)
                assert_expected(self,expected,loaded)
                with self.assertRaisesRegex(ValueError,'marker_schema'):formal_checkpoint.read_checkpoint(directory)
            c=profile.stage_config('growth',str(self.f.base/'next'))
            final=cp.read_checkpoint(paths[-1]);final['verified_final']=True
            with self.assertRaisesRegex(ValueError,'comparison_parent_unverified'):
                state.Initializer.from_checkpoint(final,c)
            intermediate=cp.read_checkpoint(paths[0]);intermediate['verified_final']=True
            with self.assertRaisesRegex(ValueError,'comparison_parent_unverified'):
                state.Initializer.from_checkpoint(intermediate,c)
            # Corrupt M but update the component hash so the provenance
            # validator, rather than only the outer hash, must reject it.
            directory = paths[-1]
            metadata_path, marker_path = directory/'metadata.json', directory/'complete.json'
            original_meta, original_marker = metadata_path.read_bytes(), marker_path.read_bytes()
            for domain in ('connection_source', 'native_origin', 'repository'):
                meta = json.loads(original_meta)
                if domain == 'repository':
                    meta['base']['provenance']['native_origin']['repository'] = str(self.f.base)
                else:
                    value = meta['base']['provenance'][domain]
                    del (value if domain == 'connection_source' else value['source'])['train.py']
                metadata_path.write_bytes(canonical(meta))
                marker = json.loads(original_marker)
                marker['files']['metadata.json'] = file_identity(metadata_path)
                marker_path.write_bytes(canonical(marker))
                with self.subTest(boundary='metadata', domain=domain), \
                     patch.object(self.torch, 'load', side_effect=AssertionError('T reached')) as load:
                    with self.assertRaisesRegex(ValueError, 'source_changed|comparison_native_origin'):
                        cp.read_checkpoint(directory)
                    load.assert_not_called()
            metadata_path.write_bytes(original_meta);marker_path.write_bytes(original_marker)
            # A changed historical recipe is also rejected during M, before T.
            recipe = Path(native['native_origin']['knn']['recipe']['path'])
            saved_recipe = recipe.read_bytes(); recipe.write_bytes(b'wrong recipe')
            with patch.object(self.torch, 'load', side_effect=AssertionError('T reached')) as load:
                with self.assertRaisesRegex(ValueError, 'build_file'): cp.read_checkpoint(directory)
                load.assert_not_called()
            recipe.write_bytes(saved_recipe)
            with self.assertRaises(FileExistsError):p.scene.save(p.inputs.config.save_updates[-1])
            self.assertTrue(p.scene._checkpoint_writer.failed)
            # Corruption is rejected before the serializer, not repaired.
            path=paths[-1]/'state.pt';path.write_bytes(path.read_bytes()[:-1])
            with patch.object(self.torch,'load',side_effect=AssertionError('T reached')) as load:
                with self.assertRaisesRegex(ValueError,'component_hash'):cp.read_checkpoint(paths[-1])
                load.assert_not_called()

    def test_provenance_binds_new_comparison_python(self):
        import staged_comparison_checkpoint as cp
        value = self.provenance_fixture()
        accepted = cp.provenance(value)
        # Independent inventory of the newly introduced execution consumers.
        # The current connection proof must cover these, not just formal*.py.
        required = {'comparison_profile.py', 'comparison_inputs.py',
                    'staged_comparison_state.py', 'staged_comparison_checkpoint.py',
                    'staged_comparison_runtime.py', 'staged_comparison_native_origin.json',
                    'pointops2/__init__.py', 'pointops2/functions/__init__.py',
                    'pointops2/functions/pointops.py'}
        self.assertFalse(required - set(accepted['connection_source']),
                         'accepted connection identity omits comparison consumers')

    def test_provenance_rejects_missing_native_source_identity(self):
        import copy
        import staged_comparison_checkpoint as cp
        value = self.provenance_fixture()
        origin = value['native_origin']
        removed = 'diff-gaussian-rasterization/cuda_rasterizer/forward.cu'
        self.assertIn(removed, origin['source'])
        del origin['source'][removed]
        # Valid binary/recipe files cannot substitute for a missing source proof.
        with self.assertRaisesRegex(ValueError, 'comparison_native_origin'):
            cp.provenance(value)

    def test_source_and_loaded_module_changes_rejected_for_both_domains(self):
        import formal_checkpoint_identity as identity
        import staged_comparison_checkpoint as cp
        value = self.provenance_fixture()
        helpers = ['comparison_profile.py', 'comparison_inputs.py',
                   'staged_comparison_checkpoint.py', 'staged_comparison_runtime.py',
                   'staged_comparison_state.py', 'pointops2/functions/pointops.py']
        for name in helpers:
            for mutation in ('missing', 'changed'):
                bad = copy.deepcopy(value)
                if mutation == 'missing': del bad['connection_source'][name]
                else: bad['connection_source'][name] = identity.byte_identity(b'changed')
                with self.subTest(name=name, mutation=mutation), \
                     self.assertRaisesRegex(ValueError, 'source_changed'):
                    cp.provenance(bad)
                formal = self.f.checkpoint_evidence().value()
                formal['source'] = bad['connection_source']
                with self.assertRaisesRegex(ValueError, 'source_changed'):
                    identity.validate_native(identity.NativeEvidence(identity.canonical(formal)))
        with patch.object(profile, '__file__', str(self.f.base/'foreign.py')):
            with self.assertRaisesRegex(ValueError, 'module_source'): cp.provenance(value)
            with self.assertRaisesRegex(ValueError, 'module_source'):
                identity.validate_native(self.f.checkpoint_evidence())

    def test_native_origin_mutations_and_trusted_record_substitution(self):
        import staged_comparison_checkpoint as cp
        from formal_checkpoint_identity import canonical, byte_identity, file_identity
        good = self.provenance_fixture()
        for mutation in ('missing', 'extra', 'changed', 'binding', 'binary', 'environment'):
            bad = copy.deepcopy(good); origin = bad['native_origin']
            if mutation == 'missing': del origin['source']['simple-knn/ext.cpp']
            if mutation == 'extra': origin['source']['simple-knn/other.cu'] = byte_identity(b'extra')
            if mutation == 'changed': origin['source']['simple-knn/ext.cpp'] = byte_identity(b'changed')
            if mutation == 'binding': origin['source']['gaussian_renderer/diff_gaussian_rasterization.py'] = byte_identity(b'other binding')
            if mutation == 'binary': origin['knn']['binary']['path'] += '.other'
            if mutation == 'environment': origin['environment']['python'] = 'other'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'comparison_native_origin'):
                cp.provenance(bad)
        # Changing the saved bytes, even together with candidate origin, cannot
        # change the separately frozen record identity.
        bad = copy.deepcopy(good);bad['native_origin']['environment']['python'] = 'substituted'
        self.origin_record.write_bytes(canonical(dict(native=bad['native_origin'])))
        with self.assertRaisesRegex(ValueError, 'comparison_native_record'): cp.provenance(bad)
        # A fixture with an independently anchored but incomplete/extra native
        # set must still fail the current required-membership check.
        for mutation in ('missing', 'extra', 'changed'):
            bad = copy.deepcopy(good)
            if mutation == 'missing': del bad['native_origin']['source']['simple-knn/ext.cpp']
            if mutation == 'extra': bad['native_origin']['source']['simple-knn/extra.cu'] = byte_identity(b'extra')
            if mutation == 'changed': bad['native_origin']['source']['simple-knn/ext.cpp'] = byte_identity(b'changed')
            self.origin_record.write_bytes(canonical(dict(native=bad['native_origin'])))
            self.origin_anchor.write_bytes(canonical(dict(path=str(self.origin_record),identity=file_identity(self.origin_record))))
            with self.subTest(anchored=mutation), self.assertRaisesRegex(ValueError, 'comparison_native_source'):
                cp.provenance(bad)

    def test_live_paths_and_environment_checks_are_not_bypassed(self):
        import staged_comparison_checkpoint as cp
        good = self.provenance_fixture()
        cp.provenance(good, live=True)
        for name in ('simple_knn._C', 'gaussian_renderer.diff_gaussian_rasterization'):
            module = sys.modules[name]
            if name.startswith('gaussian_renderer'): module = module._C
            with patch.object(module, '__file__', str(self.f.base/'foreign.so')), \
                 self.assertRaisesRegex(ValueError, 'native_loaded_path'):
                cp.provenance(good, live=True)
        bad = copy.deepcopy(good);bad['environment']['python'] = 'other'
        with self.assertRaisesRegex(ValueError, 'comparison_environment'): cp.provenance(bad)

    def test_historical_root_is_comparison_only(self):
        import formal_checkpoint_identity as identity
        import staged_comparison_checkpoint as cp
        value = self.provenance_fixture()
        self.assertNotEqual(value['native_origin']['repository'], str(identity.ROOT))
        self.assertEqual(cp.provenance(value, live=True), value)
        identity.validate_native(self.f.checkpoint_evidence())  # Current formal root still passes.
        old = copy.deepcopy(value['native_origin'])
        old['source'] = value['connection_source']  # Isolate root check, not stale Python.
        with self.assertRaisesRegex(ValueError, 'native_schema'):
            identity.validate_native(identity.NativeEvidence(identity.canonical(old)))
        with self.assertRaisesRegex(ValueError, 'native_schema'):
            identity._validate_native_build(identity.NativeEvidence(identity.canonical(old)))
        bad = copy.deepcopy(value); bad['native_origin']['repository'] = str(identity.ROOT)
        with self.assertRaisesRegex(ValueError, 'comparison_native_origin'): cp.provenance(bad)

    def test_bound_historical_root_recipe_and_build_boundaries(self):
        import formal_checkpoint_identity as identity
        import staged_comparison_checkpoint as cp
        good = self.provenance_fixture(); oldroot = Path(good['native_origin']['repository'])
        alias = self.f.base/'source-alias'; alias.symlink_to(oldroot, target_is_directory=True)
        child = oldroot/'build'; child.mkdir()
        cases = [('noncanonical', 'repository', str(oldroot)+'/../historical-source', 'native_source_root'),
                 ('alias', 'repository', str(alias), 'native_source_root'),
                 ('equal', 'build_root', str(oldroot), 'build_overlap'),
                 ('nested', 'build_root', str(child), 'build_overlap'),
                 ('ancestor', 'build_root', str(oldroot.parent), 'build_overlap'),
                 ('current', 'build_root', str(identity.ROOT), 'build_overlap'),
                 ('outside', 'build_root', str(self.f.source), 'build_file')]
        for label, key, new, error in cases:
            bad = copy.deepcopy(good); bad['native_origin'][key] = new
            self.anchor_fixture(bad)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, error): cp.provenance(bad)
        self.anchor_fixture(good)
        recipe = Path(good['native_origin']['knn']['recipe']['path'])
        saved = recipe.read_bytes()
        # Update fixture hashes/anchor too: reject wrong translation-unit root,
        # not merely a mismatching file digest.
        recipe.write_bytes(saved.replace(str(oldroot).encode(), str(identity.ROOT).encode()))
        bad = copy.deepcopy(good)
        bad['native_origin']['knn']['recipe']['identity'] = identity.file_identity(recipe)
        self.anchor_fixture(bad)
        with self.assertRaisesRegex(ValueError, 'recipe_sources'): cp.provenance(bad)

    def test_historical_files_and_build_products_remain_bound(self):
        import staged_comparison_checkpoint as cp
        good = self.provenance_fixture(); origin = good['native_origin']
        source = Path(origin['repository'])/'simple-knn/ext.cpp'
        paths = [(source, 'comparison_native_source'),
                 (Path(origin['knn']['binary']['path']), 'build_file'),
                 (Path(origin['raster']['recipe']['path']), 'build_file'),
                 (Path(next(iter(origin['raster']['objects']))), 'build_file')]
        for path, error in paths:
            before = path.read_bytes(); path.write_bytes(b'changed CPU fixture')
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, error): cp.provenance(good)
            path.write_bytes(before)
        source.unlink()
        with self.assertRaises(FileNotFoundError): cp.provenance(good)

    def test_inventory_covers_current_local_python_import_dependencies(self):
        # Static audit only: do not import these dependencies or load native.
        import ast
        from formal_checkpoint_identity import ROOT, source_snapshot
        names = set(source_snapshot())
        for name in sorted(names):
            if not name.endswith('.py') or name.startswith('simple-knn/'): continue
            for node in ast.walk(ast.parse((ROOT/name).read_text())):
                if isinstance(node, ast.Import): modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        parts = name.split('/')[:-1]
                        base = parts[:len(parts)-node.level+1]
                        module = '.'.join(base+([node.module] if node.module else []))
                    else: module = node.module or ''
                    modules = [module]+[module+'.'+a.name for a in node.names if a.name != '*']
                else: continue
                for module in modules:
                    for path in (module.replace('.', '/')+'.py', module.replace('.', '/')+'/__init__.py'):
                        if (ROOT/path).is_file():
                            self.assertIn(path, names, (name, module))


if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)
