"""Four-frame paired diagnostic with one real, retained backbone graph per frame.

Independent CUDA sparse-backbone forwards need not produce identical features.
This diagnostic isolates the changed data/geometry/suffix pipeline by feeding both
actual implementations identical backbone outputs. It does not replace the failed
independent-process result or establish end-to-end determinism or training speed.
No optimizer is constructed, no update occurs, and no formal run state is written.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def project_module(name):
    return (name in ('models', 'suction20', '_suction20_pinned_backbone') or
            name.startswith(('models.', 'suction20.')))


class Checkout:
    """Temporarily activate one checkout, including all lazy project imports.

    Torch/NumPy/spconv remain shared dependencies. Model/function objects retain
    their original module globals while the active project namespace is swapped.
    Do not use this process-global context concurrently or with loader workers.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.modules = {}

    @contextmanager
    def active(self):
        previous = {k: v for k, v in sys.modules.items() if project_module(k)}
        old_path = list(sys.path)
        for name in previous:
            del sys.modules[name]
        sys.modules.update(self.modules)
        sys.path.insert(0, str(self.root))
        try:
            yield self
        finally:
            self.modules = {k: v for k, v in sys.modules.items() if project_module(k)}
            for name in self.modules:
                del sys.modules[name]
            sys.modules.update(previous)
            sys.path[:] = old_path

    def module(self, name):
        module = importlib.import_module(name)
        filename = getattr(module, '__file__', None)
        if filename and self.root not in Path(filename).resolve().parents:
            raise RuntimeError(f'Project import escaped checkout: {name}: {filename}')
        return module


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Comparisons:
    def __init__(self, amp):
        self.atol, self.rtol = (2e-4, 2e-3) if amp else (1e-6, 1e-4)
        self.failures, self.numeric = [], {}

    def check(self, left, right, path, *, exact=False):
        import torch
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
                self.failures.append(path + ': tensor contract'); return
            if left.is_floating_point():
                a, b = left.double(), right.double()
                difference = (a - b).abs()
                self.numeric[path] = dict(max_abs=float(difference.max()) if a.numel() else 0.,
                    max_rel=float((difference / a.abs().clamp_min(1e-12)).max()) if a.numel() else 0.)
                okay = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                okay = okay and (torch.equal(left, right) if exact else torch.allclose(left, right, atol=self.atol, rtol=self.rtol))
            else:
                okay = torch.equal(left, right)
            if not okay:
                self.failures.append(path)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                self.failures.append(path + ': keys'); return
            for key in left:
                self.check(left[key], right[key], path + '.' + str(key), exact=exact)
        elif isinstance(left, (tuple, list)):
            if not isinstance(right, (tuple, list)) or len(left) != len(right):
                self.failures.append(path + ': length'); return
            for index, (a, b) in enumerate(zip(left, right)):
                self.check(a, b, f'{path}.{index}', exact=exact)
        elif left != right:
            self.failures.append(path)


def cpu_gradients(names, values):
    return {name: None if value is None else value.detach().cpu().clone()
            for name, value in zip(names, values)}


def gradient_health(values):
    import torch
    active = [v for v in values if v is not None]
    return dict(finite=all(bool(torch.isfinite(v).all()) for v in active),
                nonzero=any(bool(torch.count_nonzero(v)) for v in active),
                tensors=len(active))


def run_suffix(model, features, inputs, targets, loss_fn, *, amp, gradient_scale,
               accumulation, model_kwargs=None, loss_kwargs=None):
    """Run one sequential suffix graph; return CPU evidence, no retained graph.

    features is the common real-backbone boundary, never modified by this branch.
    A separate leaf and parameterless replacement prevent a second backbone call.
    """
    import torch
    from torch import nn
    leaf = features.detach().clone().requires_grad_(True)

    class Boundary(nn.Module):
        def forward(self, ignored_inputs):
            return leaf

    actual_backbone = model.backbone
    model.backbone = Boundary()
    try:
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        with torch.autocast(device_type=features.device.type, enabled=amp):
            prediction = model(inputs, sampler_mode='train', targets=targets, **(model_kwargs or {}))
        losses = loss_fn(prediction, targets, inputs, **(loss_kwargs or {}))
        if not torch.isfinite(losses['total']):
            raise ValueError('Non-finite paired suffix loss')
        gradients = torch.autograd.grad((losses['total'] / accumulation) * gradient_scale,
                                        [leaf] + [p for _, p in named], allow_unused=True)
        if gradients[0] is None:
            raise ValueError('Suffix disconnected from the real backbone feature boundary')
        result = dict(outputs={name: value.detach().cpu().clone() for name, value in prediction.items()
                               if isinstance(value, torch.Tensor)},
                      losses={name: value.detach().cpu().clone() for name, value in losses.items()},
                      cotangent=gradients[0].detach().cpu().clone(),
                      suffix_gradients=cpu_gradients([name for name, _ in named], gradients[1:]),
                      suffix_buffers={name: value.detach().cpu().clone() for name, value in model.named_buffers()})
        return result
    finally:
        model.backbone = actual_backbone


def backbone_vjp(features, named_parameters, cotangent, *, retain_graph):
    import torch
    values = torch.autograd.grad(features, [p for _, p in named_parameters],
        grad_outputs=cotangent.to(device=features.device, dtype=features.dtype),
        retain_graph=retain_graph, allow_unused=True)
    return cpu_gradients([name for name, _ in named_parameters], values)


def add_gradients(total, values):
    for name, value in values.items():
        if name not in total:
            total[name] = None if value is None else value.clone()
        elif value is not None:
            total[name] = value.clone() if total[name] is None else total[name] + value


def install_query_capture(model, destination):
    grouping = getattr(model, 'contact_grouping', None)
    if grouping is None:
        return
    original = grouping.query

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        destination.append(value)  # CPU arrays only; no model graph or RNG changes.
        return value
    grouping.query = capture


def run(args, report):
    import torch
    roots = [Checkout(args.baseline_repo), Checkout(args.candidate_repo)]
    unchanged_sources = {}
    for relative in ('models/heads.py',):
        hashes = [digest(root.root / relative) for root in roots]
        if hashes[0] != hashes[1]:
            raise ValueError(f'Shared-backbone dependency changed: {relative}')
        unchanged_sources[relative] = hashes[0]
    report['shared_backbone_source_sha256'] = unchanged_sources
    report['backbone_wrapper_source_sha256'] = {
        label: digest(root.root / 'models/backbones/spconv_resunet14.py')
        for label, root in zip(('baseline', 'optimized'), roots)}
    bundles = []
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    fixed_frames = None
    sampler_state = state['sampler_state']
    if args.frame_plan_report:
        fixed_frames = json.loads(Path(args.frame_plan_report).read_text())
        if len(fixed_frames.get('frames', [])) != 4:
            raise ValueError('Frame plan must contain exactly four previously recorded frames')
        sampler_state = fixed_frames['sampler_before']
        report['frame_plan_report_sha256'] = digest(args.frame_plan_report)
    report.update(checkpoint_step=state['global_step'], checkpoint_sampler_before=state['sampler_state'],
                  sampler_before=sampler_state,
                  initial_rng=fingerprint(state['rng_state']))
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Real paired pipeline diagnostic requires an authorized CUDA device')
    device = torch.device('cuda', device.index or 0)
    for phase, checkout in zip(('baseline', 'optimized'), roots):
        with checkout.active():
            training = checkout.module('suction20.training')
            runner = checkout.module('suction20.runner')
            budget_module = checkout.module('suction20.pretrain_budget')
            datasets = checkout.module('suction20.datasets')
            config = runner.override_paths(budget_module.resolve_config(training.load_config(args.config)), args)
            training.validate_resume_config(state['training_config'], config)
            budget = budget_module.resolve_budget(config)
            if budget['effective_batch_size'] != 4 or config['training']['microbatch_size'] != 1:
                raise ValueError('Paired diagnostic requires the approved 1 x 4 profile')
            training.seed_everything(config.get('seed', 42))
            torch.set_num_threads(config.get('runtime', {}).get('cpu_threads', 8))
            cap = config['runtime']['gpu_memory_limit_gib'] * 1024**3
            torch.cuda.set_per_process_memory_fraction(min(1., cap / torch.cuda.get_device_properties(device).total_memory), device)
            performance = None
            performance_path = (args.performance_config if phase == 'optimized'
                                else args.baseline_performance_config)
            if performance_path:
                performance_module = checkout.module('suction20.performance')
                config = performance_module.attach_performance(config, performance_path)
                performance = config['runtime']['performance']
            source = runner.make_source_dataset(config, augment=True)
            model = runner.make_model(config, source.data_cfg, device)
            model.load_state_dict(state['model_state_dict'], strict=True)
            model.train()
            schedule = training.FrameSchedule(datasets.split_scene_ids('train'), seed=config.get('sampling_seed', 20261110),
                views_per_scene=config['data'].get('views_per_train_scene_per_epoch', 64),
                frame_keys=config['data'].get('smoke_frame_keys'))
            schedule.load_state_dict(sampler_state)
            queries = []
            install_query_capture(model, queries)
            identity = dict(repo=str(checkout.root), code_revision=training.code_revision(), worktree_sha256=training.worktree_digest())
            if performance is not None:
                identity['performance'] = performance_module.performance_identity(config)
            bundles.append(dict(phase=phase, checkout=checkout, training=training, datasets=datasets,
                source=source, model=model, config=config, budget=budget, schedule=schedule,
                has_cpu_context=performance is not None,
                keys={key: i for i, key in enumerate(source.keys)}, queries=queries, identity=identity,
                loss_fn=checkout.module('suction20.losses').source_losses,
                sampling_epoch=budget_module.sampling_epoch))
    # One common pinned sparse network. Wrapper alignment is checked separately
    # on this network's same raw output, before projection; no second forward.
    baseline, optimized = bundles
    if fingerprint(baseline['source'].data_cfg) != fingerprint(optimized['source'].data_cfg):
        raise ValueError('Frozen source label configuration differs')
    with roots[0].active():
        baseline_wrapper = roots[0].module('models.backbones.spconv_resunet14')
    with roots[1].active():
        candidate_wrapper = roots[1].module('models.backbones.spconv_resunet14')
    if baseline_wrapper._BLOBS != candidate_wrapper._BLOBS:
        raise ValueError('Pinned upstream sparse-network sources differ')
    report['pinned_backbone_blobs'] = dict(baseline_wrapper._BLOBS)
    common_backbone = baseline['model'].backbone
    old_state, new_state = common_backbone.state_dict(), optimized['model'].backbone.state_dict()
    if old_state.keys() != new_state.keys() or any(not torch.equal(old_state[k], new_state[k]) for k in old_state):
        raise ValueError('The two backbone initial states differ')
    optimized['model'].backbone.cpu()  # No unused GPU model copy during measurement.
    del old_state, new_state, state['model_state_dict'], state['optimizer_state_dict']
    amp = bool(baseline['config']['training'].get('amp', True))
    if amp != bool(optimized['config']['training'].get('amp', True)):
        raise ValueError('AMP configuration differs')
    scale = float(state['scaler_state_dict'].get('scale', 1.)) if amp else 1.
    comparisons = Comparisons(amp)
    report.update(amp=amp, gradient_scale=scale, atol=comparisons.atol, rtol=comparisons.rtol,
                  identities={item['phase']: item['identity'] for item in bundles}, frames=[], numeric_errors=comparisons.numeric,
                  failures=comparisons.failures)
    totals = {phase: dict(suffix={}, backbone={}) for phase in ('baseline', 'optimized')}
    with roots[0].active():
        baseline['training'].restore_rng(state['rng_state'])
    for frame_index in range(4):
        started = time.perf_counter()
        prepared, evidence, rng_after, frame = [], [], [], dict(index=frame_index)
        for item in bundles:
            with item['checkout'].active():
                key, aug_seed = item['schedule'].next()
                key = item['datasets'].frame_key(key, item['config']['data'].get('camera', 'realsense'))
                sample = item['source'].get_sample(item['keys'][key], aug_seed)
                packed = item['datasets'].collate_public([sample], item['config']['model'].get('voxel_size_m', .002))
                data = fingerprint({key: packed[key] for key in ('inputs', 'targets', 'meta')})
                prepared.append(dict(packed=packed, frame_key=list(key), augmentation_seed=aug_seed, data=data))
                del sample
        if fixed_frames is not None:
            for key in ('frame_key', 'augmentation_seed'):
                comparisons.check(prepared[0][key], fixed_frames['frames'][frame_index][key],
                    f'frame.{frame_index}.fixed_plan.{key}', exact=True)
        for key in ('frame_key', 'augmentation_seed', 'data'):
            comparisons.check(prepared[0][key], prepared[1][key], f'frame.{frame_index}.{key}', exact=True)
        if any(path.startswith(f'frame.{frame_index}.') for path in comparisons.failures):
            raise ValueError('Paired data contract differs; stopped before model computation')
        frame.update({key: prepared[0][key] for key in ('frame_key', 'augmentation_seed', 'data')})
        with roots[0].active():
            inputs = baseline['training'].move_to_device(prepared[0]['packed']['inputs'], device)
            common_backbone.train()
            with torch.autocast(device_type='cuda', enabled=amp):
                alignment_checks = []
                handle = None
                if optimized['config']['runtime'].get('performance', {}).get('backbone_coordinate_fastpath', False):
                    def capture_alignment(network, arguments, output):
                        coords = arguments[0].indices
                        rows = baseline_wrapper.coordinate_rows(coords, output.indices)
                        aligned = candidate_wrapper.align_backbone_features(
                            coords, output.indices, output.features, fastpath=True)
                        comparisons.check(output.indices[rows], coords,
                            f'frame.{frame_index}.backbone_coordinate_rows', exact=True)
                        comparisons.check(output.features[rows], aligned,
                            f'frame.{frame_index}.backbone_aligned_features', exact=True)
                        same = candidate_wrapper.same_coordinate_storage(coords, output.indices)
                        alignment_checks.append(dict(same_storage=bool(same),
                            rows=int(len(coords)), exact_identity_rows=bool(torch.equal(
                                rows, torch.arange(len(rows), device=rows.device)))))
                    handle = common_backbone.network.register_forward_hook(capture_alignment)
                try:
                    features = common_backbone(inputs)
                finally:
                    if handle is not None:
                        handle.remove()
                if handle is not None and len(alignment_checks) != 1:
                    raise ValueError('Expected one raw sparse-output alignment check')
                frame['backbone_alignment_checks'] = alignment_checks
            post_backbone_rng = baseline['training'].capture_rng()
        frame['shared_backbone_features'] = fingerprint(features)
        named_backbone = [(name, p) for name, p in common_backbone.named_parameters() if p.requires_grad]
        for item, data in zip(bundles, prepared):
            with item['checkout'].active():
                item['training'].restore_rng(post_backbone_rng)
                item['model'].set_epoch(item['sampling_epoch'](item['budget'], state['global_step'], item['schedule'].epoch))
                packed = data['packed']
                branch_inputs = inputs if item is baseline else item['training'].move_to_device(packed['inputs'], device)
                targets = item['training'].move_to_device(packed['targets'], device)
                kwargs = {}
                if item['has_cpu_context']:
                    contexts = item['checkout'].module('suction20.cpu_geometry')
                    kwargs = dict(cpu_geometry=contexts.CpuGeometry.from_inputs(packed['inputs']),
                        sampler_targets_cpu=contexts.SamplerTargetsCPU.from_targets(packed['targets']))
                item['queries'].clear()
                result = run_suffix(item['model'], features, branch_inputs, targets, item['loss_fn'],
                    amp=amp, gradient_scale=scale, accumulation=4, model_kwargs=kwargs,
                    loss_kwargs=dict(quality_target=item['config']['model']['quality_target'],
                        beta=item['config'].get('loss', {}).get('smooth_l1_beta', .05),
                        weights=item['config'].get('loss', {}).get('weights')))
                result['queries'] = item['queries'][:]
                result['rng_after'] = item['training'].capture_rng()
                for label, values in (('boundary', [result['cotangent']]),
                                      ('suffix', result['suffix_gradients'].values())):
                    health = gradient_health(values)
                    if not health['finite'] or not health['nonzero']:
                        comparisons.failures.append(f'frame.{frame_index}.{item["phase"]}.{label}_gradient')
                torch.save(result, Path(args.out_dir) / f'{item["phase"]}-frame-{frame_index}.pt')
                add_gradients(totals[item['phase']]['suffix'], result['suffix_gradients'])
                evidence.append(result)
                rng_after.append(result['rng_after'])
                del branch_inputs, targets, kwargs
        for key in ('outputs', 'losses', 'cotangent', 'suffix_gradients', 'suffix_buffers'):
            comparisons.check(evidence[0][key], evidence[1][key], f'frame.{frame_index}.{key}')
        comparisons.check(fingerprint(evidence[0]['queries']), fingerprint(evidence[1]['queries']),
                          f'frame.{frame_index}.queries', exact=True)
        comparisons.check(fingerprint(rng_after[0]), fingerprint(rng_after[1]), f'frame.{frame_index}.rng_after', exact=True)
        # No suffix graph remains. Apply both actual cotangents to the SAME real
        # retained backbone graph. Release that graph after the second VJP.
        vjps = []
        with roots[0].active():
            for index, item in enumerate(bundles):
                value = backbone_vjp(features, named_backbone, evidence[index]['cotangent'], retain_graph=index == 0)
                health = gradient_health(value.values())
                if not health['finite'] or not health['nonzero']:
                    comparisons.failures.append(f'frame.{frame_index}.{item["phase"]}.backbone_gradient')
                frame.setdefault('gradient_health', {})[item['phase']] = dict(backbone=health,
                    boundary=gradient_health([evidence[index]['cotangent']]),
                    suffix=gradient_health(evidence[index]['suffix_gradients'].values()))
                torch.save(value, Path(args.out_dir) / f'{item["phase"]}-backbone-vjp-{frame_index}.pt')
                add_gradients(totals[item['phase']]['backbone'], value)
                vjps.append(value)
            comparisons.check(vjps[0], vjps[1], f'frame.{frame_index}.backbone_vjp')
            baseline['training'].restore_rng(rng_after[0])
        frame.update(losses={item['phase']: {k: float(v) for k, v in evidence[index]['losses'].items()}
                            for index, item in enumerate(bundles)},
                     seconds_including_evidence_and_checks=time.perf_counter()-started)
        report['frames'].append(frame)
        print(json.dumps(dict(frame=frame_index, failures_so_far=len(comparisons.failures))), flush=True)
        for item in bundles:
            item['queries'].clear()
        del features, inputs, prepared, packed, data, evidence, result, rng_after, named_backbone, value, vjps
    comparisons.check(totals['baseline'], totals['optimized'], 'accumulated_gradients')
    comparisons.check(baseline['schedule'].state_dict(), optimized['schedule'].state_dict(), 'sampler_after', exact=True)
    report['sampler_after'] = baseline['schedule'].state_dict()
    torch.save(totals, Path(args.out_dir) / 'accumulated-gradients.pt')
    report['passed'] = not comparisons.failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-repo', required=True)
    parser.add_argument('--candidate-repo', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frame-plan-report',
                        help='Reuse four frame IDs/seeds from an earlier paired report; diagnostic schedule only')
    parser.add_argument('--config', required=True)
    parser.add_argument('--performance-config', required=True)
    parser.add_argument('--baseline-performance-config',
                        help='Explicit baseline sidecar; pass P1 when comparing P1 to P2')
    parser.add_argument('--source-root')
    parser.add_argument('--label-root')
    parser.add_argument('--backbone-root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new paired-comparison output directory')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version='suction20.paired-pipeline-comparison.v1', passed=False,
        scope='paired suffix/data/geometry plus VJP through one shared real backbone graph per frame',
        checkpoint_sha256=digest(args.checkpoint), config_sha256=digest(args.config),
        performance_config_sha256=digest(args.performance_config),
        baseline_performance_config_sha256=(digest(args.baseline_performance_config)
            if args.baseline_performance_config else None), optimizer_updates=0,
        limitations=['Does not supersede the failed independent-process end-to-end comparison.',
            'Shared-backbone boundary removes independent sparse CUDA forward variation from this diagnostic.',
            'Two sequential backward VJPs may retain CUDA numerical variation; existing tolerances still apply.',
            'AMP gradients use the same original checkpoint scale; no unscale or optimizer step.',
            'Sequential synchronous data preparation; does not benchmark prefetch or training throughput.',
            'Four real frames only; no convergence or physical-quality conclusion.'])
    try:
        run(args, report)
    except Exception as error:
        report.update(passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (out / 'paired-comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=report['passed'], report=str(out/'paired-comparison.json'))))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
