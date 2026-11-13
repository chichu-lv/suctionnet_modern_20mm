"""Credited-sample arithmetic and configuration contracts, without torch/CUDA."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from suction20.pretrain_budget import CONTRACT_VERSION, resolve_budget, resolve_config, sampling_epoch


def config(effective=4, micro=None):
    return {'training': {'budget_contract': CONTRACT_VERSION, 'microbatch_size': micro or effective,
                         'gradient_accumulation_steps': 1 if micro is None else effective // micro,
                         'effective_batch_size': effective},
            'runtime_profile': {'profile_id': 'cpu-contract-test', 'selection_status': 'selected', 'training_version': 'v3'}}


@pytest.mark.parametrize('effective,target,warmup,every,prefix', [
    (4,57600,2880,5760,100), (8,28800,1440,2880,50), (16,14400,720,1440,25),
    (12,19200,960,1920,34)])
def test_credit_budget_and_boundaries(effective,target,warmup,every,prefix):
    original=config(effective); frozen=copy.deepcopy(original)
    resolved=resolve_config(original); budget=resolve_budget(resolved)
    assert original==frozen
    assert budget['target_updates']==target and budget['warmup_updates']==warmup
    assert budget['validation_steps']==[every * i for i in range(1,11)]
    assert budget['prefix_step']==prefix==budget['checkpoint_interval_updates']
    assert target*effective==230400 and warmup*effective==11520 and every*effective==23040
    assert resolve_config(json.loads(json.dumps(resolved)))==resolved
    assert sampling_epoch(budget,11520//effective-1,99)==1
    assert sampling_epoch(budget,11520//effective,99)==2
    # AMP consumed schedule can cross many epochs while credit stays unchanged.
    assert sampling_epoch(budget,0,999)==0


@pytest.mark.parametrize('effective', [7,11,17,230400])
def test_fractional_target_warmup_validation_or_epoch_rejected(effective):
    with pytest.raises(ValueError,match='divisible'):
        resolve_config(config(effective))


@pytest.mark.parametrize('section,key,value', [('training','max_optimizer_steps',57600),
                                              ('training','validate_every_optimizer_steps',5760),
                                              ('scheduler','warmup_optimizer_steps',2880),
                                              ('scheduler','total_optimizer_steps',57600)])
def test_v2_step_fields_cannot_silently_expand_v3_sample_budget(section,key,value):
    changed=config(8);changed.setdefault(section,{})[key]=value
    with pytest.raises(ValueError,match='conflicts'):
        resolve_config(changed)


def test_legacy_smoke_retains_original_semantics_and_sampling_epoch():
    legacy={'training':{'max_optimizer_steps':3,'effective_batch_size':4},'data':{'smoke_frame_keys':[[0,0]]}}
    assert resolve_config(legacy)==legacy
    budget=resolve_budget(legacy)
    assert not budget['formal'] and budget['validation_steps']==[3] and budget['prefix_step']==100
    assert sampling_epoch(budget,0,7)==7


def test_formal_marker_cannot_be_removed_or_replaced_with_smoke():
    changed=config();changed['training'].pop('budget_contract');changed['training']['target_effective_samples']=230400
    with pytest.raises(ValueError,match='explicit'):
        resolve_budget(changed)
    changed=config();changed['data']={'smoke_frame_keys':[[0,0]]}
    with pytest.raises(ValueError,match='smoke'):
        resolve_budget(changed)


def test_all_six_formal_configs_are_pending_profiles_with_resolvable_v3_budget():
    for experiment in range(6):
        original=yaml.safe_load(Path(f'configs/pretrain_e{experiment}.yaml').read_text())
        assert original['runtime_profile']['selection_status']=='pending'
        assert 'max_optimizer_steps' not in original['training']
        for effective in (4,8,16):
            candidate=copy.deepcopy(original)
            candidate['training'].update(microbatch_size=effective,gradient_accumulation_steps=1,effective_batch_size=effective)
            assert resolve_budget(resolve_config(candidate))['target_samples']==230400


def resumed_state():
    budget=resolve_budget(config(8));step=budget['validation_steps'][0]
    reference={'path':'validations/2880-node.json','sha256':'a'*64}
    node={'step':step,'effective_samples':23040,'frame_count':320}
    state={'global_step':step,'best_metric':{'global_step':step,'values':[.4,.2]},'runtime_state':{
        'budget':budget,'effective_samples_completed':step*8,'ledger':{'attempts':step+2,'skipped':2,'frames':(step+2)*8},
        'pending_validation':None,'validations':{str(step):reference},'completed_validation_nodes':[node],
        'last_completed_validation':dict(node,result=reference,metrics={'instance_Top1_Q20_at_0_4':.4,'AP_R20':.2})}}
    return state,budget


def test_resume_progress_accepts_consistent_completed_and_pending_nodes():
    from suction20.pretrain_budget import validate_resume_progress
    state,budget=resumed_state();validate_resume_progress(state,budget)
    runtime=state['runtime_state'];runtime.update(pending_validation={'step':state['global_step'],'id':'pending'},
                                                validations={},completed_validation_nodes=[],last_completed_validation=None)
    state['best_metric']=None
    validate_resume_progress(state,budget)


@pytest.mark.parametrize('mutation', ['budget','credit','attempts','skips','frames','references','nodes','last','pending','best'])
def test_resume_progress_rejects_corrupt_counters_and_validation_state(mutation):
    from suction20.pretrain_budget import validate_resume_progress
    state,budget=resumed_state();state=copy.deepcopy(state);runtime=state['runtime_state']
    if mutation=='budget':runtime['budget']['target_samples']+=8
    elif mutation=='credit':runtime['effective_samples_completed']+=8
    elif mutation=='attempts':runtime['ledger']['attempts']+=1
    elif mutation=='skips':runtime['ledger']['skipped']+=1
    elif mutation=='frames':runtime['ledger']['frames']+=1
    elif mutation=='references':runtime['validations']={}
    elif mutation=='nodes':runtime['completed_validation_nodes'][0]['frame_count']=319
    elif mutation=='last':runtime['last_completed_validation']['result']={'path':'wrong'}
    elif mutation=='pending':runtime['pending_validation']={'step':state['global_step']+1}
    else:state['best_metric']['global_step']=1
    with pytest.raises(ValueError,match='Resumed|resumed'):
        validate_resume_progress(state,budget)
