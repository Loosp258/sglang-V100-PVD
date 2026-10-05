"""The real serving config admits each ordered combination, never enables it implicitly."""
import json
from types import SimpleNamespace
import pytest
from sglang.srt.disaggregation.pvd import oasis_startup as startup


def config(stage):
    value=dict(eagle_source='source',eagle_checkpoint='checkpoint',eagle_manifest='manifest',
        vector_space='target-Q',capacity=32,max_new=16,top_k=4,workers=2,timeout_seconds=5,
        max_sequence_tokens=64,max_decode_steps=16,request_budget_bytes=1<<28,
        request_scratch_bytes=1<<20,bootstrap_budget_bytes=1<<28,bootstrap_transient_bytes=1<<20,
        overlap=True)
    if stage>=1:value.update(fused_search_delivery=True,binary_queries=True)
    if stage>=2:value['reuse_receive_slots']=True
    if stage>=3:value['compact_cache_snapshots']=True
    if stage>=4:value.update(reuse_pinned_scratch=True,event_bank_ready=True)
    if stage>=5:value['binary_control_channel']=True
    return value


def load(monkeypatch,tmp_path,value):
    path=tmp_path/'config.json';path.write_text(json.dumps(value))
    scheduler=SimpleNamespace(server_args=SimpleNamespace(pvd_oasis_config=str(path)),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
            model_config=SimpleNamespace(context_len=128))))
    monkeypatch.setattr(startup,'OasisSchedulerBinding',lambda *args:SimpleNamespace())
    monkeypatch.setattr(startup,'OasisResources',lambda scheduler,cfg:
        SimpleNamespace(config=cfg,prepare=lambda *args:None))
    startup.maybe_install_oasis(scheduler)
    return scheduler.pvd_oasis_resources.config


@pytest.mark.parametrize('stage',range(6))
def test_explicit_ordered_config_combinations(monkeypatch,tmp_path,stage):
    expected=config(stage);actual=load(monkeypatch,tmp_path,expected)
    for name,value in expected.items():assert actual[name]==value
    assert actual['binary_control_channel'] is (stage==5)


@pytest.mark.parametrize('option',['binary_queries','fused_search_delivery','reuse_receive_slots',
    'compact_cache_snapshots','reuse_pinned_scratch','event_bank_ready','binary_control_channel','fused_zero_miss_proof','async_layer_jobs'])
def test_nonbool_optimization_option_refused(monkeypatch,tmp_path,option):
    value=config(5);value[option]=1
    with pytest.raises(ValueError):load(monkeypatch,tmp_path,value)


@pytest.mark.parametrize('stage', range(1, 6))
def test_fused_ready_cleanup_with_each_followup_combination(monkeypatch, tmp_path, stage):
    value = config(stage)
    value['ready_before_cleanup'] = True
    assert load(monkeypatch, tmp_path, value)['ready_before_cleanup'] is True


def test_zero_miss_proof_requires_fusion_and_is_default_off(monkeypatch, tmp_path):
    assert load(monkeypatch, tmp_path, config(5))['fused_zero_miss_proof'] is False
    value = config(5)
    value['fused_zero_miss_proof'] = True
    assert load(monkeypatch, tmp_path, value)['fused_zero_miss_proof'] is True
    value['fused_search_delivery'] = False
    with pytest.raises(ValueError): load(monkeypatch, tmp_path, value)


def test_async_jobs_require_channel_and_ready_cleanup(monkeypatch, tmp_path):
    value=config(5)
    assert load(monkeypatch,tmp_path,value)['async_layer_jobs'] is False
    value.update(ready_before_cleanup=True,async_layer_jobs=True,fused_zero_miss_proof=True,request_scratch_bytes=2<<20)
    assert load(monkeypatch,tmp_path,value)['async_layer_jobs'] is True
    for field in ('binary_control_channel','ready_before_cleanup'):
        bad={**value,field:False}
        with pytest.raises(ValueError): load(monkeypatch,tmp_path,bad)
    from sglang.srt.disaggregation.pvd.oasis_async_jobs import async_tensor_bound
    bad={**value,'request_scratch_bytes':async_tensor_bound(value['capacity'])-1}
    with pytest.raises(ValueError): load(monkeypatch,tmp_path,bad)
