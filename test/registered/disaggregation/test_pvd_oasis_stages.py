"""Real Python stages/deadlines/affinity; no CUDA or native failure claim."""
import threading
import time

import pytest

from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead, LayerReply, LayerTicket
from sglang.srt.disaggregation.pvd.oasis_stages import BoundedLayerStages


def make(callbacks, *, limit=56, workers=(2, 2, 1), retired=None):
    def initialize(stage, index):
        return stage, index, threading.get_ident()
    def retire(stage, state):
        assert state[0] == stage and state[2] == threading.get_ident()
        if retired is not None:
            retired.append(state)
    return BoundedLayerStages(callbacks, initialize=initialize, retire=retire,
                              max_pending=limit, workers=workers)


def submit(stages, layer, value=None, *, timeout=5, published=None, cleanup=None):
    ticket = LayerTicket('request', 'incarnation', 0, layer)
    return stages.submit(ticket, ticket, ticket if value is None else value,
        published=time.monotonic() if published is None else published,
        timeout=timeout, cleanup=cleanup or (lambda payload: None))


def test_blocked_delivery_does_not_occupy_search_and_close_joins_owned_work():
    searched, delivery_entered, release = threading.Event(), threading.Event(), threading.Event()
    retired = []
    def search(ticket, state):
        if ticket.layer == 1:
            searched.set()
        return ticket
    def delivery(ticket, state):
        delivery_entered.set()
        assert release.wait(5)
        return ticket
    stages = make((search, delivery, lambda ticket, state: LayerReply(ticket, ticket.layer)),
                  workers=(1, 1, 1), retired=retired)
    first = submit(stages, 0)
    assert delivery_entered.wait(5)
    second = submit(stages, 1)
    assert searched.wait(5), 'native delivery blocked the independent search stage'
    assert not first.cancel() and not second.cancel(), 'published owners were abandoned'
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (stages.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.03)
    with pytest.raises(RuntimeError, match='closed'):
        submit(stages, 2)
    release.set()
    closer.join(5)
    assert closed.is_set()
    assert first.result(5)[0].value == 0 and second.result(5)[0].value == 1
    assert len(retired) == 3 and len({state[2] for state in retired}) == 3
    assert stages.snapshot()['pending'] == stages.snapshot()['retained_states'] == 0


def test_bounded_admission_and_duplicate_ticket_preserve_original_work():
    entered, release = threading.Event(), threading.Event()
    def search(ticket, state):
        entered.set()
        assert release.wait(5)
        return ticket
    stages = make((search, lambda t, s: t, lambda t, s: LayerReply(t, 'ok')), limit=1)
    future = submit(stages, 0)
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match='bounded'):
        submit(stages, 1)
    with pytest.raises(RuntimeError, match='duplicate'):
        submit(stages, 0)
    release.set()
    assert future.result(5)[0].value == 'ok'
    stages.close()
    assert stages.snapshot()['peak_pending'] == 1


def test_publication_deadline_includes_queue_and_drains_before_failure():
    drained, executed = [], []
    stages = make((lambda t, s: executed.append(t) or t,
                   lambda t, s: t, lambda t, s: LayerReply(t, None)))
    future = submit(stages, 0, published=time.monotonic() - 2, timeout=1,
                    cleanup=lambda ticket: drained.append(ticket))
    with pytest.raises(TimeoutError, match='publication'):
        future.result(5)
    assert len(drained) == 1 and not executed
    stages.close()
    assert stages.snapshot()['failed'] == 1


def test_foreign_reply_is_rejected_after_explicit_owner_drainage():
    drained = []
    stages = make((lambda t, s: t, lambda t, s: t,
        lambda t, s: LayerReply(LayerTicket('r', 'old', t.step, t.layer), None)))
    future = submit(stages, 0, cleanup=lambda t: drained.append(t))
    with pytest.raises(RuntimeError, match='foreign'):
        future.result(5)
    assert len(drained) == 1
    stages.close()


def test_unproven_stage_retirement_keeps_state_and_refuses_successful_close():
    states = []
    def initialize(stage, index):
        state = object()
        states.append(state)
        return state
    def retire(stage, state):
        if stage == 'delivery':
            raise RuntimeError('native owner unknown')
    stages = BoundedLayerStages((lambda t, s: t, lambda t, s: t,
        lambda t, s: LayerReply(t, 'ready')), initialize=initialize, retire=retire,
        workers=(1, 1, 1))
    submit(stages, 0).result(5)
    with pytest.raises(RuntimeError, match='undrained'):
        stages.close()
    assert stages.snapshot()['retained_states'] == 1 and not stages.closed
    assert stages._retained_states[0] in states


def test_exact_layer_lookahead_uses_staged_future_and_retains_monotonic_order():
    stages = make((lambda t, s: t, lambda t, s: t, lambda t, s: LayerReply(t, t.step)))
    class Callback:
        def __call__(self, ticket):
            raise AssertionError('full-chain executor ran a staged job')
        def submit_layer(self, ticket, *, published, timeout):
            return stages.submit(ticket, ticket, ticket, published=published,
                                 timeout=timeout, cleanup=lambda payload: None)
    pipeline = LayerLookahead('request', 'incarnation', layers=1)
    for step in range(3):
        pipeline.publish(step, 0, Callback())
        assert pipeline.consume(step, 0) == step
    assert pipeline.close() == ()
    stages.close()
    assert len(pipeline.trace) == stages.snapshot()['completed'] == 3
    for row in stages.trace:
        assert [phase['stage'] for phase in row['phases']] == list(stages.names)
        assert all(row['published'] <= phase['queued'] <= phase['start'] <= phase['end'] <= row['terminal']
                   for phase in row['phases'])
