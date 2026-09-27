"""Verified off-owner Q enters the existing logical V search state machine."""

import asyncio
import dataclasses
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch
from sglang.srt.disaggregation.pvd.prediction import QueryVectors
from sglang.srt.disaggregation.pvd.probe_lane_protocol import (
    ProbeLaneReply,
    ProbeLaneTicket,
)
from sglang.srt.disaggregation.pvd.probe_lane_unix import (
    ProbeLaneUnixClient,
    ProbeLaneUnixServer,
)
from sglang.srt.disaggregation.pvd.probe_search import StaleProbeSearch
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED, QueryHeadMapping
from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_probe_search import setup
from test_pvd_prompt_index import shard_client


@pytest.fixture
def socket_dir():
    with tempfile.TemporaryDirectory(prefix="pvd-lane-", dir="/tmp") as name:
        root = Path(name)
        root.chmod(0o700)
        yield root


def _ticket(window, route):
    return ProbeLaneTicket.issue(
        window,
        target_model_id=route.identity.vector_space,
        weights_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        layers=(route.identity.layer,),
        head_start=route.query_head,
        head_count=1,
        head_dim=route.scope.head_dim,
        max_reply_bytes=route.scope.head_dim * 4,
        deadline_monotonic=time.monotonic() + 30,
    )


def _reply(bound, vector):
    query = QueryVectors(
        bound.target_model_id,
        f"{bound.window.prefix.version}:probe:{bound.nonce}",
        bound.layers[0],
        bound.head_start,
        bound.head_count,
        bound.window.query_positions,
        len(bound.window.query_positions),
        torch.tensor(vector, dtype=torch.float32).reshape(
            len(bound.window.query_positions), 1, bound.head_dim
        ),
        prefix_version=bound.window.prefix.version,
        positional_encoding=ROPE_APPLIED,
        request_id=bound.window.prefix.request_id,
    )
    return ProbeLaneReply(
        bound.nonce,
        bound.window.operation_id,
        bound.prefix_digest,
        bound.weights_sha256,
        bound.tokenizer_sha256,
        (query,),
    )


def test_private_lane_q_reaches_real_v_search_without_local_probe(socket_dir):
    async def run():
        _, store, session, window, pipeline, probe, route = setup()
        bound = _ticket(window, route)
        budget = TransferBudget(1 << 20, 1)
        service = await ProbeLaneUnixServer(
            socket_dir,
            "probe.sock",
            expected_client_pid=os.getpid(),
            target_model_id=bound.target_model_id,
            weights_sha256=bound.weights_sha256,
            tokenizer_sha256=bound.tokenizer_sha256,
            handler=lambda accepted: _reply(accepted, pipeline.probe.vector),
        ).start()
        try:
            lane = ProbeLaneUnixClient(
                socket_dir,
                "probe.sock",
                expected_server_pid=os.getpid(),
                reply_budget=budget,
            )
            async with shard_client(store) as http:
                search = PVDShardSearchClient(str(http.make_url("")))
                try:
                    async with lane.request(bound) as queries:
                        prepared = session.prepare_from_lane(
                            window,
                            bound,
                            queries,
                            routes=(route,),
                            head_mapping=QueryHeadMapping(1, 1),
                        )
                        await session.search(prepared, search)
                        selection = session.take_selection(window)
                        assert selection.selections[0].token_ids == (3,)
                        assert selection.queries[0].query_version.endswith(bound.nonce)
                    assert budget.snapshot()["reservations"] == 0
                finally:
                    await search.close()
        finally:
            await service.aclose()
        assert probe.closed == 0  # no local draft or target probe ran

    asyncio.run(run())


def test_private_lane_rejects_foreign_window_before_search():
    _, _, session, window, pipeline, _, route = setup()
    bound = _ticket(window, route)
    other = _ticket(window, route)
    with pytest.raises(StaleProbeSearch, match="foreign"):
        session.prepare_from_lane(
            window,
            dataclasses.replace(other, window=dataclasses.replace(window)),
            _reply(bound, pipeline.probe.vector).queries,
            routes=(route,),
            head_mapping=QueryHeadMapping(1, 1),
        )
    assert session._pending is None
