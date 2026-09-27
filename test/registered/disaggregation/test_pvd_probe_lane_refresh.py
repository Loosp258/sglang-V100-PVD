"""Private Q lane substitutes capture, not V search or refresh ownership."""

import asyncio
import dataclasses
import os
import tempfile
import time
from pathlib import Path

from pvd_controlled_prefetch import ControlledFixture
from sglang.srt.disaggregation.pvd.probe_lane_identity import (
    ProbeLaneCheckpointIdentity,
)
from sglang.srt.disaggregation.pvd.probe_lane_protocol import ProbeLaneReply
from sglang.srt.disaggregation.pvd.probe_lane_unix import (
    ProbeLaneUnixClient,
    ProbeLaneUnixServer,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_controlled_prefetch import components


def test_private_q_lane_runs_existing_v_search_and_sparse_install(caplog):
    caplog.set_level(
        "INFO", logger="sglang.srt.disaggregation.pvd.cpu_prefetch_request"
    )

    async def run():
        fixture = ControlledFixture(*components())
        prefix = fixture.refresh_prefix(3)
        checkpoint = ProbeLaneCheckpointIdentity("a" * 64, "b" * 64, 100, 10)
        called = []

        def handler(bound):
            called.append(bound.nonce)
            source = fixture.request.pipeline.probe._queries(
                bound.window.prefix, bound.window.query_positions
            )
            queries = tuple(
                dataclasses.replace(
                    query,
                    version=f"{bound.window.prefix.version}:probe:{bound.nonce}",
                )
                for query in source
            )
            return ProbeLaneReply(
                bound.nonce,
                bound.window.operation_id,
                bound.prefix_digest,
                bound.weights_sha256,
                bound.tokenizer_sha256,
                queries,
            )

        with tempfile.TemporaryDirectory(prefix="pvd-lane-", dir="/tmp") as name:
            directory = Path(name)
            directory.chmod(0o700)
            server_budget = TransferBudget(1 << 20, 1)
            client_budget = TransferBudget(1 << 20, 1)
            service = await ProbeLaneUnixServer(
                directory,
                "probe.sock",
                expected_client_pid=os.getpid(),
                target_model_id="target",
                weights_sha256=checkpoint.weights_sha256,
                tokenizer_sha256=checkpoint.tokenizer_sha256,
                handler=handler,
                reply_budget=server_budget,
            ).start()
            lane = ProbeLaneUnixClient(
                directory,
                "probe.sock",
                expected_server_pid=os.getpid(),
                reply_budget=client_budget,
            )
            try:
                async with fixture.clients() as clients:
                    epoch = await fixture.request.refresh(
                        prefix,
                        query_positions=(len(prefix.tokens),),
                        clients=clients,
                        pack_source=fixture.pack_source,
                        lane_client=lane,
                        lane_checkpoint=checkpoint,
                        lane_deadline_monotonic=time.monotonic() + 10,
                    )
                    assert epoch.target_tokens == 4
                    assert fixture.request.try_install({0: 4, 1: 4})
                    assert len(called) == 1
                    assert fixture.request.pipeline.provider.calls == []
                    assert fixture.request.pipeline.probe.calls == []
                    assert len(fixture.packed_specs) == 4
                    assert client_budget.snapshot()["reservations"] == 0
                    assert server_budget.snapshot()["reservations"] == 0
            finally:
                await service.aclose()
                fixture.close()

    asyncio.run(run())
    assert any(
        "query_source=predicted probe_source=private_lane" in record.message
        for record in caplog.records
    )
