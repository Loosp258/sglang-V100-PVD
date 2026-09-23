"""Fail-closed completion proof for the native cross-node validation runner."""

from types import SimpleNamespace

import pytest
from run_pvd_native_cross_node import _wait_terminal
from sglang.srt.disaggregation.pvd.transfer_engine import TransferStatus
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransportState


class Poller:
    def __init__(self, *statuses):
        self.statuses = iter(statuses)

    def poll(self, _handle):
        return next(self.statuses)


@pytest.mark.parametrize(
    ("status", "state"),
    [
        (TransferStatus.SUCCESS, TransportState.TERMINAL_SUCCESS),
        (TransferStatus.FAILED, TransportState.TERMINAL_FAILED),
        (TransferStatus.FAILED, TransportState.NOT_SUBMITTED),
    ],
)
def test_only_proved_completion_can_release_memory(status, state):
    handle = SimpleNamespace(transport_state=state)
    assert _wait_terminal(Poller(status), handle, TransferStatus, 1) == status


@pytest.mark.parametrize(
    "state", [TransportState.UNKNOWN, TransportState.IN_FLIGHT, TransportState.DRAINING]
)
def test_failed_status_without_terminal_proof_is_rejected(state):
    handle = SimpleNamespace(transport_state=state)
    with pytest.raises(RuntimeError, match="retain source MR"):
        _wait_terminal(Poller(TransferStatus.FAILED), handle, TransferStatus, 1)


def test_pending_then_terminal_success():
    handle = SimpleNamespace(transport_state=TransportState.IN_FLIGHT)

    class FinishingPoller:
        calls = 0

        def poll(self, current):
            self.calls += 1
            if self.calls == 1:
                return TransferStatus.PENDING
            current.transport_state = TransportState.TERMINAL_SUCCESS
            return TransferStatus.SUCCESS

    assert (
        _wait_terminal(FinishingPoller(), handle, TransferStatus, 1)
        == TransferStatus.SUCCESS
    )


def test_pending_timeout_is_rejected():
    handle = SimpleNamespace(transport_state=TransportState.IN_FLIGHT)
    with pytest.raises(RuntimeError, match="completion is unknown"):
        _wait_terminal(Poller(TransferStatus.PENDING), handle, TransferStatus, 0)
