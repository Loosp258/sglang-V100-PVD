"""CPU contract tests for the three-node native-GPU relay acceptance tool."""

import importlib.util
import io
import json
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch


_SPEC = importlib.util.spec_from_file_location(
    "run_pvd_native_relay", Path(__file__).with_name("run_pvd_native_relay.py")
)
relay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(relay)


class _Descriptor:
    def __init__(self, endpoint, rail="mlx5_0", length=4096):
        self.endpoint, self.rail, self.length = endpoint, rail, length

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)

    def to_dict(self):
        return dict(endpoint=self.endpoint, rail=self.rail, length=self.length)


class _Stream:
    def __init__(self, messages):
        self.incoming = deque(json.dumps(value) + "\n" for value in messages)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def readline(self):
        return self.incoming.popleft() if self.incoming else ""

    def write(self, line):
        self.sent.append(json.loads(line))

    def flush(self):
        pass


class _Connection:
    def __init__(self, peer, stream):
        self.peer, self.stream = peer, stream

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getpeername(self):
        return self.peer, 12345

    def settimeout(self, _):
        pass

    def makefile(self, *_args, **_kwargs):
        return self.stream


class _Listener:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def accept(self):
        return self.connection, (self.connection.peer, 12345)


class _Engine:
    def __init__(self, status):
        self.status = status
        self.registered = []
        self.submitted = []
        self.released = []

    def register_memory(self, tensor, **_kwargs):
        registration = types.SimpleNamespace(
            tensor=tensor, descriptor=_Descriptor("10.0.1.2:12345")
        )
        self.registered.append(registration)
        return registration

    def submit_put(self, source, remote):
        self.submitted.append((source, remote))
        return types.SimpleNamespace(
            transport_state=types.SimpleNamespace(is_locally_safe_to_release=True),
            error=None,
        )

    def poll(self, _handle):
        return self.status

    def release_memory(self, registration):
        self.released.append(registration)


class _Statuses:
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class NativeRelayTests(unittest.TestCase):
    def setUp(self):
        self.args = relay._arguments([
            "V", "--p-ip", "10.0.1.1", "--v-ip", "10.0.1.2",
            "--d-ip", "10.0.1.3", "--rail", "mlx5_0",
            "--timeout-seconds", "0.02",
        ])

    def test_distinct_private_ip_and_bounded_arguments(self):
        for extra in (
            ["--d-ip", "10.0.1.1"],
            ["--d-ip", "8.8.8.8"],
            ["--p-to-v-port", "0"],
            ["--v-to-d-port", "28175"],
            ["--length", "1048577"],
            ["--timeout-seconds", "0"],
            ["--gpu-id", "-1"],
        ):
            with self.subTest(extra=extra), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    relay._arguments([
                        "V", "--p-ip", "10.0.1.1", "--v-ip", "10.0.1.2",
                        "--d-ip", "10.0.1.3", "--rail", "mlx5_0",
                        *extra,
                    ])

    def test_descriptor_must_match_peer_rail_and_length(self):
        expected = dict(endpoint="10.0.1.3:15640", rail="mlx5_0", length=4096)
        self.assertEqual(
            relay._require_descriptor(
                expected, _Descriptor, endpoint="10.0.1.3",
                rail="mlx5_0", length=4096,
            ).endpoint,
            expected["endpoint"],
        )
        for change in (
            dict(endpoint="10.0.1.2:15640"), dict(rail="mlx5_1"),
            dict(length=4095),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                relay._require_descriptor(
                    expected | change, _Descriptor, endpoint="10.0.1.3",
                    rail="mlx5_0", length=4096,
                )

    def test_status_alone_does_not_prove_safe_release(self):
        engine = _Engine(_Statuses.SUCCESS)
        handle = types.SimpleNamespace(
            transport_state=types.SimpleNamespace(is_locally_safe_to_release=False)
        )
        with self.assertRaisesRegex(RuntimeError, "safe-to-release"):
            relay._terminal(engine, handle, _Statuses, 0.01)

    def test_p_releases_only_after_terminal_proof(self):
        p_stream = _Stream([
            _Descriptor("10.0.1.2:15640").to_dict(), {"bytes_equal": True},
        ])
        connection = _Connection("10.0.1.2", p_stream)
        engine = _Engine(_Statuses.SUCCESS)
        with (
            patch.object(relay.socket, "create_connection", return_value=connection),
            patch.object(relay, "_pattern", return_value=object()),
        ):
            relay._run_p(
                self.args, None, engine, _Descriptor,
                lambda registration, offset, length: (registration, offset, length),
                _Statuses,
            )
        self.assertEqual(engine.released, engine.registered)
        self.assertIs(engine.submitted[0][0][0], engine.registered[0])
        self.assertTrue(p_stream.sent[-1]["terminal"])

    def test_d_keeps_mr_without_sender_terminal_message(self):
        stream = _Stream([{"terminal": False, "success": False}])
        connection = _Connection("10.0.1.2", stream)
        engine = _Engine(_Statuses.SUCCESS)
        torch = types.SimpleNamespace(
            uint8="uint8", zeros=lambda *_a, **_k: object(),
            cuda=types.SimpleNamespace(synchronize=lambda _device: None),
        )
        with (
            patch.object(relay, "_listener", return_value=_Listener(connection)),
            patch("sys.stdout", new=io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "terminal WRITE"):
                relay._run_d(self.args, torch, engine)
        self.assertEqual(engine.released, [])

    def _run_v(self, status="success", p_outcome=None):
        d_stream = _Stream([
            _Descriptor("10.0.1.3:15640").to_dict(), {"bytes_equal": True},
        ])
        p_stream = _Stream([p_outcome or {"terminal": True, "success": True}])
        d_connection = _Connection("10.0.1.3", d_stream)
        p_connection = _Connection("10.0.1.1", p_stream)
        engine = _Engine(status)
        self.last_engine = engine
        tensor = object()
        torch = types.SimpleNamespace(
            uint8="uint8", zeros=lambda *_a, **_k: tensor,
            equal=lambda a, b: a is b,
            cuda=types.SimpleNamespace(synchronize=lambda _device: None),
        )
        with (
            patch.object(relay.socket, "create_connection", return_value=d_connection),
            patch.object(relay, "_listener", return_value=_Listener(p_connection)),
            patch.object(relay, "_pattern", return_value=tensor),
            patch("sys.stdout", new=io.StringIO()),
        ):
            relay._run_v(
                self.args, torch, engine, _Descriptor,
                lambda registration, offset, length: (registration, offset, length),
                _Statuses,
            )
        return engine, d_stream, p_stream

    def test_v_reuses_its_receive_registration_as_send_source(self):
        engine, d_stream, p_stream = self._run_v()
        self.assertEqual(len(engine.registered), 1)
        self.assertEqual(len(engine.submitted), 1)
        self.assertIs(engine.submitted[0][0][0], engine.registered[0])
        self.assertEqual(engine.released, engine.registered)
        self.assertEqual(p_stream.sent[-1], {"bytes_equal": True})
        self.assertTrue(d_stream.sent[-1]["terminal"])

    def test_unknown_second_write_retains_v_registration(self):
        with self.assertRaisesRegex(RuntimeError, "completion is unknown"):
            self._run_v(status="pending")
        self.assertEqual(len(self.last_engine.submitted), 1)
        self.assertEqual(self.last_engine.released, [])

    def test_bad_first_terminal_never_starts_second_write(self):
        with self.assertRaisesRegex(RuntimeError, "P has not proved"):
            self._run_v(p_outcome={"terminal": False, "success": False})
        self.assertEqual(self.last_engine.submitted, [])
        self.assertEqual(self.last_engine.released, [])


if __name__ == "__main__":
    unittest.main()
