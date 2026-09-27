"""CPU process fixture for sidecar owner tests; never loads a model."""

import argparse
import json
import os
import signal
import socket
import threading
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--target-model-id", required=True)
parser.add_argument("--socket-dir", required=True)
parser.add_argument("--expected-d-pid", type=int, required=True)
args = parser.parse_args()
mode = os.environ.get("PVD_FAKE_SIDECAR_MODE", "normal")
if mode == "exit":
    raise SystemExit(7)
if args.expected_d_pid != os.getppid():
    raise SystemExit(8)
path = Path(args.socket_dir) / "probe.sock"
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
sock = socket.socket(socket.AF_UNIX)
sock.bind(str(path))
sock.listen(1)
print(
    json.dumps(
        {
            "schema": "pvd.probe.sidecar.ready.v1",
            "pid": os.getpid() + (1 if mode == "wrong_pid" else 0),
            "socket": str(path),
            "weights_sha256": ("c" if mode == "wrong_hash" else "a") * 64,
            "tokenizer_sha256": "b" * 64,
            "device": (
                "cuda:0"
                if mode != "assert_cuda_env"
                or os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
                else "cuda:wrong"
            ),
        }
    ),
    flush=True,
)
if mode == "diagnostic":
    print("PVD probe lane reject reason=reply_capacity count=1", flush=True)
stop.wait()
sock.close()
path.unlink()
