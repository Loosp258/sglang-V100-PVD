"""Deploy isolated V/D delivery changes and preserve the complete gate evidence.

Run under WSL, where the configured CloudLab SSH key is available. CPU tests
hide CUDA. Slots may additionally name an explicit project-local native probe;
that hook runs on D only after the CPU regression passes.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile


ROLES = {
    "v": {
        "node": 1,
        "alias": "clgpu021.clemson.cloudlab.us",
        "ip": "130.127.134.35",
        "checkout": "/proj/llm-course-PG0/Yizhzhu-node1-sglang-pvd/validation/pvd-oasis-v-search-20261002",
    },
    "d": {
        "node": 2,
        "alias": "clgpu019.clemson.cloudlab.us",
        "ip": "130.127.134.33",
        "checkout": "/proj/llm-course-PG0/Yizhzhu-node2-sglang-pvd/validation/pvd-oasis-alignment-20261002",
    },
}
SOURCE_MODULES = (
    "control_server.py", "sparse_receiver.py", "cuda_sparse_receiver.py",
    "oasis_transport.py", "oasis_startup.py", "oasis_receive_slots.py",
)
CPU_TESTS = (
    "oasis_attention", "oasis_request", "oasis_pipeline", "oasis_serving",
    "oasis_transport_io", "sparse_receiver", "cuda_sparse_receiver",
    "combined_delivery", "control_background_io", "search_client",
    "sparse_store_delivery", "sparse_delivery", "sparse_payload", "sparse_copy",
    "absent_write_fence", "oasis_receive_slots", "prompt_index", "prompt_chunks",
    "vector_lifecycle", "transfer_authorization", "core",
)
# Other tests import these fixture modules directly from their directory.
FIXTURE_TESTS = ("prompt_vectors", "cpu_sparse_delivery", "sparse_working_set")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mode", required=True, choices=("combine", "slots"))
    parser.add_argument("--native-probe", help="optional project-relative Python probe, slots only")
    parser.add_argument("--native-arg", action="append", default=[],
                        help="one native probe argument; use --native-arg=--option")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.tag):
        parser.error("fresh lowercase filename-safe tag required")
    if args.native_probe and args.mode != "slots":
        parser.error("native probe hook is available only with --mode slots")
    if args.native_arg and not args.native_probe:
        parser.error("--native-arg requires --native-probe")
    return args


def gpu_empty(data):
    rows = data.decode("utf-8").splitlines()
    return len(rows) == 2 and all(
        int(row.split(",", 1)[1].split()[0]) == 0 for row in rows
    )


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / "artifacts" / args.tag).resolve()
    if not output.is_relative_to((root / "artifacts").resolve()):
        raise RuntimeError("output must remain in checkout artifacts")
    source_names = ["python/sglang/srt/disaggregation/pvd/" + n for n in SOURCE_MODULES]
    required_tests = list(CPU_TESTS)
    if args.mode == "combine":
        # Stage 1 can run before the independent slot implementation exists.
        required_tests.remove("oasis_receive_slots")
        optional_tests = ["oasis_receive_slots"]
    else:
        optional_tests = []
    test_names = ["test/registered/disaggregation/test_pvd_" + n + ".py"
                  for n in required_tests + list(FIXTURE_TESTS)]
    for name in optional_tests:
        path = "test/registered/disaggregation/test_pvd_" + name + ".py"
        if (root / path).is_file():
            test_names.append(path)
            required_tests.append(name)
    names = source_names + test_names
    if args.mode == "combine" and not (root / source_names[-1]).is_file():
        names.remove(source_names[-1])
    if args.native_probe:
        names.append(args.native_probe)
    names = list(dict.fromkeys(names))
    payloads = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("missing or non-project deployment source: " + name)
        data = path.read_bytes().replace(b"\r\n", b"\n")
        data.decode("utf-8")
        payloads[name] = data
    output.mkdir(parents=True, exist_ok=False)
    hashes = {name: hashlib.sha256(data).hexdigest()
              for name, data in payloads.items()}
    (output / "local_source_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "configuration.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8", newline="\n")
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for name, data in payloads.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    bundle = archive_bytes.getvalue()
    (output / "deployed.tar.gz").write_bytes(bundle)
    sequence = {role: 0 for role in ROLES}

    def call(role, label, command, *, data=None, check=True, timeout=180):
        sequence[role] += 1
        prefix = output / f"{role}_{sequence[role]:02d}_{label}"
        info = ROLES[role]
        ssh = ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o",
               "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "-o",
               "ServerAliveCountMax=3", "-o", "HostKeyAlias=" + info["alias"],
               "-i", "/home/loosp/.ssh/cloudlab_pub_wsl", "Yizhzhu@" + info["ip"]]
        prefix.with_suffix(".command").write_text(command + "\n", encoding="utf-8", newline="\n")
        try:
            result = subprocess.run(ssh + [command], input=data,
                                    capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            prefix.with_suffix(".stdout").write_bytes(exc.stdout or b"")
            prefix.with_suffix(".stderr").write_bytes(exc.stderr or b"")
            prefix.with_suffix(".status.json").write_text(
                json.dumps({"timed_out": True, "timeout_seconds": timeout}) + "\n",
                encoding="utf-8", newline="\n")
            raise
        prefix.with_suffix(".stdout").write_bytes(result.stdout)
        prefix.with_suffix(".stderr").write_bytes(result.stderr)
        prefix.with_suffix(".status.json").write_text(
            json.dumps({"exit_code": result.returncode}) + "\n",
            encoding="utf-8", newline="\n")
        if check and result.returncode:
            raise RuntimeError(f"{role}/{label} failed; full command/output retained at {prefix}")
        return result

    remote_outputs = {}
    # Check both roles before changing either checkout.
    for role in ROLES:
        initial = call(role, "initial_gpu", "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader")
        (output / (role + "_initial_gpu.txt")).write_bytes(initial.stdout)
        if not gpu_empty(initial.stdout):
            raise RuntimeError(role + " GPUs occupied; no source deployment started")
    try:
        for role, info in ROLES.items():
            remote = info["checkout"]
            remote_output = remote + "/artifacts/" + args.tag
            remote_outputs[role] = remote_output
            create = (
                "from pathlib import Path; "
                f"root=Path({remote!r}).resolve(); "
                f"out=(root/'artifacts'/{args.tag!r}).resolve(); "
                "assert out.is_relative_to(root/'artifacts'); "
                "out.mkdir(parents=True,exist_ok=False); "
                "(out/'tmp').mkdir(); (out/'cache').mkdir(); print(out)"
            )
            call(role, "create_artifacts", "python3 -c " + shlex.quote(create))
            call(role, "deploy", "tar -xzf - -C " + shlex.quote(remote), data=bundle)
            proof = (
                "import hashlib,json; from pathlib import Path; "
                f"root=Path({remote!r}); names={names!r}; "
                "print(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() "
                "for name in names},sort_keys=True))"
            )
            actual = json.loads(call(role, "hashes", "python3 -c " + shlex.quote(proof)).stdout)
            (output / (role + "_source_hashes.json")).write_text(
                json.dumps(actual, indent=2) + "\n", encoding="utf-8", newline="\n")
            if actual != hashes:
                raise RuntimeError(role + " deployed source hashes differ")
            head = call(role, "remote_head", "git -C " + shlex.quote(remote) + " rev-parse HEAD")
            (output / (role + "_remote_head.txt")).write_bytes(head.stdout)

        info = ROLES["d"]
        remote, remote_output = info["checkout"], remote_outputs["d"]
        env = (
            "set -e; source /users/Yizhzhu/.sglang-v100-pvd-env.sh; "
            "export PYTHONPATH=" + shlex.quote(remote + "/python") + "; "
            "export PYTHONDONTWRITEBYTECODE=1; "
            "export TMPDIR=" + shlex.quote(remote_output + "/tmp") + "; "
            "export TMP=$TMPDIR TEMP=$TMPDIR; "
            "export XDG_CACHE_HOME=" + shlex.quote(remote_output + "/cache") + "; "
            "export TRITON_CACHE_DIR=" + shlex.quote(remote_output + "/cache/triton") + "; "
            "export TORCH_EXTENSIONS_DIR=" + shlex.quote(remote_output + "/cache/torch_extensions") + "; "
            "cd " + shlex.quote(remote) + "; "
        )

        def gate(label, command, *, cuda="", timeout=240):
            filename = remote_output + "/" + label + ".txt"
            full = env + "export CUDA_VISIBLE_DEVICES=" + shlex.quote(cuda) + "; " + command
            wrapper = full + " > " + shlex.quote(filename) + " 2>&1"
            (output / (label + ".command")).write_text(wrapper + "\n", encoding="utf-8", newline="\n")
            result = call("d", label, "bash -c " + shlex.quote(wrapper), check=False, timeout=timeout)
            # Compression bounds transfer time without discarding raw output.
            raw = gzip.decompress(call("d", label + "_collect", "gzip -c -- " + shlex.quote(filename)).stdout)
            (output / (label + ".txt")).write_bytes(raw)
            (output / (label + "_status.json")).write_text(
                json.dumps({"exit_code": result.returncode}) + "\n", encoding="utf-8", newline="\n")
            print(label, "exit", result.returncode, "bytes", len(raw), flush=True)
            if result.returncode:
                raise RuntimeError(label + " failed; complete output preserved")

        unit_paths = ["test/registered/disaggregation/test_pvd_" + n + ".py"
                      for n in required_tests]
        pytest_args = ["-q", "-p", "no:cacheprovider", "--basetemp",
                       remote_output + "/pytest"] + unit_paths
        unit_code = (
            "import sys,types; "
            "package=types.ModuleType('sglang'); "
            f"package.__path__=[{(remote + '/python/sglang')!r}]; "
            "sys.modules['sglang']=package; import pytest; "
            f"raise SystemExit(pytest.main({pytest_args!r}))"
        )
        gate("unit", '"$CONDA_PREFIX/bin/python" -B -c ' + shlex.quote(unit_code))
        if args.native_probe:
            gate("native", '"$CONDA_PREFIX/bin/python" -B '
                 + shlex.quote(args.native_probe) + " "
                 + " ".join(shlex.quote(arg) for arg in args.native_arg),
                 cuda="0,1", timeout=300)
    finally:
        remaining = []
        for role in ROLES:
            final = call(role, "final_gpu", "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader")
            (output / (role + "_final_gpu.txt")).write_bytes(final.stdout)
            if not gpu_empty(final.stdout):
                remaining.append(role)
        if remaining:
            raise RuntimeError("gate GPU ownership remains on " + ",".join(remaining))
    print("delivery", args.mode, "gates passed; V and D GPUs empty", flush=True)


if __name__ == "__main__":
    main()
