"""Request HTTP reuse with real worker threads/I/O, without a CUDA substitute claim."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import pytest

import sglang.srt.disaggregation.pvd.oasis_transport as oasis


class BackgroundIO:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.submissions = 0

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.call_soon(self.ready.set)
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert self.ready.wait(5)

    def submit(self, coroutine):
        self.submissions += 1
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self):
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            assert not self.thread.is_alive()
            self.loop.close()


@pytest.fixture
def background_io():
    control = BackgroundIO()
    yield control
    control.close()


@pytest.fixture
def echo_server():
    received = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                received.append((self.path, body, self.client_address))
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def transport(monkeypatch, control, *, reuse_io=True, url="http://127.0.0.1:1", timeout=5):
    class Registry:
        def __init__(self, *args, **kwargs):
            self.owner_thread = threading.get_ident()
            self._records = {}

    # No native operations run: these fixtures exercise real HTTP sessions,
    # Python thread affinity, request closure and retention of opaque owners.
    monkeypatch.setattr(oasis, "OasisCUDAReceiveRegistry", Registry)
    monkeypatch.setattr(oasis.torch.cuda, "Stream", lambda **kwargs:
        SimpleNamespace(owner_thread=threading.get_ident()))
    manager = SimpleNamespace(control=control, worker_epoch="D-incarnation",
        transfer_budget=object(), sparse_receive_engine=SimpleNamespace(
            health=lambda: {"healthy": True, "session_id": "D-session"}))
    layout = SimpleNamespace(num_layers=28, total_kv_heads=4,
        kv_heads_per_rank=2, head_dim=128, kv_dtype="torch.float16", page_size=4)
    selected = SimpleNamespace(manifest=SimpleNamespace(layout=layout,
        shards=(SimpleNamespace(page_count=1, last_page_valid_tokens=4),)),
        shards=tuple(SimpleNamespace(rank=rank, url=url, rail="rail") for rank in (0, 1)))
    return oasis.OasisLayerTransport(manager, selected, request_id="request",
        incarnation="incarnation", device="cpu", vector_space="target-Q",
        capacity=4, max_new=2, top_k=4, timeout=timeout, reuse_io=reuse_io)


@pytest.mark.parametrize("reuse_io", [False, True])
def test_two_real_workers_reuse_clients_and_sessions_without_reusing_native_owners(
    monkeypatch, background_io, echo_server, reuse_io
):
    url, received = echo_server
    owner = transport(monkeypatch, background_io, reuse_io=reuse_io, url=url)
    barrier = threading.Barrier(2)
    observed = []
    observed_lock = threading.Lock()

    def worker(sequence):
        for layer in range(2):
            state = owner._worker()
            barrier.wait(timeout=5)

            async def requests():
                for rank in (0, 1):
                    payload = dict(request="request", worker=sequence, layer=layer, rank=rank)
                    assert await state["search"][rank]._post_json("/search", payload) == payload
                    assert await state["control"][rank]._request("POST", "/control", payload) == payload
                return tuple(state[kind][rank]._session
                    for kind in ("search", "control") for rank in (0, 1))

            sessions = state["loop"].run_until_complete(requests())
            assert state["registry"].owner_thread == threading.get_ident()
            assert state["stream"].owner_thread == threading.get_ident()
            with observed_lock:
                observed.append((state, sessions))
            owner._retire_worker(state)
            barrier.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, sequence) for sequence in range(2)]
        for future in futures:
            future.result(timeout=10)

    assert len(received) == 16
    assert len({state["owner_thread"] for state, _ in observed}) == 2
    assert len({id(state["registry"]) for state, _ in observed}) == 4
    assert len({id(state["loop"]) for state, _ in observed}) == 4
    assert all(state["loop"].is_closed() for state, _ in observed)
    assert not owner.workers
    expected = 2 if reuse_io else 8
    stats = owner.io_snapshot()
    assert stats["job_count"] == stats["worker_loops_created"] == 4
    assert stats["search_clients_created"] == stats["control_clients_created"] == expected
    assert stats["search_sessions_created"] == stats["control_sessions_created"] == expected
    assert len({id(session) for _, sessions in observed for session in sessions}) == expected * 2
    assert all(session.closed is not reuse_io for _, sessions in observed for session in sessions)
    owner.close()
    assert owner.closed and all(session.closed for _, sessions in observed for session in sessions)
    assert background_io.submissions == int(reuse_io)


def test_cancelled_close_await_keeps_one_real_close_and_owners_until_completion(
    monkeypatch, background_io
):
    owner = transport(monkeypatch, background_io)
    state = owner._worker()
    entered, release = threading.Event(), threading.Event()

    class SlowSession:
        calls = 0

        async def close(self):
            self.calls += 1
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)

    session = SlowSession()
    state["search"][0]._session = session
    owner._retire_worker(state)
    original_cache, clients = owner._cpu_cache, owner._shared_clients

    async def run():
        closing = asyncio.create_task(asyncio.to_thread(owner.close))
        assert await asyncio.to_thread(entered.wait, 5)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not owner.closed and owner._cpu_cache is original_cache
        assert owner._shared_clients is clients
        retry = asyncio.create_task(asyncio.to_thread(owner.close))
        await asyncio.sleep(0.02)
        assert not retry.done() and session.calls == background_io.submissions == 1
        release.set()
        await retry

    try:
        asyncio.run(run())
    finally:
        release.set()
        if owner._shared_close_future is not None:
            owner._shared_close_future.result(timeout=5)
    assert owner.closed and owner._cpu_cache is None and owner._shared_clients is None
    assert session.calls == background_io.submissions == 1


def test_close_timeout_retains_cached_future_clients_and_cache(monkeypatch, background_io):
    owner = transport(monkeypatch, background_io, timeout=0.02)
    state = owner._worker()
    entered, release = threading.Event(), threading.Event()

    class SlowSession:
        calls = 0

        async def close(self):
            self.calls += 1
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)

    session = SlowSession()
    state["search"][0]._session = session
    owner._retire_worker(state)
    original_cache, clients = owner._cpu_cache, owner._shared_clients
    try:
        with pytest.raises(TimeoutError):
            owner.close()
        assert entered.is_set()
        first = owner._shared_close_future
        assert not owner.closed and owner._cpu_cache is original_cache
        assert owner._shared_clients is clients and not first.done()
        with pytest.raises(RuntimeError, match="closing"):
            owner._worker()
        release.set()
        first.result(timeout=5)
        owner.close()
        assert owner._shared_close_future is first
        assert owner.closed and session.calls == background_io.submissions == 1
    finally:
        release.set()
        if owner._shared_close_future is not None:
            owner._shared_close_future.result(timeout=5)


def test_failed_close_does_not_free_or_resubmit_unknown_session(monkeypatch, background_io):
    owner = transport(monkeypatch, background_io)
    state = owner._worker()

    class FailedSession:
        calls = 0

        async def close(self):
            self.calls += 1
            raise RuntimeError("native session retirement unknown")

    session = FailedSession()
    state["search"][0]._session = session
    owner._retire_worker(state)
    original_cache, clients = owner._cpu_cache, owner._shared_clients
    for _ in range(2):
        with pytest.raises(RuntimeError, match="retain request owners"):
            owner.close()
        assert not owner.closed and owner._cpu_cache is original_cache
        assert owner._shared_clients is clients
    assert session.calls == background_io.submissions == 1


@pytest.mark.parametrize("unknown", ["registration", "CUDA"])
def test_native_unknown_keeps_worker_clients_loop_and_request_cache(
    monkeypatch, background_io, unknown
):
    owner = transport(monkeypatch, background_io)
    state = owner._worker()
    native_owner = object()
    if unknown == "registration":
        state["registry"]._records["pending"] = native_owner
    else:
        state["quarantine"] = [native_owner]
    owner._retire_worker(state)
    original_cache, clients = owner._cpu_cache, owner._shared_clients
    with pytest.raises(RuntimeError, match="retain undrained"):
        owner.close()
    assert owner.workers == [state] and not state["loop"].is_closed()
    assert owner._cpu_cache is original_cache and owner._shared_clients is clients
    assert not owner.closed and owner._shared_close_future is None
    assert not any(client._closed for kind in ("search", "control")
        for client in clients[kind].values())
    # The fixture now supplies explicit completion proof for its opaque fake
    # owner. There is no production force-free path or inferred native proof.
    state["registry"]._records.clear()
    state.pop("quarantine", None)
    state["loop"].close()
    owner.workers.remove(state)
    owner.close()


def test_close_on_background_loop_refuses_before_blocking_or_releasing(monkeypatch, background_io):
    owner = transport(monkeypatch, background_io)
    state = owner._worker()
    owner._retire_worker(state)
    original_cache = owner._cpu_cache

    async def wrong_loop():
        with pytest.raises(RuntimeError, match="cannot block"):
            owner.close()

    background_io.submit(wrong_loop()).result(timeout=5)
    assert not owner.closed and not owner._closing and owner._cpu_cache is original_cache
    assert owner._shared_close_future is None
    owner.close()


def test_worker_retirement_rejects_foreign_thread(monkeypatch, background_io):
    owner = transport(monkeypatch, background_io)
    state = owner._worker()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner._retire_worker, state)
        with pytest.raises(RuntimeError, match="worker thread"):
            future.result(timeout=5)
    assert owner.workers == [state] and not state["loop"].is_closed()
    owner._retire_worker(state)
    owner.close()


def test_stopped_io_loop_cannot_release_request_owners(monkeypatch, background_io):
    owner = transport(monkeypatch, background_io)
    state = owner._worker()
    owner._retire_worker(state)
    original_cache, clients = owner._cpu_cache, owner._shared_clients
    background_io.close()
    with pytest.raises(RuntimeError, match="I/O loop stopped"):
        owner.close()
    assert not owner.closed and owner._closing
    assert owner._cpu_cache is original_cache and owner._shared_clients is clients
    assert owner._shared_close_future is None


def test_serving_owner_keeps_reservation_on_transport_retirement_error(
    monkeypatch, background_io
):
    from sglang.srt.disaggregation.pvd.oasis_startup import OasisServingOwner
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    layer_transport = transport(monkeypatch, background_io)
    state = layer_transport._worker()
    state["registry"]._records["unknown-write"] = object()
    layer_transport._retire_worker(state)
    budget = TransferBudget(1048576, 1)
    reservation = "request-reservation"
    budget.reserve(reservation, 4096, 1)
    resources = SimpleNamespace(budget=budget, owners=[])
    owner = OasisServingOwner("request", "incarnation",
        decoder=SimpleNamespace(layers=1, generated=[]), initial_banks=[object()],
        predict_one=lambda *args: 1, fetch_layer=layer_transport.job,
        current_token=1, position=1, max_steps=1,
        transport=layer_transport, resources=resources, reservation=reservation)
    resources.owners.append(owner)
    original_cache = layer_transport._cpu_cache
    errors = owner.close()
    assert len(errors) == 1 and "retain undrained" in str(errors[0])
    assert owner.close() is errors
    assert not owner._retired and resources.owners == [owner]
    assert budget.snapshot()["used_staging_bytes"] == 4096
    assert layer_transport._cpu_cache is original_cache
    # Explicit fixture-only terminal proof permits transport cleanup. The
    # serving request deliberately retains its failed retirement reservation.
    state["registry"]._records.clear()
    state["loop"].close()
    layer_transport.workers.remove(state)
    layer_transport.close()


def test_reuse_requires_running_manager_loop_and_strict_bool(monkeypatch, background_io):
    with pytest.raises(ValueError, match="routes required"):
        transport(monkeypatch, background_io, reuse_io=1)
    stopped = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError, match="running I/O loop"):
            transport(monkeypatch, SimpleNamespace(loop=stopped))
    finally:
        stopped.close()


@pytest.mark.parametrize("reuse_io", [None, False, True, 0])
@pytest.mark.parametrize("option", ["reuse_io", "sort_missing_tokens", "batched_bank_install", "batched_cache_install"])
def test_json_option_is_optional_and_requires_an_actual_bool(monkeypatch, tmp_path, reuse_io, option):
    import sglang.srt.disaggregation.pvd.oasis_startup as startup

    config = dict(eagle_source="source", eagle_checkpoint="checkpoint",
        eagle_manifest="manifest", vector_space="target-Q", capacity=4,
        max_new=2, top_k=4, workers=2, timeout_seconds=5,
        max_sequence_tokens=64, max_decode_steps=16, request_budget_bytes=1024,
        request_scratch_bytes=32 << 20, bootstrap_budget_bytes=1024,
        bootstrap_transient_bytes=1024, overlap=True)
    if reuse_io is not None:
        config[option] = reuse_io
    path = tmp_path / "oasis.json"
    path.write_text(json.dumps(config))
    scheduler = SimpleNamespace(server_args=SimpleNamespace(pvd_oasis_config=str(path)),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
            model_config=SimpleNamespace(context_len=128))))
    monkeypatch.setattr(startup, "OasisSchedulerBinding", lambda *args: SimpleNamespace())
    monkeypatch.setattr(startup, "OasisResources", lambda scheduler, cfg:
        SimpleNamespace(config=cfg, prepare=lambda *args: None))
    if type(reuse_io) is not bool and reuse_io is not None:
        with pytest.raises(ValueError, match="unsupported Oasis bounds"):
            startup.maybe_install_oasis(scheduler)
    else:
        startup.maybe_install_oasis(scheduler)
        assert scheduler.pvd_oasis_resources.config[option] is (reuse_io is True)
