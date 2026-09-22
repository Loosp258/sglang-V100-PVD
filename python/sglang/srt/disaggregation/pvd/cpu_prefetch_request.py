"""Controlled CPU request loop: shared capture -> shard search -> rank install.

Not production Scheduler wiring. Bootstrap must already have installed the
complete Prompt via the owned CPUInstallGroup. Select either an explicit HTTP
Delivery sink or pack_source(rank, specs), a local reference context manager
over independently version-checked payloads. The local path must own/pin its
source for the scope. Only the Delivery sink publishes write destinations.
"""

import asyncio
from contextlib import nullcontext

from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix
from sglang.srt.disaggregation.pvd.probe_search import (
    ProbeSearchSession,
    StaleProbeSearch,
)
from sglang.srt.disaggregation.pvd.sparse_install import (
    CPUInstallGroup,
    InstallProtocolError,
)
from sglang.srt.disaggregation.pvd.sparse_union import union_query_head_selections


class LatePrefetchStart(ValueError):
    """Decode illegally advanced past an uninstalled refresh boundary."""


class _PrefetchRequestCore:
    def __init__(
        self,
        group,
        pipeline,
        *,
        head_mapping,
        rank_routes,
        max_union_tokens,
        delivery=None,
    ):
        self._validate_group(group)
        state = group.coordinator.snapshot()
        if state["lead_tokens"] <= 0:
            raise ValueError(
                "controlled predictive loop requires a positive lead window"
            )
        if state["state"] != "idle" or state["installed_tokens"] != 0:
            raise InstallProtocolError(
                "complete initial Prompt must be installed first"
            )
        self.group, self.pipeline, self.mapping = group, pipeline, head_mapping
        self._validate_delivery(delivery, group)
        self.delivery = delivery
        self._metadata = group.describe_banks()
        self._routes = {rank: tuple(routes) for rank, routes in rank_routes.items()}
        if set(self._routes) != set(self._metadata) or any(
            type(r) is not int for r in self._routes
        ):
            raise ValueError("route membership must match the bank ranks exactly")
        for rank, routes in self._routes.items():
            if (
                not routes
                or {(r.identity.layer, r.identity.kv_head) for r in routes}
                != self._metadata[rank]["groups"]
            ):
                raise ValueError(
                    "routed layer/KV heads must match their receiving bank"
                )
        if type(max_union_tokens) is not int or max_union_tokens <= 0:
            raise ValueError("explicit positive union limit required")
        self._limit = max_union_tokens
        request, incarnation, entry = group.coordinator.identity
        self._session = self._create_session(request, entry, incarnation=incarnation)
        self._active = self._ready = None
        self._tasks = ()
        self._closed = False

    def _live(self):
        self.group.coordinator._live()
        if self._closed:
            raise StaleProbeSearch("prefetch request is closed")

    async def refresh(
        self,
        prefix,
        *,
        query_positions,
        clients,
        pack_source=None,
        execution_scope=None,
    ):
        """Predict ahead of boundary; a first start AT it uses committed Q.

        Boundary fallback probes explicit positions inside the actual prefix,
        never runs draft, and waits for all installs. Starting past it fails.
        Predictions are CPU synchronous; network searches alone run concurrently.
        """
        self._live()
        if (self.delivery is None and not callable(pack_source)) or (
            self.delivery is not None and pack_source is not None
        ):
            raise ValueError("choose exactly one local source or owned Delivery sink")
        if self._active is not None or self._tasks:
            raise ValueError("one outstanding refresh per request")
        if (
            not isinstance(prefix, CommittedPrefix)
            or prefix.request_id != self._session.request_id
        ):
            raise ValueError("immutable prefix for this request required")
        if set(clients) != set(self._routes) or any(
            type(r) is not int for r in clients
        ):
            raise ValueError("one search client per declared rank required")
        boundary = self.group.coordinator.snapshot()["next_boundary"]
        if prefix.committed_position > boundary:
            raise LatePrefetchStart(
                "Decode advanced past an uninstalled refresh boundary"
            )
        query_source = (
            "committed" if prefix.committed_position == boundary else "predicted"
        )
        epoch = self.group.begin(prefix.committed_position)
        self._active = epoch
        try:
            window = self._session.begin(
                prefix,
                target_tokens=epoch.target_tokens,
                query_positions=query_positions,
                install_epoch=epoch,
                query_source=query_source,
            )
            routes, partitions = [], {}
            for rank in sorted(self._routes):
                start = len(routes)
                routes.extend(self._routes[rank])
                partitions[rank] = tuple(range(start, len(routes)))
            # The optional scheduler-owned scope covers synchronous draft/probe
            # only; it MUST NOT remain held over HTTP waits.
            with nullcontext() if execution_scope is None else execution_scope():
                prepared = self._session.prepare(
                    window,
                    self.pipeline,
                    routes=tuple(routes),
                    head_mapping=self.mapping,
                )
            children = self._session.fork_prepared(prepared, partitions)
            self._tasks = tuple(
                asyncio.create_task(child.search(part, clients[rank]))
                for rank, (child, part) in children.items()
            )
            await asyncio.gather(*self._tasks)
            self._live()
            self.group.coordinator._match(epoch)
            if self._active is not epoch:
                raise StaleProbeSearch("refresh no longer belongs to active request")
            selections = {}
            for rank, (child, _) in children.items():
                selection = child.take_selection(window)
                specs = union_query_head_selections(
                    selection,
                    mapping=self.mapping,
                    layout_fingerprint=self._metadata[rank]["identity"][3],
                    max_union_tokens=self._limit,
                )
                selections[rank] = specs
            if self.delivery is None:
                for rank, specs in selections.items():
                    # Local reference source remains independently version checked.
                    with pack_source(rank, specs) as payloads:
                        self.group.stage(epoch, rank, payloads)
            else:
                self._tasks = tuple(
                    asyncio.create_task(self.delivery.stage(epoch, rank, specs))
                    for rank, specs in selections.items()
                )
                await asyncio.gather(*self._tasks)
                self._live()
                self.group.coordinator._match(epoch)
            self._session.invalidate()
            self._ready = epoch
            return epoch  # ready-to-install only; clock has NOT advanced
        except BaseException:
            self.cancel("probe/search/packing failed or was cancelled")
            raise
        finally:
            tasks, self._tasks = self._tasks, ()
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def can_decode(self, committed_tokens):
        if self._closed:
            return False
        permitted = self.group.can_decode(committed_tokens)
        self._session.observe(committed_tokens)
        return permitted

    def try_install(self, rank_counts):
        self._live()
        if self._ready is None:
            return False
        try:
            if self.delivery is not None:
                self.delivery.require_installable(self._ready)
            done = self.group.try_install(self._ready, rank_counts)
            if done and self.delivery is not None:
                self.delivery.installed(self._ready)
        except BaseException:
            self.cancel("install failed")
            raise
        if done:
            self._active = self._ready = None
        return done

    @property
    def pending_install_boundary(self):
        """The ready round's boundary, even after APPLIED advances the coordinator.

        RESUMED/Delivery finalization may still be pending at the old committed
        token count. The coordinator's next boundary is not this round's count.
        """
        self.group.coordinator._owner()
        return None if self._ready is None else self._ready.target_tokens

    def cancel(self, reason="request cancelled or prefix/Entry replaced"):
        self.group.cancel(reason)
        self._closed = True
        self._session.close()
        if self.delivery is not None:
            self.delivery.cancel()
        self._active = self._ready = None
        for task in self._tasks:
            task.cancel()

    def close(self):
        self.cancel()
        if self.delivery is not None and (
            self.delivery.snapshot()["pending_rounds"]
            or self.delivery.snapshot()["retained_destinations"]
        ):
            raise InstallProtocolError(
                "await aclose() to drain remote Delivery ownership"
            )
        self._require_query_drained()
        self.group.close()  # Group-specific reader/fence policy; never force free.

    def _require_query_drained(self):
        """Device subclasses refuse successful close after UNKNOWN query work."""

    async def aclose(self):
        self.cancel()
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        errors = await self.delivery.close() if self.delivery is not None else {}
        self._require_query_drained()
        self.group.close()
        if errors:
            raise InstallProtocolError(f"remote Delivery ownership remains: {errors}")


class CPUPrefetchRequest(_PrefetchRequestCore):
    def _validate_group(self, group):
        if not isinstance(group, CPUInstallGroup):
            raise TypeError("an exclusively owned CPUInstallGroup is required")

    def _validate_delivery(self, delivery, group):
        from sglang.srt.disaggregation.pvd.cpu_sparse_delivery import CPUSparseDelivery

        if delivery is not None and (
            not isinstance(delivery, CPUSparseDelivery) or delivery.group is not group
        ):
            raise ValueError("Delivery sink must own this exact CPU install group")

    def _create_session(self, *args, **kwargs):
        return ProbeSearchSession(*args, **kwargs)
