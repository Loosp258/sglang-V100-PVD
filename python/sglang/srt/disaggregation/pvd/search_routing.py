"""One D compute rank can query several already-selected V storage shards.

This routes logical queries only. It does not choose a V group, grant writes,
merge cross-head scores, or claim that sparse multi-source delivery is installed.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
from sglang.srt.disaggregation.pvd.prompt_vectors import ROPE_APPLIED
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature
from sglang.srt.disaggregation.pvd.search_client import (
    PVDShardSearchClient,
    SearchScope,
    ShardSearchError,
)
from sglang.srt.disaggregation.pvd.sharding import source_shard_intersections
from sglang.srt.disaggregation.pvd.sparse_delivery import (
    MAX_GROUPS,
    SparseDeliveryManifest,
)
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec


@dataclass(frozen=True)
class SourceSparseSelection:
    """V wire manifest plus the exact corresponding D-local logical specs.

    Two layout identities are deliberate: V validates its storage layout;
    D validates its compute bank. This plan authorizes neither a WRITE nor
    installation. Completion/ownership must still be proved for every source.
    """

    storage_rank: int
    manifest: SparseDeliveryManifest
    decode_specs: tuple[SparseKVSpec, ...]


class RoutedShardSearchClient:
    """Immutable rank/head routing over caller-owned HTTP shard clients.

    Every source retains its own index/mapping version namespace. The same V
    source still uses one version across all queries in a probe window. Identity
    comes from trusted Entry/compute layouts, never from a response's routing hint.
    """

    def __init__(
        self,
        *,
        storage_layout,
        compute_layout,
        compute_rank,
        entry_transfer_id,
        prompt_tokens,
        vector_space,
        metric,
        clients,
    ):
        if not isinstance(storage_layout, KVLayoutSignature) or not isinstance(
            compute_layout, KVLayoutSignature
        ):
            raise ValueError("explicit storage and compute layouts required")
        for value in (entry_transfer_id, vector_space):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("explicit Entry and vector-space identities required")
        if storage_layout.pp_size != 1:
            raise ValueError("routed search currently requires PP1")
        intersections = source_shard_intersections(
            storage_layout, compute_layout, compute_rank
        )
        required = {part.storage_rank for part in intersections}
        if set(clients) != required or any(
            type(rank) is not int or not isinstance(client, PVDShardSearchClient)
            for rank, client in clients.items()
        ):
            raise ValueError(
                "exact selected V shard clients required for D head interval"
            )
        if len({client.base_url for client in clients.values()}) != len(clients):
            raise ValueError(
                "different V shards require distinct selected shard endpoints"
            )
        self.entry_transfer_id, self.vector_space = entry_transfer_id, vector_space
        self.scope = SearchScope(
            prompt_tokens, storage_layout.page_size, storage_layout.head_dim, metric
        )
        self.compute_rank = compute_rank
        self.storage_fingerprint = storage_layout.fingerprint
        self.compute_fingerprint = compute_layout.fingerprint
        self.dtype = storage_layout.kv_dtype
        groups = {}
        for part in intersections:
            start = (
                compute_rank * compute_layout.kv_heads_per_rank
                + part.compute_head_offset
            )
            for head in range(start, start + part.head_count):
                for layer in range(storage_layout.num_layers):
                    groups[layer, head] = part.storage_rank
        self.groups = MappingProxyType(groups)
        self.clients = MappingProxyType(dict(clients))
        self._endpoints = {rank: client.base_url for rank, client in clients.items()}
        self._closed = False

    def version_scope(self, identity):
        """Trusted source rank, used to pin versions BEFORE each HTTP query."""
        if self._closed:
            raise ShardSearchError("routed search client is closed")
        if (
            not isinstance(identity, SearchRequestIdentity)
            or identity.entry_transfer_id != self.entry_transfer_id
            or identity.vector_space != self.vector_space
            or identity.positional_encoding != ROPE_APPLIED
            or type(identity.layer) is not int
            or type(identity.kv_head) is not int
            or (identity.layer, identity.kv_head) not in self.groups
        ):
            raise ValueError(
                "search identity is outside the bound D/Entry/vector-space scope"
            )
        rank = self.groups[identity.layer, identity.kv_head]
        if self.clients[rank].base_url != self._endpoints[rank]:
            raise ValueError("selected V shard endpoint changed")
        return rank

    async def search(self, identity, *, queries, top_k, scope):
        rank = self.version_scope(identity)
        if scope != self.scope:
            raise ValueError("search scope differs from trusted Entry layout")
        return await self.clients[rank].search(
            identity, queries=queries, top_k=top_k, scope=scope
        )

    def partition_specs(self, specs):
        """Split a complete D-bank selection into independently versioned V PUTs."""
        if self._closed:
            raise ShardSearchError("routed search client is closed")
        if not isinstance(specs, tuple) or not 0 < len(specs) <= MAX_GROUPS:
            raise ValueError("bounded complete tuple of sparse specs required")
        groups, contexts, parts = set(), set(), {}
        for spec in specs:
            if not isinstance(spec, SparseKVSpec):
                raise ValueError("explicit sparse spec required")
            group = (spec.layer, spec.kv_head)
            if (
                spec.entry_transfer_id != self.entry_transfer_id
                or spec.layout_fingerprint != self.compute_fingerprint
                or group not in self.groups
                or group in groups
                or any(token >= self.scope.prompt_tokens for token in spec.token_ids)
            ):
                raise ValueError("sparse spec is outside the bound D bank")
            groups.add(group)
            contexts.add(
                (
                    spec.request_id,
                    spec.incarnation,
                    spec.operation_id,
                    spec.target_tokens,
                )
            )
            parts.setdefault(self.groups[group], []).append(spec)
        if groups != set(self.groups) or len(contexts) != 1:
            raise ValueError("complete D bank in exactly one refresh epoch required")
        plans = []
        for rank, items in sorted(parts.items()):
            items = tuple(sorted(items, key=lambda spec: (spec.layer, spec.kv_head)))
            manifest = SparseDeliveryManifest(
                tuple(
                    replace(spec, layout_fingerprint=self.storage_fingerprint)
                    for spec in items
                ),
                self.dtype,
                self.scope.head_dim,
            )
            # Manifest validates one version pair per V source, never a fake
            # shared version across different V shards.
            plans.append(SourceSparseSelection(rank, manifest, items))
        return tuple(plans)

    async def close(self):
        # Borrowed clients may also serve another D/request. Their owner closes
        # them only after all windows using them have drained.
        self._closed = True
