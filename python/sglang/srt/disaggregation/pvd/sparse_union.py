"""Explicit per-layer/KV-head GQA union. No score merging or truncation."""

from sglang.srt.disaggregation.pvd.probe_search import ProbeSelection
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.sparse_payload import (
    SparseKVSpec,
    SparsePayloadError,
)


def union_query_head_selections(
    selection: ProbeSelection,
    *,
    mapping: QueryHeadMapping,
    layout_fingerprint: str,
    max_union_tokens: int,
):
    """Require every Q head of each represented KV head, then union token ids.

    The limit is mandatory. Over-capacity refuses the whole operation; dropping
    tokens by a hidden score policy would change the user's model semantics.
    Multiple V shards keep independent index/mapping versions per KV-head group.
    This function accepts one completed shard search window at a time.
    """
    if type(max_union_tokens) is not int or max_union_tokens <= 0:
        raise SparsePayloadError("max_union_tokens must be explicitly positive")
    if not isinstance(selection, ProbeSelection) or not isinstance(
        mapping, QueryHeadMapping
    ):
        raise SparsePayloadError(
            "complete probe selection and explicit head mapping required"
        )
    if not selection.queries or len(selection.queries) != len(selection.selections):
        raise SparsePayloadError("search query/result counts disagree")
    groups = {}
    window = selection.window
    for query, result in zip(selection.queries, selection.selections, strict=True):
        route = query.route
        identity = result.identity
        if (
            identity.entry_transfer_id != window.entry_transfer_id
            or identity.layer != route.identity.layer
            or identity.kv_head != route.identity.kv_head
            or identity.vector_space != route.identity.vector_space
            or identity.positional_encoding != route.identity.positional_encoding
            or mapping.kv_head_for(route.query_head) != identity.kv_head
        ):
            raise SparsePayloadError("result does not match its query route/Entry")
        key = (identity.layer, identity.kv_head)
        version = (result.index_version, result.id_mapping_version)
        context = (
            identity.vector_space,
            identity.positional_encoding,
            route.scope,
            query.query_version,
        )
        if key not in groups:
            groups[key] = (version, context, set(), set())
        versions, bound_context, heads, tokens = groups[key]
        if version != versions or context != bound_context or route.query_head in heads:
            raise SparsePayloadError("mixed versions or duplicate Q head in union")
        if (
            identity.entry_transfer_id != route.identity.entry_transfer_id
            or result.metric != route.scope.metric
            or not result.token_ids
            or len(set(result.token_ids)) != len(result.token_ids)
        ):
            raise SparsePayloadError("inconsistent search result or token list")
        if any(
            type(t) is not int or not 0 <= t < route.scope.prompt_tokens
            for t in result.token_ids
        ):
            raise SparsePayloadError("out-of-range union token")
        heads.add(route.query_head)
        tokens.update(result.token_ids)
        if len(tokens) > max_union_tokens:
            raise SparsePayloadError(
                "per-layer/KV-head union capacity exceeded; no truncation"
            )
    specs = []
    for (layer, kv_head), (versions, _, heads, tokens) in sorted(groups.items()):
        expected_heads = set(
            range(kv_head * mapping.group_size, (kv_head + 1) * mapping.group_size)
        )
        if heads != expected_heads:
            raise SparsePayloadError("union requires every Q head sharing this KV head")
        specs.append(
            SparseKVSpec(
                request_id=window.prefix.request_id,
                incarnation=window.incarnation,
                operation_id=window.operation_id,
                target_tokens=window.target_tokens,
                entry_transfer_id=window.entry_transfer_id,
                index_version=versions[0],
                id_mapping_version=versions[1],
                layout_fingerprint=layout_fingerprint,
                layer=layer,
                kv_head=kv_head,
                token_ids=tuple(sorted(tokens)),
            )
        )
    return tuple(specs)
