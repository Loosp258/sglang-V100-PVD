"""Checked identities and ordered progress for provisional Prompt KV chunks.

This module does not authorize a PUT. A caller must create one native write
authorization for each ``PromptChunkIdentity.write`` and prove its terminal
success before calling ``PromptChunkProgress.complete``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from sglang.srt.disaggregation.pvd.protocol import (
    KVLayoutSignature,
    KVShardManifest,
    ProtocolValidationError,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.sharding import (
    PackedTransferSlice,
    plan_prompt_chunk_puts,
)


@dataclass(frozen=True)
class PromptChunkIdentity:
    """One page interval within an existing immutable Entry shard allocation."""

    write: WriteIdentity
    first_page: int
    page_count: int
    total_pages: int
    expected_bytes: int
    chunk_bytes: int
    final: bool
    layout: KVLayoutSignature = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.write, WriteIdentity):
            raise ProtocolValidationError("chunk needs a complete write identity")
        for name in (
            "first_page",
            "page_count",
            "total_pages",
            "expected_bytes",
            "chunk_bytes",
        ):
            if type(getattr(self, name)) is not int:
                raise ProtocolValidationError(f"chunk {name} must be an integer")
        if type(self.final) is not bool:
            raise ProtocolValidationError("chunk final flag must be boolean")
        slices = self.slices
        if self.final != (self.first_page + self.page_count == self.total_pages):
            raise ProtocolValidationError("chunk final flag disagrees with page range")
        if self.chunk_bytes != sum(item.length for item in slices):
            raise ProtocolValidationError("chunk byte count disagrees with layout")
        suffix = f":pages:{self.first_page}:{self.first_page + self.page_count}"
        if not self.write.transfer_id.endswith(suffix):
            raise ProtocolValidationError("chunk transfer id does not bind its range")

    @property
    def slices(self) -> tuple[PackedTransferSlice, ...]:
        return plan_prompt_chunk_puts(
            self.layout,
            total_pages=self.total_pages,
            expected_bytes=self.expected_bytes,
            first_page=self.first_page,
            page_count=self.page_count,
        )

    @classmethod
    def issue(
        cls,
        base: WriteIdentity,
        layout: KVLayoutSignature,
        shard: KVShardManifest,
        *,
        first_page: int,
        page_count: int,
    ) -> PromptChunkIdentity:
        if base.shard_rank != shard.rank:
            raise ProtocolValidationError("chunk identity and shard rank differ")
        slices = plan_prompt_chunk_puts(
            layout,
            total_pages=shard.page_count,
            expected_bytes=shard.expected_bytes,
            first_page=first_page,
            page_count=page_count,
        )
        write = replace(
            base,
            transfer_id=f"{base.transfer_id}:pages:{first_page}:{first_page + page_count}",
        )
        return cls(
            write=write,
            first_page=first_page,
            page_count=page_count,
            total_pages=shard.page_count,
            expected_bytes=shard.expected_bytes,
            chunk_bytes=sum(item.length for item in slices),
            final=first_page + page_count == shard.page_count,
            layout=layout,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "write": self.write.to_dict(),
            "first_page": self.first_page,
            "page_count": self.page_count,
            "total_pages": self.total_pages,
            "expected_bytes": self.expected_bytes,
            "chunk_bytes": self.chunk_bytes,
            "final": self.final,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        layout: KVLayoutSignature,
        shard: KVShardManifest,
        base: WriteIdentity,
    ) -> PromptChunkIdentity:
        fields = {
            "write",
            "first_page",
            "page_count",
            "total_pages",
            "expected_bytes",
            "chunk_bytes",
            "final",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ProtocolValidationError("chunk identity has unexpected fields")
        if type(value["final"]) is not bool or any(
            type(value[name]) is not int
            for name in (
                "first_page",
                "page_count",
                "total_pages",
                "expected_bytes",
                "chunk_bytes",
            )
        ):
            raise ProtocolValidationError(
                "chunk range and byte fields have invalid types"
            )
        result = cls.issue(
            base,
            layout,
            shard,
            first_page=value["first_page"],
            page_count=value["page_count"],
        )
        if result.to_dict() != dict(value):
            raise ProtocolValidationError("chunk identity differs from Entry authority")
        return result


class PromptChunkProgress:
    """Accept only terminal-success chunks in Prompt order.

    One chunk may be submitted at a time per shard. This deliberately bounds
    staging and ensures the index worker can consume a contiguous prefix.
    """

    def __init__(
        self, base: WriteIdentity, layout: KVLayoutSignature, shard: KVShardManifest
    ):
        if base.shard_rank != shard.rank:
            raise ProtocolValidationError("chunk progress belongs to another shard")
        self.base = base
        self.layout = layout
        self.shard = shard
        self.next_page = 0
        self.pending: PromptChunkIdentity | None = None
        self.failed = False

    @property
    def complete_pages(self) -> int:
        return self.next_page

    @property
    def finished(self) -> bool:
        return self.next_page == self.shard.page_count and self.pending is None

    def begin(self, first_page: int, page_count: int) -> PromptChunkIdentity:
        if self.failed or self.finished or self.pending is not None:
            raise ProtocolValidationError("chunk progress is unavailable")
        if first_page != self.next_page:
            raise ProtocolValidationError("chunk pages must be contiguous and ordered")
        chunk = PromptChunkIdentity.issue(
            self.base,
            self.layout,
            self.shard,
            first_page=first_page,
            page_count=page_count,
        )
        self.pending = chunk
        return chunk

    def complete(
        self,
        chunk: PromptChunkIdentity,
        *,
        transferred_bytes: int,
        terminal_success: bool,
    ) -> int:
        if self.failed or self.pending != chunk:
            raise ProtocolValidationError("chunk completion identity does not match")
        if type(terminal_success) is not bool or not terminal_success:
            self.failed = True
            raise ProtocolValidationError("chunk has no successful native terminal")
        if type(transferred_bytes) is not int or transferred_bytes != chunk.chunk_bytes:
            self.failed = True
            raise ProtocolValidationError("chunk native byte count does not match")
        self.next_page += chunk.page_count
        self.pending = None
        return self.next_page
