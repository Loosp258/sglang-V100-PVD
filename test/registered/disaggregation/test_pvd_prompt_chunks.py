"""Chunk identity stays bound to one Entry and advances only after proof."""

import unittest

from sglang.srt.disaggregation.pvd.prompt_chunks import (
    PromptChunkIdentity,
    PromptChunkProgress,
)
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    KVEntryKey,
    KVLayoutSignature,
    KVShardManifest,
    ProtocolValidationError,
    WriteIdentity,
)


class TestPromptChunks(unittest.TestCase):
    def setUp(self):
        self.layout = KVLayoutSignature(
            model_id="model",
            model_revision="revision",
            kv_dtype="float16",
            page_size=2,
            num_layers=1,
            total_kv_heads=2,
            kv_heads_per_rank=1,
            head_dim=2,
            tp_size=2,
            pp_size=1,
            tensor_layout="component-major",
            extra={"component_bytes_per_token": [4, 8]},
        )
        self.shard = KVShardManifest(
            rank=0,
            rail="rail",
            expected_bytes=72,
            page_count=3,
            last_page_valid_tokens=1,
            layer_start=0,
            layer_end=1,
        )
        self.base = WriteIdentity(
            protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch="sender",
            receiver_epoch="receiver",
            transfer_id="upload:entry:v0",
            region_id="region",
            generation="generation",
            shard_rank=0,
            key=KVEntryKey("model", "request", "entry"),
        )

    def test_ordered_chunks_only_advance_on_proven_success(self):
        progress = PromptChunkProgress(self.base, self.layout, self.shard)
        first = progress.begin(0, 1)
        self.assertEqual(first.chunk_bytes, 24)
        self.assertEqual(first.write.transfer_id, "upload:entry:v0:pages:0:1")
        self.assertEqual(
            first,
            PromptChunkIdentity.from_dict(
                first.to_dict(),
                layout=self.layout,
                shard=self.shard,
                base=self.base,
            ),
        )
        self.assertEqual(progress.complete_pages, 0)
        with self.assertRaises(ProtocolValidationError):
            progress.begin(1, 2)
        self.assertEqual(
            progress.complete(first, transferred_bytes=24, terminal_success=True), 1
        )
        with self.assertRaises(ProtocolValidationError):
            progress.begin(0, 1)
        final = progress.begin(1, 2)
        self.assertTrue(final.final)
        self.assertEqual(final.chunk_bytes, 48)
        self.assertEqual(
            progress.complete(final, transferred_bytes=48, terminal_success=True), 3
        )
        self.assertTrue(progress.finished)
        with self.assertRaises(ProtocolValidationError):
            progress.begin(3, 1)

    def test_replay_and_changed_authority_are_rejected(self):
        progress = PromptChunkProgress(self.base, self.layout, self.shard)
        first = progress.begin(0, 1)
        changed = first.to_dict()
        changed["write"] = {**changed["write"], "sender_epoch": "other"}
        with self.assertRaises(ProtocolValidationError):
            PromptChunkIdentity.from_dict(
                changed, layout=self.layout, shard=self.shard, base=self.base
            )
        with self.assertRaises(ProtocolValidationError):
            progress.complete(first, transferred_bytes=23, terminal_success=True)
        self.assertTrue(progress.failed)
        with self.assertRaises(ProtocolValidationError):
            progress.begin(0, 1)

    def test_failed_or_unknown_terminal_never_advances(self):
        progress = PromptChunkProgress(self.base, self.layout, self.shard)
        first = progress.begin(0, 1)
        with self.assertRaises(ProtocolValidationError):
            progress.complete(first, transferred_bytes=24, terminal_success=False)
        self.assertEqual(progress.complete_pages, 0)
        self.assertFalse(progress.finished)


if __name__ == "__main__":
    unittest.main()
