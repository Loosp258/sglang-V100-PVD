"""The wire byte plan must reconstruct a component-major Prompt shard."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.pvd.protocol import ProtocolValidationError
from sglang.srt.disaggregation.pvd.sharding import (
    PackedTransferSlice,
    plan_prompt_chunk_puts,
)


class TestPromptChunkLayout(unittest.TestCase):
    def setUp(self):
        self.layout = SimpleNamespace(
            page_size=2, extra={"component_bytes_per_token": [4, 8]}
        )
        self.total_pages = 3
        self.complete = bytes(range(24)) + bytes(range(100, 148))

    def plan(self, first_page, page_count, expected_bytes=72):
        return plan_prompt_chunk_puts(
            self.layout,
            total_pages=self.total_pages,
            expected_bytes=expected_bytes,
            first_page=first_page,
            page_count=page_count,
        )

    def test_page_chunks_reconstruct_complete_shard(self):
        destination = bytearray(72)
        for first_page, page_count in ((0, 1), (1, 2)):
            plan = self.plan(first_page, page_count)
            source = b"".join(
                self.complete[item.remote_offset : item.remote_offset + item.length]
                for item in plan
            )
            for item in plan:
                destination[item.remote_offset : item.remote_offset + item.length] = (
                    source[item.local_offset : item.local_offset + item.length]
                )
        self.assertEqual(bytes(destination), self.complete)
        self.assertEqual(
            self.plan(1, 1),
            (
                # K component: page 1 of a three-page slab.
                PackedTransferSlice(0, 8, 8),
                # V component starts at byte 24, then page 1 is +16.
                PackedTransferSlice(8, 40, 16),
            ),
        )

    def test_invalid_ranges_and_manifest_bytes_are_refused(self):
        for first_page, page_count, expected_bytes in (
            (-1, 1, 72),
            (0, 0, 72),
            (2, 2, 72),
            (0, 1, 71),
            (True, 1, 72),
        ):
            with self.subTest(first_page=first_page, page_count=page_count):
                with self.assertRaises(ProtocolValidationError):
                    self.plan(first_page, page_count, expected_bytes)


if __name__ == "__main__":
    unittest.main()
