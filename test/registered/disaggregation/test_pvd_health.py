"""PVD workers cannot run identity-free synthetic health generation."""

import unittest

from sglang.srt.disaggregation.pvd.health import pvd_health_http_status


class PvdHealthTests(unittest.TestCase):
    def test_pvd_health_reads_status_without_generating(self):
        for mode in ("prefill", "decode"):
            for status, expected in (("Up", 200), ("Starting", 503), ("UnHealthy", 503)):
                with self.subTest(mode=mode, status=status):
                    self.assertEqual(
                        pvd_health_http_status(
                            topology="pvd",
                            mode=mode,
                            endpoint="/health",
                            server_status=status,
                        ),
                        expected,
                    )
            self.assertEqual(
                pvd_health_http_status(
                    topology="pvd",
                    mode=mode,
                    endpoint="/health_generate",
                    server_status="Up",
                ),
                503,
            )

    def test_non_pvd_health_is_unchanged(self):
        for topology, mode in (("pd", "prefill"), ("pd", "decode"), ("pvd", "null")):
            with self.subTest(topology=topology, mode=mode):
                self.assertIsNone(
                    pvd_health_http_status(
                        topology=topology,
                        mode=mode,
                        endpoint="/health_generate",
                        server_status="Up",
                    )
                )
