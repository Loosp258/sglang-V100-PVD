"""HTTP health semantics for Gateway-routed PVD workers.

PVD requires Gateway-assigned transfer, delivery, and vector identities.  A
synthetic generation request cannot supply them, so it must never be used as a
worker health probe.
"""


def pvd_health_http_status(
    *, topology: str, mode: str, endpoint: str, server_status: str
) -> int | None:
    """Return a PVD-only status, or ``None`` for the ordinary PD health path."""
    if topology != "pvd" or mode not in ("prefill", "decode"):
        return None
    if endpoint == "/health_generate":
        # This endpoint promises an actual generation probe.  PVD can only
        # execute one after a Gateway has assigned its request identities.
        return 503
    return 200 if server_status == "Up" else 503
