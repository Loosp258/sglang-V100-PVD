"""Launch PVD V with native CAGRA before SGLang/torch loads CUDA libraries.

Use ``python -m pvd_cagra_server`` with the usual PVD V-server arguments when
``--prompt-index-backend`` is ``cagra`` or ``cagra-auto``. The cuVS 25.02
wheel in the V100S candidate needs this import order; the normal index-off or
exact V launcher has no cuVS dependency.
"""

# This import must stay before *any* sglang import. Python imports sglang's
# package initializer before running a module inside that namespace, so an
# equivalent import in sglang.srt.disaggregation.pvd.server is too late.
import cuvs.neighbors.cagra  # noqa: F401

from sglang.srt.disaggregation.pvd.server import main


if __name__ == "__main__":
    main()
