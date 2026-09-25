"""Opt-in stack dump for isolated PVD load experiments.

Add this directory to PYTHONPATH and set PVD_STACK_SIGNAL=1 before starting
the sidecar. SIGUSR2 then prints all Python thread stacks to that process's
stderr without stopping it. Never enable this for unrelated services.
"""

import faulthandler
import os
import signal
import sys

if os.environ.get("PVD_STACK_SIGNAL") == "1":
    faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
