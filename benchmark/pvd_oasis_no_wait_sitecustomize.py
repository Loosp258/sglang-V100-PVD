"""Copy verbatim to an explicit diagnostic PYTHONPATH directory as sitecustomize.py.

Every spawned Python interpreter installs the lazy hook independently. Normal
serving has no diagnostic directory or PVD_OASIS_REPLAY_CONFIG environment.
"""

import os

if os.environ.get("PVD_OASIS_REPLAY_CONFIG"):
    from pvd_oasis_no_wait_replay import install

    install()
