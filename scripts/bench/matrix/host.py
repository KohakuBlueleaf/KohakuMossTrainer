"""Bench matrix host runner (on the host, from the frozen tree):

    python scripts/bench/matrix/host.py --config <root>/configs/<bench>-<host>.json

See kohakumoss_bench.run.host and docs/performance/bench-matrix.md#host-runner.
"""

import sys

from kohakumoss_bench.run.host import main

if __name__ == "__main__":
    sys.exit(main())
