"""Check that the whole pipeline runs with no network access.

Run from the project root:  python tests/check_offline.py [folder | image ...]

Every outgoing socket connection in this process is blocked (socket.socket.connect,
connect_ex, and socket.create_connection raise), so any attempt to reach the network,
including a model download, makes the run fail. The guard is tested first by trying to
reach huggingface.co. For a full check, also run `python pipeline.py` with networking
turned off in Windows (e.g. airplane mode).
https://docs.python.org/3/library/socket.html#socket.socket.connect
"""

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class NetworkBlocked(OSError):
    pass


def blocked(*args, **kwargs):
    raise NetworkBlocked(f"network access attempted: {args[1:] or args}")


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked

try:
    socket.create_connection(("huggingface.co", 443), timeout=5)
    sys.exit("FAIL  the network guard did not block a connection")
except NetworkBlocked:
    print("PASS  network guard blocks connections")

import pipeline  # noqa: E402

pipeline.main(sys.argv[1:])
if not pipeline.RESULTS_PATH.is_file():
    sys.exit(f"FAIL  {pipeline.RESULTS_PATH} was not written")
print(f"PASS  pipeline ran with the network blocked and wrote {pipeline.RESULTS_PATH}")
