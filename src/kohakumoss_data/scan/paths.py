"""Where a source's catalog file lives: `<cache>/catalog/<kind>-<digest>.db`, the digest
over the source kind, its absolute path and its options."""

import hashlib
import json
import os


def scan_path(cache_dir: str, kwargs: dict) -> str:
    """The catalog file of a source built from `kwargs` (`source` + its arguments)."""
    canon = dict(kwargs)
    for key in ("path", "root"):
        if canon.get(key) and "*" not in str(canon[key]):
            canon[key] = os.path.abspath(str(canon[key]))
    blob = json.dumps(canon, sort_keys=True, default=str).encode()
    digest = hashlib.sha1(blob).hexdigest()[:16]
    return os.path.join(cache_dir, "catalog", f"{kwargs['source']}-{digest}.db")
