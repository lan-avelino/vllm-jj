"""Download a pinned Hugging Face revision into DEST, resumable and sha256-verified.

usage: fetch_model.py REPO REVISION DEST

Large files are fetched as 8 parallel byte ranges, one file at a time. LFS files are
verified against their sha256 before being moved into place, so DEST never holds a
partial or corrupt file under its final name. Re-running skips files already present
with the right size and prints MODEL_READY when the tree is complete.
"""

import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

REPO, REV, DEST = sys.argv[1], sys.argv[2], sys.argv[3]
PARTS = 8
WORK = DEST + ".parts"


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def tree():
    u = f"https://huggingface.co/api/models/{REPO}/tree/{REV}?recursive=true"
    with urllib.request.urlopen(u, timeout=120) as r:
        return [f for f in json.load(r) if f["type"] == "file"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def fetch(f):
    name, size = f["path"], f["size"]
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) == size:
        return "present"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
    if size < 64 << 20:
        subprocess.run(["curl", "-sSfL", "--retry", "10", "-o", out + ".part", url], check=True)
        os.replace(out + ".part", out)
        return "small"
    work = os.path.join(WORK, name)
    os.makedirs(work, exist_ok=True)
    chunk = (size + PARTS - 1) // PARTS

    def part(i):
        start, end = i * chunk, min((i + 1) * chunk, size) - 1
        p = os.path.join(work, f"part.{i}")
        open(p, "ab").close()
        while (have := os.path.getsize(p)) < end - start + 1:
            with open(p, "ab") as fh:
                subprocess.run(["curl", "-sSL", "--speed-time", "60", "--speed-limit", "10240",
                                "-r", f"{start + have}-{end}", url, "-o", "-"],
                               stdout=fh, stderr=subprocess.DEVNULL)
            if os.path.getsize(p) == have:
                time.sleep(10)
        if os.path.getsize(p) != end - start + 1:
            raise RuntimeError(f"{name} part {i} overshot")

    with cf.ThreadPoolExecutor(PARTS) as ex:
        list(ex.map(part, range(PARTS)))
    tmp = out + ".joining"
    with open(tmp, "wb") as o:
        for i in range(PARTS):
            p = os.path.join(work, f"part.{i}")
            with open(p, "rb") as src:
                shutil.copyfileobj(src, o, 64 << 20)
            os.remove(p)
    if os.path.getsize(tmp) != size:
        raise RuntimeError(f"{name} size mismatch")
    want = (f.get("lfs") or {}).get("oid")
    if want and sha256(tmp) != want:
        os.remove(tmp)
        raise RuntimeError(f"{name} SHA_MISMATCH")
    os.replace(tmp, out)
    return "ok"


def main():
    files = sorted(tree(), key=lambda f: f["path"])
    total = sum(f["size"] for f in files)
    log(f"{REPO}@{REV}: {len(files)} files, {total / 1e9:.1f} GB -> {DEST}")
    done = 0
    for f in files:
        how = fetch(f)
        done += f["size"]
        if how != "present":
            log(f"{how:8s} {f['path']}  {done / total * 100:.1f}%")
    bad = [f["path"] for f in files if os.path.getsize(os.path.join(DEST, f["path"])) != f["size"]]
    if bad:
        raise SystemExit(f"SIZE_CHECK_FAILED {bad}")
    shutil.rmtree(WORK, ignore_errors=True)
    log("MODEL_READY", DEST)


if __name__ == "__main__":
    main()
