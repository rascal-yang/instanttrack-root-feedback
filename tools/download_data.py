#!/usr/bin/env python3
"""Download pinned G1 references/assets and check the recorded experiment hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
BASE = f"https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/{REVISION}"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("INSTANTTRACK_DATA_DIR", ROOT / "data")))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((ROOT / "reproducibility/data_sha256.json").read_text())
    for name, expected in manifest.items():
        target = output / name
        if target.is_file() and sha256(target) == expected:
            continue
        if args.verify_only:
            raise RuntimeError("Missing or changed experiment input: " + str(target))
        if target.exists():
            raise RuntimeError("Refusing to overwrite a different input: " + str(target))
        if name == "g1_description/g1_29dof.xml":
            continue
        remote = ("g1/" + Path(name).name if name.startswith("lafan1_g1/")
                  else "robot_description/g1/" + name.removeprefix("g1_description/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(BASE + "/" + remote, timeout=60) as response, temporary.open("wb") as stream:
                    while block := response.read(2 ** 20):
                        stream.write(block)
                if sha256(temporary) != expected:
                    raise RuntimeError("Downloaded checksum mismatch: " + name)
                temporary.replace(target)
                print("verified", name, flush=True)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    model = output / "g1_description/g1_29dof.xml"
    if not args.verify_only and not model.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/urdf_to_mjcf.py")], check=True,
                       env=dict(os.environ, INSTANTTRACK_DATA_DIR=str(output)))
    changed = [name for name, expected in manifest.items()
               if not (output / name).is_file() or sha256(output / name) != expected]
    if changed:
        raise RuntimeError("Inputs differ from the recorded experiment: " + ", ".join(changed))
    print(json.dumps(dict(verified_files=len(manifest), motions=40, revision=REVISION, data_dir=str(output))))


if __name__ == "__main__":
    main()
