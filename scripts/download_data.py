"""Download and verify the official Corr2Cause splits from the pinned manifest."""

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    directory = ROOT / "corr2cause"
    manifest = json.loads((directory / "download_manifest.json").read_text())
    base_url = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{manifest['revision']}"
    for name, expected in manifest["files"].items():
        target = directory / name
        if target.is_file() and checksum(target) == expected["sha256"]:
            print(f"{name}: already verified ({expected['rows']} rows)", flush=True)
            continue

        with NamedTemporaryFile(dir=directory, prefix=f"{name}.", suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            subprocess.run(
                ["curl", "-fsSL", "--retry", "3", f"{base_url}/{name}", "-o", str(temporary)],
                check=True,
            )
            if checksum(temporary) != expected["sha256"]:
                raise ValueError(f"Checksum mismatch for {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{name}: downloaded and verified ({expected['rows']} rows)", flush=True)


if __name__ == "__main__":
    main()
