"""Copy the published results into the local demo and test output folders."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def main():
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    for phase in ("phase1", "phase2"):
        destination = outputs / phase
        if not destination.exists():
            with TemporaryDirectory(dir=outputs) as staging:
                snapshot = Path(staging) / phase
                shutil.copytree(ROOT / "results" / phase, snapshot)
                try:
                    snapshot.rename(destination)
                except OSError:
                    if not destination.is_dir():
                        raise
                else:
                    print(f"{phase}: restored saved results")
                    continue
        print(f"{phase}: using existing {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
