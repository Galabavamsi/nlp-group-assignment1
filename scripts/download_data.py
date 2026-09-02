"""Download the corpora required by Question 1."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--english-only", action="store_true")
    args = parser.parse_args()
    import nltk

    nltk.download("brown")
    if not args.english_only:
        destination = Path(args.data_dir) / "UD_Spanish-GSD"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/UniversalDependencies/UD_Spanish-GSD.git", str(destination)],
                check=True,
            )
    print("Corpus setup complete.")


if __name__ == "__main__":
    main()
