"""Download the corpora required by the assignment."""

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

    data_dir = Path(args.data_dir)
    nltk_data_dir = data_dir / "nltk"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    if not nltk.download("brown", download_dir=str(nltk_data_dir)):
        raise SystemExit("Failed to download the Brown corpus.")
    if not args.english_only:
        destination = data_dir / "UD_Spanish-GSD"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/UniversalDependencies/UD_Spanish-GSD.git", str(destination)],
                check=True,
            )
    print(f"Corpus setup complete. NLTK data: {nltk_data_dir}")


if __name__ == "__main__":
    main()
