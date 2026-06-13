#!/usr/bin/env python3
"""Upload en_fr_artifacts/ to a Hugging Face model repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "en_fr_artifacts"
REQUIRED = ("best_model.keras", "src_tokenizer.pkl", "tgt_tokenizer.pkl", "meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload EN→FR artifacts to Hugging Face Hub")
    parser.add_argument("repo_id", help="Model repo id, e.g. your-username/en-fr-translator")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Artifacts folder")
    parser.add_argument("--private", action="store_true", help="Create a private model repo")
    args = parser.parse_args()

    folder = args.dir.resolve()
    missing = [name for name in REQUIRED if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"Missing in {folder}: {', '.join(missing)}")

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload EN→FR model artifacts",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    print(f"Set HF_MODEL_REPO={args.repo_id} in your Space settings.")


if __name__ == "__main__":
    main()
