"""One-time download of the TrOCR handwriting models into models/<name>/ (the only network use).

Run once:  python download_models.py
The pipeline then loads them from disk with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository
"""

from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS = ["microsoft/trocr-small-handwritten", "microsoft/trocr-base-handwritten"]

for repo in MODELS:
    target = MODELS_DIR / repo.split("/")[-1]
    snapshot_download(repo_id=repo, local_dir=target)
    print(f"{repo} -> {target}")
