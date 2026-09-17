import os
from pathlib import Path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_symlink(src, dst):
    # os.replace is atomic, so dst never disappears mid-update.
    dst_path = Path(dst)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    if tmp_path.is_symlink() or tmp_path.exists():
        tmp_path.unlink()
    tmp_path.symlink_to(Path(src).resolve())
    os.replace(tmp_path, dst_path)
