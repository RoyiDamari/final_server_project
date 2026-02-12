import os
from contextlib import suppress
from typing import Set
from app.utils.files import safe_unlink


def sweep_tmp_files(base_dir: str = "saved_models") -> int:
    """
    Delete all *.tmp files under saved_models/**.
    Safe if you run this before accepting requests (startup reconciliation).
    Returns count deleted.
    """
    deleted = 0
    if not os.path.isdir(base_dir):
        return 0

    for root, _dirs, files in os.walk(base_dir):
        for name in files:
            if not name.endswith(".tmp"):
                continue
            path = os.path.join(root, name)
            with suppress(Exception):
                safe_unlink(path)
                deleted += 1
    return deleted


def sweep_orphan_final_files(referenced_final_paths: Set[str], base_dir: str = "saved_models") -> int:
    """
    Delete final *.pkl files that exist on disk but are NOT referenced by the DB.

    Why can this happen?
    - move_temp_to_final succeeded
    - process died before DB commit
    => file exists, row does not

    Returns count deleted.
    """
    deleted = 0
    if not os.path.isdir(base_dir):
        return 0

    # Normalize referenced paths to absolute paths for robust comparison
    base_abs = os.path.abspath(base_dir)
    referenced_abs = set()

    for p in referenced_final_paths:
        # If DB stored relative path, make it comparable
        if os.path.isabs(p):
            referenced_abs.add(os.path.abspath(p))
        else:
            referenced_abs.add(os.path.abspath(os.path.join(base_abs, p)))

    for root, _dirs, files in os.walk(base_dir):
        for name in files:
            # only finals
            if not name.endswith(".pkl"):
                continue

            full_path = os.path.abspath(os.path.join(root, name))

            # ignore any accidental "pkl.tmp" (already handled by tmp sweep)
            if full_path.endswith(".tmp"):
                continue

            if full_path not in referenced_abs:
                with suppress(Exception):
                    safe_unlink(full_path)
                    deleted += 1

    return deleted


def sweep_tmp_dir(base_dir: str) -> int:
    """
    Delete all files inside base_dir (non-recursive is fine for _tmp; you can
    make it recursive if you nest).
    Returns number of deleted files.
    """
    if not os.path.isdir(base_dir):
        return 0

    deleted = 0
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            with suppress(Exception):
                os.remove(p)
                deleted += 1
    return deleted