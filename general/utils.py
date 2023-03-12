import os
from typing import List


def traverse_all_files(data_root: str) -> List[str]:
    all_files: List[str] = []
    for outer, inner, files in os.walk(data_root):
        for f in files:
            all_files.append(os.path.join(outer, f))
    return all_files


def keep_files_with_extension(files: List[str], extension: str) -> List[str]:
    kept_files: List[str] = list(
        filter(lambda p: os.path.splitext(p)[1] == extension, files)
    )
    return kept_files
