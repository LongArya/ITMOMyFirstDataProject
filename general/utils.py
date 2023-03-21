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


def get_new_pattern_name_folder(root: str, pattern: str) -> str:
    """Get new folder with name like {pattern}_{num}"""
    if not os.path.exists(root):
        return os.path.join(root, f"{pattern}_{0:05d}")
    root_content = os.listdir(root)
    pattern_named_dirs = sorted(
        list(filter(lambda path: path.startswith(pattern), root_content))
    )
    if not pattern_named_dirs:
        return os.path.join(root, f"{pattern}_{0:05d}")
    max_pattern_name = os.path.splitext(pattern_named_dirs[-1])[0]
    max_pattern_num = int(max_pattern_name.split(f"_")[-1])
    return os.path.join(root, f"{pattern}_{max_pattern_num + 1:05d}")
