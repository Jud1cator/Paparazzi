import os
from pathlib import Path

import pendulum
from pendulum.datetime import DateTime


def touch_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def get_files_created_between(
    dirpath: str,
    interval_start_utc: DateTime,
    interval_end_utc: DateTime,
) -> list[str]:
    files_to_save = []
    for fname in os.listdir(dirpath):
        fpath = os.path.join(dirpath, fname)
        ftime = pendulum.from_timestamp(os.path.getmtime(fpath))
        if ftime >= interval_start_utc and ftime < interval_end_utc:
            files_to_save.append(fpath)

    return files_to_save
