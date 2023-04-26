import os

import yadisk


def yadisk_upload(token: str, yadisk_save_dir: str, filepath: str, timeout: int | tuple[int, int]):
    y = yadisk.YaDisk(token=token)
    filename = os.path.basename(filepath)
    ydisk_filepath = os.path.join(yadisk_save_dir, filename)
    y.upload(filepath, ydisk_filepath, timeout=timeout)
    return ydisk_filepath
