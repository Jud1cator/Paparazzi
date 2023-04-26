import logging
import os
from zipfile import ZipFile

from pendulum.datetime import DateTime


class Archiver:
    """
    Puts all data from data_dirpath into zip archive and saves it to data_archive_root
    """

    def __init__(self, name: str, data_archive_root: str, timezone: str):
        self.name = name
        self.data_archive_root = data_archive_root
        self.timezone = timezone

    def _build_archive_name(self, data_interval_end_utc) -> str:
        date = data_interval_end_utc.in_tz(self.timezone).date().isoformat()
        return f"{date}_{self.name}.zip"

    def archive(
        self,
        files_to_archive: list[str],
        data_interval_end_utc: DateTime,
    ) -> str:
        """
        Archives all files from self.data_dirpath created past last_loaded_ts
        :returns: Path to saved archive
        """
        archive_path = os.path.join(
            self.data_archive_root, self._build_archive_name(data_interval_end_utc)
        )

        if os.path.exists(archive_path):
            raise FileExistsError(f"File already exists: {archive_path}")

        if len(files_to_archive) == 0:
            raise RuntimeError("No files to add to archive.")
        
        logging.info("Files to add to archive: %s", len(files_to_archive))

        with ZipFile(archive_path, "w") as zip_obj:
            for fp in files_to_archive:
                zip_obj.write(fp, os.path.basename(fp))

        return archive_path
