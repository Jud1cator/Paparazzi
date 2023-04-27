import datetime
import os
import subprocess

import pendulum

from airflow.operators.bash import BashOperator
from airflow.models import Variable


class RTSPStreamExtractor:
    name: str
    rtsp_addr: str
    target_data_root: str
    image_format: str

    def __init__(
        self,
        name: str,
        rtsp_addr: str,
        target_data_root: str,
        image_format: str,
    ) -> None:
        self.rtsp_addr = rtsp_addr
        self.target_data_root = target_data_root
        self.name = name
        self.image_format = image_format

    def _build_save_path(self) -> str:
        return os.path.join(
            self.target_data_root,
            f"{self.name}_{pendulum.now('Europe/Moscow').isoformat()}.{self.image_format}",
        )

    def _build_bash_command(self) -> list[str]:
        fp = self._build_save_path()
        return [
            "mkdir",
            "-p",
            self.target_data_root,
            "&&",
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-y",
            "-i",
            f"'{self.rtsp_addr}'",
            "-vframes",
            "1",
            fp,
        ]

    def build_operator(self):
        return BashOperator(
            task_id=f"{self.name}_RTSPStreamExtractor",
            bash_command=" ".join(self._build_bash_command()),
            execution_timeout=datetime.timedelta(
                seconds=int(Variable.get("RTSPSTREAMEXTRACTOR_TIMEOUT", 60))
            ),
        )

    def extract(self):
        subprocess.Popen(self._build_bash_command())
