from typing import Optional
import os

from airflow.operators.bash import BashOperator


class RSyncExtractor:
    """
    Downloads all files in source_data_root from specified host to target_data_root using rsync.
    """

    host: str
    username: str
    ssh_port: str
    source_data_root: str
    target_data_root: str
    name: str

    def __init__(
        self,
        host: str,
        username: str,
        source_data_root: str,
        target_data_root: str,
        ssh_port: str = "22",
        name: Optional[str] = None,
    ) -> None:
        """
        :param host: host address in format 'user@ip'
        :param source_data_root: absolute path to folder with source data on host
        :param target_data_root: absolute path to destination folder on executor
        :ssh_port: optional argument to specify ssh port is its not standard (22)
        """
        self.host = host
        self.username = username
        self.ssh_port = ssh_port
        self.source_data_root = source_data_root
        self.target_data_root = target_data_root
        self.name = self.host if name is None else name

        if not os.path.isabs(self.source_data_root):
            raise ValueError(
                "source_data_root must be an absolute path: %s", self.source_data_root
            )

        if not os.path.isabs(self.target_data_root):
            raise ValueError(
                "target_data_root must be an absolute path: %s", self.target_data_root
            )

    def _build_bash_command(self):
        ssh_options = (
            "-e 'ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i /ssh_key -p"
            f" {self.ssh_port}'"
        )
        # Two lines below add trailing slashes to source and target data root paths if they are
        # missing so the rsync will synchronize content of source_data_root with content of
        # target_data_root folder
        source = os.path.join(self.source_data_root, str())
        target = os.path.join(self.target_data_root, str())
        cmd = (
            f"mkdir -p {target} && rsync -azvt --stats"
            f" {ssh_options} {self.username}@{self.host}:{source} {target}"
        )
        return cmd

    def build_operator(self):
        return BashOperator(
            task_id=f"{self.name}_RSYNCExtractor",
            bash_command=self._build_bash_command(),
        )
