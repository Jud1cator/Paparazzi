import json
import logging
import os
from typing import Any

from airflow.decorators import dag, task
from airflow.models import Variable
from pendulum import datetime
from pendulum.datetime import DateTime
from pendulum.parser import parse as parse_dt
from airflow.decorators import dag, task
from airflow.models import Variable

TIMEZONE = "Europe/Moscow"


@task
def select_files_task(
    extractor_name: str,
    data_interval_start_utc: str,
    data_interval_end_utc: str,
) -> list[str]:
    from dags.ydisk_loader.utils.os_utils import get_files_created_between

    data_dirpath = os.path.join(Variable.get("DATA_ROOT"), extractor_name)

    dis_utc = parse_dt(data_interval_start_utc)
    die_utc = parse_dt(data_interval_end_utc)

    if not isinstance(dis_utc, DateTime):
        raise TypeError("data_interval_start_utc must parse to a datetime.")

    if not isinstance(die_utc, DateTime):
        raise TypeError("data_interval_end_utc must parse to a datetime.")

    files = get_files_created_between(data_dirpath, dis_utc, die_utc)

    logging.info(
        "Found %s files in %s modified between %s and %s.",
        len(files),
        data_dirpath,
        data_interval_start_utc,
        data_interval_end_utc,
    )

    if len(files) == 0:
        raise RuntimeError("No files created for that interval found.")

    return files


@task
def autolabel_task(
    model_config: dict[str, Any],
    class_mapping: dict[int, str],
    extractor_name: str,
    images_to_label: list[str],
    data_interval_end_utc: str,
):
    from dags.ydisk_loader.utils.yolov5_client import (
        YOLOv5DetectionsImposter,
        get_image_sizes,
        get_batches,
        process_images,
        yolov5_detections_to_coco,
    )
    from tqdm import tqdm

    date = parse_dt(data_interval_end_utc).in_tz(TIMEZONE).date().isoformat()
    fname = f"{date}_{extractor_name}.json"

    labels_root = Variable.get("LABELS_ROOT")
    extractor_labels_dir = os.path.join(labels_root, extractor_name)
    os.makedirs(extractor_labels_dir, exist_ok=True)

    fpath = os.path.join(extractor_labels_dir, fname)
    if os.path.exists(fpath):
        raise FileExistsError(f"File already exists: {fpath}")

    yolov5_results = YOLOv5DetectionsImposter(class_mapping=class_mapping)

    logging.info("Images to label: %s", len(images_to_label))

    max_batch_size = model_config["max_batch_size"]
    stride = model_config["stride"]

    image_sizes = get_image_sizes(images_to_label)
    size_batches = get_batches(image_sizes, model_stride=stride, max_batch_size=max_batch_size)

    logging.info("Batching statistics:")
    logging.info("Batch size: %s", max_batch_size)
    for (h, w), batches in size_batches.items():
        logging.info(
            "Size (%s, %s): %s batches, < %s images",
            h,
            w,
            len(batches),
            len(batches) * max_batch_size,
        )

    for (h, w), batches in size_batches.items():
        for image_filepaths in tqdm(batches):
            results = process_images(image_filepaths, **model_config)
            yolov5_results.extend(
                filepaths=[os.path.basename(i) for i in image_filepaths],
                img_sizes=[image_sizes[i] for i in image_filepaths],
                detection_results=results,
            )

    coco_json = yolov5_detections_to_coco(yolov5_results)

    with open(fpath, "w") as fp:
        json.dump(coco_json, fp)

    logging.info("Labels saved as %s", fpath)

    return fpath


@task
def archive_task(extractor_name: str, files_to_archive: list[str], data_interval_end_utc: str):
    from dags.ydisk_loader.utils.archiver import Archiver

    archives_root = Variable.get("ARCHIVES_ROOT")

    if not os.path.exists(archives_root):
        raise RuntimeError("Make sure directory %s exists", archives_root)

    archiver = Archiver(extractor_name, archives_root, TIMEZONE)

    die_utc = parse_dt(data_interval_end_utc)

    if not isinstance(die_utc, DateTime):
        raise TypeError("data_interval_end_utc must parse to a datetime.")

    archive_path = archiver.archive(
        files_to_archive=files_to_archive, data_interval_end_utc=die_utc
    )

    logging.info(f"Successfully created archive {archive_path}")

    return archive_path


@task
def yadisk_upload_task(filepath: str, yadisk_save_dir: str):
    from dags.ydisk_loader.utils.ydisk_utils import yadisk_upload

    logging.info(f"Trying to upload {filepath}")

    ydisk_file_path = yadisk_upload(
        filepath=filepath,
        yadisk_save_dir=yadisk_save_dir,
        token=Variable.get("YDISK_TOKEN"),
        timeout=(
            int(Variable.get("YDISK_UPLOAD_WRITE_TIMEOUT")),
            int(Variable.get("YDISK_UPLOAD_READ_TIMEOUT")),
        ),
    )

    logging.info(f"Successfully uploaded as {ydisk_file_path}")


@dag(
    schedule_interval="0 0 * * 1",
    start_date=datetime(2022, 12, 1, tz=TIMEZONE),
    is_paused_upon_creation=True,
    catchup=True,
    max_active_runs=1,
    tags=["ydisk_loader"],
)
def yadisk_loader_dag():
    with open("/opt/airflow/dags/image_collector/image_collector_config.json", "r") as fp:
        config = json.load(fp)

    with open("/opt/airflow/dags/ydisk_loader/autolabeler_config.json", "r") as fp:
        autolabeler_config = json.load(fp)

    for extractor_config in config["extractors"]:
        extractor_name = extractor_config["params"]["name"]
        yadisk_save_dir = extractor_config["upload_path"]

        selected_files = select_files_task.override(task_id=f"{extractor_name}_select")(
            extractor_name=extractor_name,
            data_interval_start_utc="{{ data_interval_start }}",
            data_interval_end_utc="{{ data_interval_end }}",
        )

        archive_filepath = archive_task.override(task_id=f"{extractor_name}_archive")(
            extractor_name=extractor_name,
            files_to_archive=selected_files,
            data_interval_end_utc="{{ data_interval_end }}",
        )

        labels_filepath = autolabel_task.override(
            task_id=f"{extractor_name}_autolabel",
            pool="autolabeling",
            pool_slots=1,
        )(
            extractor_name=extractor_name,
            images_to_label=selected_files,
            data_interval_end_utc="{{ data_interval_end }}",
            **autolabeler_config,
        )

        yadisk_upload_task.override(task_id=f"{extractor_name}_upload_images")(
            filepath=archive_filepath, yadisk_save_dir=yadisk_save_dir
        )

        yadisk_upload_task.override(task_id=f"{extractor_name}_upload_annotations")(
            filepath=labels_filepath, yadisk_save_dir=yadisk_save_dir
        )


yadisk_loader_dag()
