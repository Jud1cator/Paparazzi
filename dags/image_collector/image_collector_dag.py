import json

import pendulum
from airflow.decorators import dag, task

from image_collector.extractors.builder import build_extractor


@task
def get_images(config_filepath: str):
    with open(config_filepath, "r") as fp:
        config = json.load(fp)

    for extractor_config in config["extractors"]:
        build_extractor(**extractor_config).extract()


@dag(
    schedule_interval="0/5 * * * *",
    start_date=pendulum.datetime(2022, 12, 23, tz="Europe/Moscow"),
    is_paused_upon_creation=True,
    catchup=False,
    tags=["image_collector"],
)
def image_collector_dag():
    get_images_task = get_images.override(task_id="get_images")(
        config_filepath="/opt/airflow/dags/image_collector/image_collector_config.json",
    )


image_collector_dag()
