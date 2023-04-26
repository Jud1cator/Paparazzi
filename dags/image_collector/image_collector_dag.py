import json

import pendulum
from airflow.decorators import dag

from image_collector.extractors.builder import build_extractor


@dag(
    schedule_interval="0/5 * * * *",
    start_date=pendulum.datetime(2022, 12, 23, tz="Europe/Moscow"),
    is_paused_upon_creation=True,
    catchup=False,
    tags=["image_collector"],
)
def image_collector_dag():
    with open("/opt/airflow/dags/image_collector/image_collector_config.json", "r") as fp:
        config = json.load(fp)

    for source_conf in config["extractors"]:
        build_extractor(**source_conf).build_operator()


image_collector_dag()
