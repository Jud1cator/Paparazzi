# Paparazzi - Airflow DAGs to collect image datasets from RTSP streams
## Configuration
### Extract
Currently system is configured via JSON in following format:
```
{
    "extractors": [
        {
            "name": "ClassName",
            "params": {...}
        },
        ....
    ]
}
```

`extractors` holds the list of all extractors which will be configured and added as separate tasks in Airflow DAG. Currently implemented extractors:

-  `RTSPStreamExtractor`, which periodically dumps images from RTSP stream.
- `RSyncExtractor`, which periodically uses `rsync` to copy files from remote host. Path to ssh key file must be specified in `SSH_KEY_FILE` variable.

## To run locally:
1. `mkdir -p ./logs ./plugins`
2. `echo -e "AIRFLOW_UID=$(id -u)" > .env`
3. `export PAPARAZZI_DATA_ROOT={path to root directory for data}`
4. `export SSH_KEY_FILE={path to ssh key file for RSyncExtractor}`
5. `docker compose up -d`
6. Set Airflow variables

## Aiflow variables
```
ARCHIVES_ROOT=/raw_data/archived
DATA_ROOT=/raw_data/images
ANNOTATIONS_ROOT=/raw_data/annotations
RTSPSTREAMEXTRACTOR_TIMEOUT=60
YDISK_TOKEN=********
YDISK_UPLOAD_READ_TIMEOUT=600
YDISK_UPLOAD_WRITE_TIMEOU=60
```
