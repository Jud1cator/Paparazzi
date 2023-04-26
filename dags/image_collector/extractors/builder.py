from typing import Any, Dict

from image_collector.extractors.rtsp_stream_extractor import RTSPStreamExtractor
from image_collector.extractors.rsync_extractor import RSyncExtractor

SOURCES = {"RTSPStreamExtractor": RTSPStreamExtractor, "RSyncExtractor": RSyncExtractor}


def build_extractor(name: str, params: Dict[str, Any], **kwargs):
    return SOURCES[name](**params)
