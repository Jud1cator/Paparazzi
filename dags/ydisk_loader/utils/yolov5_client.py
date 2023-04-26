from __future__ import annotations

import json
import math
import os
from argparse import ArgumentParser
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from tqdm import tqdm


class YOLOv5DetectionsImposter:
    def __init__(
        self,
        class_mapping: Optional[dict[int, str]] = None,
        filepaths: Optional[list[str]] = None,
        img_sizes: Optional[list[tuple[int, int]]] = None,
        detection_results: Optional[list[np.ndarray]] = None,
    ) -> None:
        self.names = {} if class_mapping is None else {int(k): v for k, v in class_mapping.items()}
        self.files = [] if filepaths is None else filepaths
        self.img_sizes = [] if img_sizes is None else img_sizes
        self.xyxy = [] if detection_results is None else detection_results

    def extend(
        self,
        class_mapping: Optional[dict[int, str]] = None,
        filepaths: Optional[list[str]] = None,
        img_sizes: Optional[list[tuple[int, int]]] = None,
        detection_results: Optional[list[np.ndarray]] = None,
    ) -> None:
        if class_mapping is not None:
            self.names.update(class_mapping)

        if filepaths is not None:
            self.files.extend(filepaths)

        if img_sizes is not None:
            self.img_sizes.extend(img_sizes)

        if detection_results is not None:
            self.xyxy.extend(detection_results)


def read_image(image_path: str):
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def get_image_sizes(images: list[str]) -> dict[str, tuple[int, int]]:
    return {i: Image.open(i).size[::-1] for i in images}


def get_batches(
    image_sizes: dict[str, tuple[int, int]], model_stride: int = 64, max_batch_size: int = 32
) -> dict[tuple[int, int], list[list[str]]]:
    """
    Parses list of image file paths to dictionary of preprocessed image size to list of batches:
    [img1, img2, ...] -> {(height, width): [[img1, img2, ...], ...], ...}
    """
    sizes = list(
        map(
            lambda x: tuple(
                map(lambda y: make_divisible(y, model_stride), image_sizes[x])
            ),
            image_sizes,
        )
    )
    equal_size_images = {}
    for size, img in zip(sizes, image_sizes.keys()):
        if size not in equal_size_images:
            equal_size_images[size] = [[img]]
        elif len(equal_size_images[size][-1]) < max_batch_size:
            equal_size_images[size][-1].append(img)
        else:
            equal_size_images[size].append([img])
    return equal_size_images


class PaddingMode(str, Enum):
    CENTER = "center"
    TOP_LEFT = "top_left"


def letterbox(
    im,
    new_shape,
    color=(0, 0, 0),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
    padding_mode: PaddingMode = PaddingMode.TOP_LEFT,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    if padding_mode == PaddingMode.CENTER:
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    elif padding_mode == PaddingMode.TOP_LEFT:
        top, bottom, left, right = 0, dh, 0, dw
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode}")
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def preprocess(image, model_stride=64, det_res=None):
    img_res = image.shape[:2]  # (h, w)
    if det_res is not None:
        g = max(det_res) / max(img_res)  # gain
        ar_preserved_res = tuple(int(i * g) for i in img_res)  # (h, w)
        image = cv2.resize(image, (ar_preserved_res[1], ar_preserved_res[0]))
    else:
        ar_preserved_res = img_res  # (h, w)
    ar_preserved_strided_res = tuple(
        make_divisible(i, model_stride) for i in ar_preserved_res
    )  # (h, w)
    image, ratio = letterbox(
        image, ar_preserved_strided_res, stride=model_stride, padding_mode=PaddingMode.TOP_LEFT
    )
    image = image / 255.0
    return np.expand_dims(image.transpose(2, 0, 1), 0).astype(np.float32), ratio


def fix_yolov5_detection_output(res, ratio: tuple[float, float]):
    fixed_res = np.zeros((res.shape[0], res.shape[1], 6), dtype=np.float64)
    fixed_res[:, :, :2] = res[:, :, :2] - res[:, :, 2:4] / 2
    fixed_res[:, :, :4] = fixed_res[:, :, :4] / (*ratio, *ratio)
    fixed_res[:, :, 2:5] = res[:, :, 2:5]
    fixed_res[:, :, 5] = np.argmax(res[:, :, 5:], axis=2)
    return fixed_res


def filter_yolov5_fixed_output(res, confidence_threshold, nms_iou_threshold):
    batches = []
    for i in range(res.shape[0]):
        # Filter by confidence
        idx = np.argwhere(res[i, :, 4] > confidence_threshold).ravel()
        # Filter by NMS
        nms_idx = cv2.dnn.NMSBoxes(res[i, idx, :4].tolist(), res[i, idx, 4], 0, nms_iou_threshold)
        nms_idx = np.array(nms_idx, dtype=int).ravel()
        idx = idx[nms_idx]
        res[i, idx, 2:4] = res[i, idx, :2] + res[i, idx, 2:4]  # xywh to xyxy
        batches.append(res[i, idx])
    return batches


def yolov5_detections_to_coco(yolov5_detections, conf_thresh=0.0):
    """
    Parse YOLOv5 Detections class
    (https://github.com/ultralytics/yolov5/blob/master/models/common.py#L721) to COCO JSON
    """
    coco_annos = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }

    for class_id, class_name in yolov5_detections.names.items():
        coco_annos["categories"].append({"id": class_id, "name": class_name, "supercategory": ""})

    for img_id, (img_path, (img_h, img_w), detections) in enumerate(
        zip(yolov5_detections.files, yolov5_detections.img_sizes, yolov5_detections.xyxy)
    ):
        coco_annos["images"].append(
            {
                "id": img_id,
                "file_name": img_path,
                "width": img_w,
                "height": img_h,
                "license": None,
                "coco_url": None,
            }
        )

        for det_id, det in enumerate(detections):
            x, y, x2, y2, conf, class_id = map(lambda x: round(x.item(), 2), det)

            if conf < conf_thresh:
                continue

            w = round(x2 - x, 2)
            h = round(y2 - y, 2)
            coco_annos["annotations"].append(
                {
                    "id": det_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "segmentation": [],
                    "area": w * h,
                    "bbox": [x, y, w, h],
                    "score": conf,
                    "iscrowd": 0,
                    "attributes": {
                        "undefined": False,
                        "occluded": False,
                        "rotation": 0.0,
                    },
                }
            )

    return coco_annos


def process_images(
    images: list[str],
    server_url: str,
    model_name: str,
    input_layer: str,
    output_layer: str,
    confidence_threshold: float,
    nms_iou_threshold: float,
    **kwargs,
) -> list[np.ndarray]:
    # Preparing image for inference
    batch = []
    for img_path in images:
        source_image = read_image(img_path)
        preprocessed_image, ratio = preprocess(source_image)
        batch.append(preprocessed_image)
    batch = np.concatenate(batch, axis=0)

    # Creating inputs/outputs for client request
    infer_input = grpcclient.InferInput(input_layer, batch.shape, datatype="FP32")
    infer_input.set_data_from_numpy(batch)
    output = grpcclient.InferRequestedOutput(output_layer)

    # Performing client request
    client = grpcclient.InferenceServerClient(url=server_url)
    results = client.infer(model_name=model_name, inputs=[infer_input], outputs=[output])
    inference_output = results.as_numpy(output_layer)
    if inference_output is None:
        raise RuntimeError("Unprocessed error: InferResult.as_numpy returned None.")

    # Output from grpc client is read-only for some reason, so we create a copy to modify it
    inference_output = inference_output.copy()

    # Postprocessing
    yolov5_fixed_output = fix_yolov5_detection_output(inference_output, ratio)
    filtered_output = filter_yolov5_fixed_output(
        yolov5_fixed_output, confidence_threshold, nms_iou_threshold
    )

    return filtered_output


def main(args):
    # TODO can we get class mapping from model via triton client?
    yolov5_results = YOLOv5DetectionsImposter(class_mapping={0: "vehicle", 1: "person"})

    if os.path.isfile(args.input):
        images = [args.input]
    elif os.path.isdir(args.input):
        images = list(map(lambda x: os.path.join(args.input, x), os.listdir(args.input)))
    else:
        raise ValueError(
            f"{args.input} must be a path to image file or a path to directory with images."
        )

    size_batches = get_batches(
        images, model_stride=args.model_stride, max_batch_size=args.max_batch_size
    )

    for size, batches in size_batches.items():
        w, h = size
        for batch in tqdm(batches):
            results = process_images(batch, **vars(args))
            yolov5_results.extend(
                filepaths=list(map(lambda x: os.path.basename(x), batch)),
                img_sizes=[(h, w)] * len(batch),
                detection_results=results,
            )

    coco = yolov5_detections_to_coco(yolov5_results)
    with open(args.output, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input image or directory with images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="coco.json",
        help="Path to output COCO annotation file.",
    )
    parser.add_argument(
        "-s",
        "--server-url",
        required=True,
        help="Inference serverl URL.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="yolov5l_vehicle_person_autolabeler",
        help="Name of the model deployed on the inference server.",
    )
    parser.add_argument("--input-layer", default="images", help="Input layer name.")
    parser.add_argument("--output-layer", default="output0", help="Output layer name.")
    parser.add_argument("--class-count", default=2, help="Count of classes which model predicts.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Boxes with objectiveness score lower than this threshold will be filtered out.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help=(
            "If IoU between 2 boxes is higher than this threshold, the box with smaller"
            " objectiveness score will be filtered out."
        ),
    )
    parser.add_argument(
        "--model-stride",
        type=int,
        default=64,
        help="Maximum model anchor strided which specifies the divisor of each image dimension.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum batch size for an inference request.",
    )

    args = parser.parse_args()

    main(args)
