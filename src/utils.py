from pathlib import Path
from typing import Tuple, Iterable

import numpy as np
import numpy.typing as npt


def check_file(path: str, message: str) -> None:
    """Check if file is exist or not.

    Args:
        path (str): File path
        message (str): Message if file is not exist

    Raises:
        FileNotFoundError: If file is not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(message)


def get_divable_size(imgsz: Iterable[int], stride: int) -> Iterable[int]:
    """Get divable image size by model stride

    Args:
        imgsz (Iterable[int]): Current image size [width, height]
        stride (int): Model stride

    Returns:
        Divable image size by model stride
    """
    for i in range(len(imgsz)):
        div, mod = divmod(imgsz[i], stride)
        if mod > stride / 2:
            div += 1
        imgsz[i] = div * stride
    return imgsz


def handle_overflow_box(
    box: npt.NDArray[np.int32], imgsz: Tuple[int, int]
) -> npt.NDArray[np.int32]:
    """Handle if box contain overflowing coordinate based on image size

    Args:
        box (npt.NDArray[np.int32]): box to draw [left, top, width, height]
        imgsz (Tuple[int, int]): Current image size [width, height]

    Returns:
        Non overflowing box
    """
    if box[0] < 0:
        box[0] = 0
    elif box[0] >= imgsz[0]:
        box[0] = imgsz[0] - 1
    if box[1] < 0:
        box[1] = 0
    elif box[1] >= imgsz[1]:
        box[1] = imgsz[1] - 1
    box[2] = box[2] if box[0] + box[2] <= imgsz[0] else imgsz[0] - box[0]
    box[3] = box[3] if box[1] + box[3] <= imgsz[1] else box[3] - box[1]
    return box


# fmt: off
labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush",
]
# fmt: on
