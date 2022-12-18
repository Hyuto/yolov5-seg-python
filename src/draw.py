from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt


def draw_boxes(
    source: npt.NDArray[np.uint8],
    box: npt.NDArray[np.int32],
    label: str,
    score: float,
    color: Tuple[int, int, int],
) -> None:
    """Draw boxes on images

    Args:
        source (npt.NDArray[np.uint8]): image array
        box (npt.NDArray[np.int32]): box to draw [left, top, width, height]
        label (str): box label
        score (float): box score
        color (Tuple[int, int, int]): box color in rgb format
    """
    cv2.rectangle(source, box, color, 2)  # draw box
    (label_width, label_height), _ = cv2.getTextSize(
        f"{label} - {round(score, 2)}",
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    cv2.rectangle(
        source,
        (box[0] - 1, box[1] - label_height - 6),
        (box[0] + label_width + 1, box[1]),
        color,
        -1,
    )
    cv2.putText(
        source,
        f"{label} - {round(score, 2)}",
        (box[0], box[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        [255, 255, 255],
        1,
    )


class Colors:
    """Ultralytics color palette https://ultralytics.com/"""

    def __init__(self) -> None:
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i: int, bgr: bool = False) -> Tuple[int, int, int]:
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> Tuple[int, int, int]:  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()
