from typing import Union

import cv2
import numpy as np
import numpy.typing as npt

from .models import ORTModelLoader, DNNModelLoader
from .draw import draw_boxes, colors
from .utils import get_divable_size, handle_overflow_box


def run_yolov5_seg(
    model: Union[ORTModelLoader, DNNModelLoader],
    source: npt.NDArray[np.uint8],
    conf_tresh: float,
    iou_tresh: float,
    score_tresh: float,
    topk: int,
    mask_tresh: float,
    mask_alpha: float,
) -> npt.NDArray[np.uint8]:
    """Run YOLOv5 Segmentation model

    Args:
        model (Union[ORTModelLoader, DNNModelLoader]): Model loader
        source (npt.NDArray[np.uint8]): Source array
        conf_tresh (float): Confidences treshold
        iou_tresh (float): IoU or NMS treshold
        score_tresh (float): Scores treshold
        topk (int): TopK classes
        mask_tresh (float): Mask treshold
        mask_alpha (float): Mask opacity on overlay
    """
    source_height, source_width, _ = source.shape

    ## resize to divable size by stride
    source_width, source_height = get_divable_size([source_width, source_height], model.stride)
    source = cv2.resize(source, [source_width, source_height])

    ## padding image
    max_size = max(source_width, source_height)  # get max size
    source_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)  # initial zeros mat
    source_padded[:source_height, :source_width] = source.copy()  # place original image
    overlay = source_padded.copy()  # make overlay mat

    ## ratios
    x_ratio = max_size / model.width
    y_ratio = max_size / model.height

    # run model
    input_img = cv2.dnn.blobFromImage(
        source_padded,
        1 / 255.0,
        (model.width, model.height),
        swapRB=False,
        crop=False,
    )  # normalize and resize: [h, w, 3] => [1, 3, h, w]
    result = model.forward(input_img)

    # box preprocessing
    result[0][0, :, 0] = (result[0][0, :, 0] - 0.5 * result[0][0, :, 2]) * x_ratio
    result[0][0, :, 1] = (result[0][0, :, 1] - 0.5 * result[0][0, :, 3]) * y_ratio
    result[0][0, :, 2] *= x_ratio
    result[0][0, :, 3] *= y_ratio

    # get boxes, conf, score, and mask
    boxes = result[0][0, :, :4]
    confidences = result[0][0, :, 4]
    scores = confidences.reshape(-1, 1) * result[0][0, :, 5 : len(model.labels) + 5]
    masks = result[0][0, :, len(model.labels) + 5 :]

    # NMS
    selected = cv2.dnn.NMSBoxes(boxes, confidences, conf_tresh, iou_tresh, top_k=topk)

    boxes_to_draw = []  # boxes to draw

    for i in selected:  # loop through selected
        box = boxes[i].round().astype(np.int32)  # to int
        box = handle_overflow_box(box, [max_size, max_size])  # handle overflow boxes

        _, score, _, label = cv2.minMaxLoc(scores[i])  # get score and classId
        if score >= score_tresh:  # filtering by score_tresh
            color = colors(label[1], True)  # get color

            # save box to draw latter (add mask first)
            boxes_to_draw.append([box, model.labels[label[1]], score, color])

            # crop mask from proto
            x = int(round(box[0] * model.seg_width / max_size))
            y = int(round(box[1] * model.seg_height / max_size))
            w = int(round(box[2] * model.seg_width / max_size))
            h = int(round(box[3] * model.seg_height / max_size))

            # process protos
            protos = result[1][0, :, y : y + h, x : x + w].reshape(model.seg_chanels, -1)
            protos = np.expand_dims(masks[i], 0) @ protos  # matmul
            protos = 1 / (1 + np.exp(-protos))  # sigmoid
            protos = protos.reshape(h, w)  # reshape
            mask = cv2.resize(protos, (box[2], box[3]))  # resize mask
            mask = mask >= mask_tresh  # filtering mask by tresh

            # add mask to overlay layer
            to_mask = overlay[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]  # get box roi
            mask = mask[: to_mask.shape[0], : to_mask.shape[1]]  # crop mask
            to_mask[mask] = color  # apply mask

    # combine image and overlay
    source_padded = cv2.addWeighted(source_padded, 1 - mask_alpha, overlay, mask_alpha, 0)

    for draw_box in boxes_to_draw:  # draw boxes
        draw_boxes(source_padded, *draw_box)

    source = source_padded[:source_height, :source_width]  # crop padding

    return source
