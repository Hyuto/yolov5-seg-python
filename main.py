import ast

import numpy as np
import cv2
import onnxruntime as ort

from utils import draw_boxes, get_divable_size, Colors


img_path = "./images/horse.jpg"
model_path = "./models/yolov5s-seg.onnx"

colors = Colors()
score_tresh = 0.25
conf_tresh = 0.2
iou_tresh = 0.45
mask_tresh = 0.45


if __name__ == "__main__":
    model = ort.InferenceSession(model_path)
    _, _, model_width, model_height = model.get_inputs()[0].shape
    _, seg_chanels, seg_width, seg_height = model.get_outputs()[1].shape
    metadata = model.get_modelmeta().custom_metadata_map
    labels, stride = ast.literal_eval(metadata["names"]), ast.literal_eval(metadata["stride"])

    # Image preprocessing
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    ## resize to divable image size by stride
    img_width, img_height = get_divable_size([img_width, img_height], stride)
    img = cv2.resize(img, [img_width, img_height])

    ## padding image
    max_size = max(img_width, img_height)  # get max size
    img_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)  # initial zeros img mat
    img_padded[:img_height, :img_width] = img.copy()  # place original image
    overlay = img_padded.copy()  # make overlay mat

    ## ratios
    x_ratio = max_size / model_width
    y_ratio = max_size / model_height

    # run model
    input_img = cv2.dnn.blobFromImage(
        img_padded,
        1 / 255.0,
        (model_width, model_height),
        swapRB=True,
        crop=False,
    )  # normalize, resize and swapRB : [h, w, 3] => [1, 3, w, h]
    result = model.run(None, {"images": input_img})

    # box preprocessing
    result[0][0, :, 0] = (result[0][0, :, 0] - 0.5 * result[0][0, :, 2]) * x_ratio
    result[0][0, :, 1] = (result[0][0, :, 1] - 0.5 * result[0][0, :, 3]) * y_ratio
    result[0][0, :, 2] *= x_ratio
    result[0][0, :, 3] *= y_ratio

    # get boxes, conf, score, and mask
    boxes = result[0][0, :, :4]
    confidences = result[0][0, :, 4]
    scores = confidences.reshape(-1, 1) * result[0][0, :, 5:85]
    masks = result[0][0, :, 85:]

    # NMS
    selected = cv2.dnn.NMSBoxes(boxes, confidences, conf_tresh, iou_tresh)

    boxes_to_draw = []  # boxes to draw

    for i in selected:  # loop through selected
        box = boxes[i].round().astype(np.int32)  # to int

        _, score, _, label = cv2.minMaxLoc(scores[i])  # get score and classId
        if score >= score_tresh:  # filtering by score_tresh
            color = colors(label[1], True)  # get color

            # save box to draw latter (add mask first)
            boxes_to_draw.append([box, labels[label[1]], score, color])

            # crop mask from proto
            x = int(round(box[0] * seg_width / max_size))
            y = int(round(box[1] * seg_height / max_size))
            w = int(round(box[2] * seg_width / max_size))
            h = int(round(box[3] * seg_height / max_size))

            # process protos
            protos = result[1][0, :, y : y + h, x : x + w].reshape(seg_chanels, w * h)  # get protos
            protos = np.expand_dims(masks[i], 0) @ protos  # matmul
            protos = protos.reshape(h, w)  # reshape
            protos = 1 / (1 + np.exp(-protos))  # sigmoid
            mask = cv2.resize(protos, (box[2], box[3]))  # resize mask
            mask = mask >= mask_tresh  # filtering mask by tresh

            # add mask to overlay layer
            to_mask = overlay[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
            to_mask[mask] = color

    img_padded = cv2.addWeighted(img_padded, 0.7, overlay, 0.3, 0)  # combine image and overlay

    for draw_box in boxes_to_draw:  # draw boxes
        draw_boxes(img_padded, *draw_box)

    img = img_padded[:img_height, :img_width]  # crop padding

    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
