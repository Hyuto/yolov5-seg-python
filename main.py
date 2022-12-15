import ast
import numpy as np
import cv2

import onnxruntime as ort

from utils import handle_overflow_box, Colors


img_path = "./images/horse.jpg"
model_path = "./models/yolov5s-seg.onnx"

colors = Colors()
conf_tresh = 0.25
iou_tresh = 0.45
mask_tresh = 0.5


if __name__ == "__main__":
    model = ort.InferenceSession(model_path)
    _, _, model_width, model_height = model.get_inputs()[0].shape
    _, seg_chanels, seg_width, seg_height = model.get_outputs()[1].shape
    labels = ast.literal_eval(model.get_modelmeta().custom_metadata_map["names"])

    img = cv2.imread(img_path)
    overlay = img.copy()
    img_height, img_width, _ = img.shape

    x_ratio = img_width / model_width
    y_ratio = img_height / model_height

    input_img = cv2.dnn.blobFromImage(
        img,
        1 / 255.0,
        (model_width, model_height),
        swapRB=True,
        crop=False,
    )
    result = model.run(None, {"images": input_img})

    # box preprocessing
    result[0][0, :, 0] = (result[0][0, :, 0] - 0.5 * result[0][0, :, 2]) * x_ratio
    result[0][0, :, 1] = (result[0][0, :, 1] - 0.5 * result[0][0, :, 3]) * y_ratio
    result[0][0, :, 2] *= x_ratio
    result[0][0, :, 3] *= y_ratio

    boxes = result[0][0, :, :4]
    confidences = result[0][0, :, 4]
    scores = confidences.reshape(-1, 1) * result[0][0, :, 5:85]
    masks = result[0][0, :, 85:]

    selected = cv2.dnn.NMSBoxes(boxes, confidences, conf_tresh, iou_tresh)

    for i in selected:
        box = boxes[i].round().astype(np.int32)  # to int
        box = handle_overflow_box(box, [img_width, img_height])

        _, score, _, label = cv2.minMaxLoc(scores[i])
        color = colors(label[1], True)
        cv2.rectangle(img, box, color, 2)
        cv2.rectangle(img, (box[0] - 1, box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(
            img,
            f"{labels[label[1]]} - {round(score, 2)}",
            (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            [255, 255, 255],
        )

        # crop mask from proto
        x = box[0] * seg_width // img_width
        y = box[1] * seg_height // img_height
        w = box[2] * seg_width // img_width
        h = box[3] * seg_height // img_height

        protos = result[1][0, :, y : y + h, x : x + w].reshape(seg_chanels, w * h)  # get protos
        protos = np.expand_dims(masks[i], 0) @ protos  # matmul
        protos = protos.reshape(h, w)  # reshape
        protos = 1 / (1 + np.exp(-protos))  # sigmoid
        mask = cv2.resize(protos, (box[2], box[3]))
        mask = mask >= mask_tresh

        to_mask = overlay[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
        to_mask[mask] = color

    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
