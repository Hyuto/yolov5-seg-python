import argparse

import numpy as np
import cv2

from src.models import ORTModelLoader, DNNModelLoader
from src.general import run_yolov5_seg
from src.utils import check_file


def parse_opt():
    parser = argparse.ArgumentParser(description="Detect using YOLOv5 Segmentation model")
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-m", "--model", type=str, required=True, help="YOLOv5 Segmentation onnx model path"
    )
    source = parser.add_argument_group("source arguments")
    source.add_argument("-i", "--image", type=str, help="Image source")
    source.add_argument("-v", "--video", type=str, help="Video source")

    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Integer representing the maximum number of boxes to be selected per class",
    )
    parser.add_argument(
        "--conf-tresh",
        type=float,
        default=0.2,
        help="Float representing the threshold for deciding when to remove boxes based on confidence score",
    )
    parser.add_argument(
        "--iou-tresh",
        type=float,
        default=0.45,
        help="Float representing the threshold for deciding whether boxes overlap too much with respect to IOU",
    )
    parser.add_argument(
        "--score-tresh",
        type=float,
        default=0.25,
        help="Float representing the threshold for deciding whether render boxes or not",
    )
    parser.add_argument(
        "--mask-tresh",
        type=float,
        default=0.5,
        help="Float representing the threshold for deciding mask area",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.4,
        help="Float representing the opacity of mask layer",
    )
    parser.add_argument(
        "--dnn",
        action="store_true",
        help="Use OpenCV DNN module [if false using onnxruntime] for backend",
    )

    opt = parser.parse_args()
    if opt.image is None and opt.video is None:
        raise argparse.ArgumentError("Please specify image or video source!")
    elif opt.image and opt.video:
        raise argparse.ArgumentError("Please specify either image or video source!")
    return opt


def main(opt) -> None:
    if opt.dnn:
        model = DNNModelLoader(opt.model)  # use Opencv DNN module
    else:
        model = ORTModelLoader(opt.model)  # use onnxruntime

    # warmup model
    _ = run_yolov5_seg(
        model,
        (np.random.rand(model.width, model.height, 3) * 255).astype(np.uint8),  # random image
        opt.conf_tresh,
        opt.iou_tresh,
        opt.score_tresh,
        opt.topk,
        opt.mask_tresh,
        opt.mask_alpha,
    )

    if opt.image:
        check_file(opt.image, "Image file not found!")

        # Image preprocessing
        img = cv2.imread(opt.image)
        img = run_yolov5_seg(
            model,
            img,
            opt.conf_tresh,
            opt.iou_tresh,
            opt.score_tresh,
            opt.topk,
            opt.mask_tresh,
            opt.mask_alpha,
        )

        cv2.imshow("output", img)
        cv2.waitKey(0)
    elif opt.video:
        # Video processing
        vid_source = 0 if opt.video == "0" else opt.video
        cap = cv2.VideoCapture(vid_source)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = run_yolov5_seg(
                model,
                frame,
                opt.conf_tresh,
                opt.iou_tresh,
                opt.score_tresh,
                opt.topk,
                opt.mask_tresh,
                opt.mask_alpha,
            )

            cv2.imshow("output", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
