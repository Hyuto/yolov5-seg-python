from pathlib import Path


def check_file(path, message):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(message)


def get_divable_size(imgsz, stride):
    for i in range(len(imgsz)):
        div, mod = divmod(imgsz[i], stride)
        if mod > stride / 2:
            div += 1
        imgsz[i] = div * stride
    return imgsz


def handle_overflow_box(box, imgsz):
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


def get_video_size(size, max_size):
    if max_size == -1 or (size[0] <= max_size[0] and size[1] <= max_size[1]):
        return list(map(lambda x: int(round(x)), size))

    x_ratio, y_ratio = max_size[0] / size[0], max_size[1] / size[1]

    if x_ratio < 1 and y_ratio < 1:
        _min = min(x_ratio, y_ratio)
        new_size = [size[0] * _min, size[1] * _min]
    elif x_ratio < 1:
        new_size = [max_size[0], size[1] * x_ratio]
    else:
        new_size = [size[0] * y_ratio, max_size[1]]
    return list(map(lambda x: int(round(x)), new_size))


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
