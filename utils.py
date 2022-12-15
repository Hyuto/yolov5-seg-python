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


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
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

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
