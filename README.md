# YOLOv5 Segmentation Python

![horse](./assets/horse.png)

---

Run yolov5 segmentation model on _onnxruntime_ or _opencv dnn_ **without torch**!

## Usage

All you need is on `segment.py`, it provides cli to run yolov5-seg onnx model.

```bash
python segment.py -- help
```

### Image

|                                |                          |
| :----------------------------: | :----------------------: |
| ![zidane](./assets/zidane.png) | ![bus](./assets/bus.png) |

```bash
python segment.py -m <YOLOV5-SEG-ONNX-PATH> -i <IMAGE-PATH>
```

### Video

|                              |                                |
| :--------------------------: | :----------------------------: |
| ![horse](./assets/horse.gif) | ![street](./assets/street.gif) |

```bash
python segment.py -m <YOLOV5-SEG-ONNX-PATH> -v 0            # webcam
                                               <VIDEO-PATH> # local video
```

**Note** : Press `q` to stop video processing.

## OpenCV DNN

Use Opencv DNN as backend with `--dnn` arguments.

```bash
python segment.py -m <YOLOV5-SEG-ONNX-PATH> -v 0 --dnn
```

## Run on GPU

Auto using gpu to run model when devices is supported.

- `onnxruntime` need `onnxruntime-gpu` to be installed.

  **Note** : `onnxruntime-gpu` must be installed with the same version as `onnxruntime` to be able to use GPU.

- `opencv-dnn` need custom build.

```bash
pip install onnxruntime-gpu==<YOUR-ONNXRUNTIME-VERSION>
```

## Reference

- https://github.com/ultralytics/yolov5
- https://github.com/UNeedCryDear/yolov5-seg-opencv-dnn-cpp
