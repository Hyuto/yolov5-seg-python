import ast


import numpy as np
import onnxruntime as ort

from .utils import check_file, labels


class ModelLoader:
    def __init__(self, path) -> None:
        self._load_model(path)
        self._get_metadata()

    def __call__(self, *args, **kwargs):
        return self.model.run(*args, **kwargs)

    def _load_model(self, model_path):
        check_file(model_path, "Model is not exist!")

        device = ort.get_device()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "GPU"
            else ["CPUExecutionProvider"]
        )
        self.model = ort.InferenceSession(model_path, providers=providers)
        bs, c, self.width, self.height = self.model.get_inputs()[0].shape
        _, self.seg_chanels, self.seg_width, self.seg_height = self.model.get_outputs()[1].shape

        # warmup model
        for _ in range(3):
            self.model.run(
                None,
                {"images": np.random.rand(bs, c, self.width, self.height).astype(np.float32)},
            )

    def _get_metadata(self, default_labels=labels, default_stride=32):
        metadata = self.model.get_modelmeta().custom_metadata_map
        self.labels = ast.literal_eval(metadata["names"]) if "names" in metadata else default_labels
        self.stride = (
            ast.literal_eval(metadata["stride"]) if "stride" in metadata else default_stride
        )
