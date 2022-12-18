import ast
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from .utils import check_file, labels


class ORTModelLoader:
    """ONNXRUNTIME model handler"""

    def __init__(self, path: str) -> None:
        self._load_model(path)
        self._get_metadata()

    def _load_model(self, model_path: str) -> None:
        """Load model and get model input and output information

        Args:
            model_path (str): Model path
        """
        check_file(model_path, "Model is not exist!")  # check model existence

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]  # use cuda if gpu is available
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )  # get providers
        self.model = ort.InferenceSession(model_path, providers=providers)  # load session

        model_input = self.model.get_inputs()[0]  # get input info
        self.input_name = model_input.name
        _, _, self.width, self.height = model_input.shape

        output, seg_output = self.model.get_outputs()  # get output info
        self.output_names = [output.name, seg_output.name]
        _, self.seg_chanels, self.seg_width, self.seg_height = seg_output.shape

    def _get_metadata(self, default_labels: List[str] = labels, default_stride: int = 32) -> None:
        """Get model metadata

        Args:
            default_labels (List[str], optional): Get model labels if specified. If model metadata
                doesn't contain label then use default labels (utils.labels).
            default_stride (int, optional): Get model stride if specified. model metadata
                doesn't contain stride the use default stride=32.
        """
        metadata = self.model.get_modelmeta().custom_metadata_map
        self.labels = ast.literal_eval(metadata["names"]) if "names" in metadata else default_labels
        self.stride = (
            ast.literal_eval(metadata["stride"]) if "stride" in metadata else default_stride
        )

    def forward(self, input: npt.NDArray[np.float32]) -> List[npt.NDArray[np.float32]]:
        """Get model prediction

        Args:
            input (npt.NDArray[np.float32]): Input image.

        Returns:
            List[npt.NDArray[np.float32]]: Model outputs
        """
        return self.model.run(self.output_names, {self.input_name: input})


class DNNModelLoader(ORTModelLoader):
    """OpenCV DNN model handler"""

    def __init__(self, path) -> None:
        super().__init__(path)

        self.model = cv2.dnn.readNet(path)  # overide ort model

        if cv2.cuda.getCudaEnabledDeviceCount():  # use CUDA if available
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:  # use CPU
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def forward(self, input: npt.NDArray[np.float32]) -> List[npt.NDArray[np.float32]]:
        """Get model prediction

        Args:
            input (npt.NDArray[np.float32]): Input image.

        Returns:
            List[npt.NDArray[np.float32]]: Model outputs
        """
        self.model.setInput(input, self.input_name)
        return self.model.forward(self.output_names)
