"""This file will have the code for the Class that will be used to interact with a model trained on images.

This class will provide the necessary methods to request information from an nn trained on images
and then return a classification string.

"""

from typing import Any

import onnxruntime as ort
import numpy as np

from onnxruntime import InferenceSession


class ImageNNApi:
    session: InferenceSession
    onnx_file_path: str
    onnx_inp_key: str

    def __init__(self, onnx_file_path: str, onnx_inp_key: str = "input"):
        """This class is responsible for the management of an underlying onnx mmodel trained on images.

        Args:
            onnx_file_path (str): The path to the onnx file to load
        """

        self.onnx_file_path = onnx_file_path
        self.onnx_inp_key = onnx_inp_key
        self.session = ort.InferenceSession(self.onnx_file_path)

    def recognize_image_from_ndarry(self, image: np.ndarray) -> Any:
        """Given an ndarray, run the classification and return the most likely output"""
        result = self.session.run(None, {self.onnx_inp_key: image})
        y: Any = result[0]

        return y

    def recognize_image(self, path: str):  # This will actually take in a stream
        pass
