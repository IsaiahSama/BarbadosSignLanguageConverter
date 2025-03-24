"""This file represents the API that makes use of an onnx model designed to recognize and return the labels of different hand gestures"""

import onnxruntime as ort
import numpy as np

from onnxruntime import InferenceSession
from numpy import ndarray


class NNAPI:
    session: InferenceSession
    onnx_file_path: str

    def __init__(self, onnx_file_path: str):
        self.onnx_file_path = onnx_file_path
        self.session = ort.InferenceSession(self.onnx_file_path)

    def recognize_image(self, image: ndarray):
        result = self.session.run(None, {"input": image})
        print(result)
        data = result[0]
        return self.get_response(True, data)

    def get_response(self, success: bool, data=None | str):
        return {"success": success, "label": data}
