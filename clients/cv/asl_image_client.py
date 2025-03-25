"""This class will be responsible for providing a client for the ASL image recognition.

This is built out as a sample proof of concept to interact with the model

"""

from string import ascii_uppercase

import cv2
import numpy as np

from bslapi import ImageNNApi


class ASLImageClient:
    mean: float  # The mean
    std: float  # Standard Deviation

    def __init__(self, mean: float = 0.485, std: float = 0.229):
        self.mean = mean * 255
        self.std = std * 255

    def run(self, api: ImageNNApi):
        """This run function will handle the creation and management of the OpenCV camera"""

        capture = cv2.VideoCapture(0)

        self.camera_loop(capture, api)

    def camera_loop(self, capture: cv2.VideoCapture, api: ImageNNApi):
        """This will run the main loop for the camera, reading each frame

        Args:

        capture (cv2.VideoCapture): The camera that will be used to capture input
        """

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            frame = self.center_crop(frame)
            # Converting to Gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Next, we need to modify the frame to make it match our expected input

            x = cv2.resize(frame, (28, 28))  # Resizing to a 28 x 28 image
            x: np.ndarray = (x - self.mean) / self.std

            x = x.reshape(1, 1, 28, 28).astype(np.float32)  # Reshape and type cast

            y = api.recognize_image_from_ndarray(x)  # Recognize image from the frame
            index = np.argmax(y, axis=1)

            letter = ascii_uppercase[int(index)]

            cv2.putText(
                frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2
            )
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        capture.release()
        cv2.destroyAllWindows()

    def center_crop(self, frame: np.ndarray) -> np.ndarray:
        """This method will crop the frame to the center of the received image"""

        height, width, _ = frame.shape

        start = abs(height - width) // 2

        if height > width:
            return frame[start : start + width]
        else:
            return frame[:, start : start + height]


if __name__ == "__main__":
    client = ASLImageClient()
    api = ImageNNApi("./sign_language.onnx")

    client.run(api)
