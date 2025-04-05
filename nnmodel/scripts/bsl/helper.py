"""This contains helper methods used by the various scripts for training."""

import cv2
from cv2.typing import MatLike

from typing import List

from pytorchvideo.data.encoded_video import EncodedVideo
import skvideo.io
import albumentations as A


def load_video(video_path:str):

    frame_list: List[MatLike] = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)

    cap.release()

    print(len(frame_list))
    video = EncodedVideo.from_path(video_path)
    print("Obtaining frames from video: ", video_path)
    return frame_list, len(frame_list), fps, int(video.duration)


def create_augmented_video(frame_list: List[MatLike], path_augmented_video: str, fps: int):

    print("Creating augmented video in ", path_augmented_video)
    writer = skvideo.io.FFmpegWriter(
        path_augmented_video,
        inputdict={"-r": str(fps)},
        outputdict={
            "-r": str(fps),
            "-c:v": "libx264",
            "-preset": "ultrafast",
            "-pix_fmt": "yuv444p",
        },
        verbosity=1,
    )

    for i, image in enumerate(frame_list):
        image = image.astype("uint8")
        writer.writeFrame(image)

    # close writer
    writer.close()


"""
transform = A.ReplayCompose([
    #A.ElasticTransform(alpha=0.1, p=0.5),
    A.GridDistortion(distort_limit=0.4, p=0.6),
    ##A.OpticalDistortion(distort_limit=0.5, p=1),
    #A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=5, p=0.5),
    A.Rotate(limit=5, p=0.6),
    ##A.GaussNoise(var_limit=[30.0, 70.0], mean=1, p=1),
    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.CLAHE(p=0.6),
    A.PixelDropout(drop_value=0, dropout_prob=0.02, p=0.5),
    A.PixelDropout(drop_value=255, dropout_prob=0.02, p=0.5),
    A.Blur(blur_limit=(2, 4), p=0.5)
])
"""

transform = A.ReplayCompose(
    [
        A.ElasticTransform(alpha=0.5, p=0.5),
        A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(p=0.5),
        A.PixelDropout(drop_value=0, dropout_prob=0.01, p=0.5),
        A.PixelDropout(drop_value=255, dropout_prob=0.01, p=0.5),
        A.Blur(blur_limit=(2, 4), p=0.5),
    ]
)


def augment_frames(frame_list: List[MatLike]):
    data = None
    augmented_frame_list = []

    for i, item in enumerate(frame_list):
        if i == 0:
            first_image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            data = transform(image=first_image)
            new_image = data["image"]
        else:
            image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            new_image = A.ReplayCompose.replay(data["replay"], image=image)["image"]

        # new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR): the images have to output as RGB images
        augmented_frame_list.append(new_image)

    return augmented_frame_list
