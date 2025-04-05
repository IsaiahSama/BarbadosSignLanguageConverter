"""The preprocess stage should be responsible for the following:

1. Taking the videos and creating a folder structure
1. Generating synthetic videos for usage.
1. Creation of csv files for train, test and validation
"""

# Step 0: Initial Setups

import os
import random
import pandas as pd
import shutil

# from pytorchvideo.data.encoded_video import EncodedVideo
from enum import Enum
from helper import load_video, create_augmented_video, augment_frames

VIDEO_BASE_FOLDER = "..\\..\\data\\datasets\\videos\\BSL"
VIDEO_ORIGINAL_FOLDER = os.path.join(VIDEO_BASE_FOLDER, "original")
VIDEO_MOD_FOLDER = os.path.join(VIDEO_BASE_FOLDER, "modified")
BSL_DATASET_FOLDER = "..\\..\\data\\datasets\\BSL"

csv_file_names = ["BSL_train.csv", "BSL_test.csv", "BSL_val.csv"]


class VideoTypes(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    ALL = "all"


# Step 1: Creating a folder structure

if not os.path.exists(VIDEO_MOD_FOLDER):
    os.makedirs(VIDEO_MOD_FOLDER)
    os.makedirs(os.path.join(VIDEO_MOD_FOLDER, VideoTypes.TRAIN.value))
    os.makedirs(os.path.join(VIDEO_MOD_FOLDER, VideoTypes.TEST.value))
    os.makedirs(os.path.join(VIDEO_MOD_FOLDER, VideoTypes.VAL.value))
    os.makedirs(os.path.join(VIDEO_MOD_FOLDER, VideoTypes.ALL.value))

for video in os.listdir(VIDEO_ORIGINAL_FOLDER):
    label = video.split(".")[0]
    new_loc = os.path.join(VIDEO_MOD_FOLDER, VideoTypes.ALL.value, label)
    test_loc = os.path.join(VIDEO_MOD_FOLDER, VideoTypes.TEST.value, label)

    os.makedirs(new_loc, exist_ok=True)
    os.makedirs(test_loc, exist_ok=True)

    if not os.path.exists(os.path.join(new_loc, video)):
        shutil.copy2(
            os.path.join(VIDEO_ORIGINAL_FOLDER, video), os.path.join(new_loc, video)
        )
        shutil.copy2(
            os.path.join(VIDEO_ORIGINAL_FOLDER, video), os.path.join(test_loc, video)
        )


PERC_TRAIN = 0.5
PERC_TEST = 0.4
NUMBER_VIDEO_TRAIN = 30
NUMBER_VIDEO_VAL = 5

# Step 2: Create the augmented videos from the original videos


def make_augmented_videos(
    input_path: str,
    output_path: str,
    subdir: VideoTypes = VideoTypes.ALL,
    final_number: int = 0,
    apply_augmentation: bool = True,
):
    input_path = os.path.join(input_path, subdir.value)
    output_path = os.path.join(output_path, subdir.value)

    for vid_dir in os.listdir(input_path):
        path_vid_dir = os.path.join(input_path, vid_dir)
        number_videos = len(os.listdir(path_vid_dir))

        output_dir = os.path.join(output_path, vid_dir)

        print(f"{vid_dir=}, {number_videos=}, {output_dir=}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Creating augmented videos for ", vid_dir)

        if apply_augmentation:
            # Number of times to apply a transformation on a single video.
            # Each transformation creates a new video. So for each video, how many transformations need to be made
            # To reach the desired number of resulting videos.
            # i.e, how many new video does each current video need to make to get to final_number
            n_applications = round((final_number - number_videos) / number_videos)

            for video in os.listdir(path_vid_dir):
                video_path = os.path.join(path_vid_dir, video)
                print(f"{video_path=}")
                # Load video and divide into frames
                frames, _, fps, _ = load_video(video_path)

                # Go through the number of times to apply the transformation
                for i in range(n_applications):
                    # Apply augmentation to each frame
                    augmented_frames = augment_frames(frames)

                    # Create augmented video

                    name_augmented_video = (
                        video.split(".")[0] + "_" + str(i + 1) + ".mp4"
                    )  # video_1.mp4
                    path_augmented_video = os.path.join(
                        output_dir, name_augmented_video
                    )
                    create_augmented_video(augmented_frames, path_augmented_video, fps)

                new_file_name = video.split(".")[0] + "_0.mp4"
                new_file_path = os.path.join(output_dir, new_file_name)
                shutil.move(video_path, new_file_path)


# make_augmented_videos(VIDEO_MOD_FOLDER, VIDEO_MOD_FOLDER, VideoTypes.ALL, 10)

# (Only have 1 clip per vid, so will need to make the augmented videos first before splitting.)
# Step 3: Split dataset into train, test and validation


def split_dataset(input_path: str, output_path: str):

    print("Partition into TRAIN - TEST - VAL")
    for subdir, dirs, files in os.walk(input_path):
        for dir in dirs:
            path_subdir = os.path.join(input_path, dir)
            print("CLASS: {}".format(dir))

            # collect the path for mp4 files with duration grater then 1 sec
            list_path_mp4_files = []
            for file in os.listdir(path_subdir):
                list_path_mp4_files.append(os.path.join(path_subdir, file))

            print(list_path_mp4_files)

            number_files = len(list_path_mp4_files)
            index_list = [i for i in range(number_files)]

            number_file_train = round(number_files * PERC_TRAIN)
            number_file_test = round(number_files * PERC_TEST)

            sum_train_test = number_file_train + number_file_test
            # if the sum of the train and test files equals the total number,
            # I decrease the number of train files by one, so I have space for a validation file
            if sum_train_test == number_files:
                if sum_train_test != 2:
                    number_file_train = number_file_train - 1

            if sum_train_test != 2:
                train_index_list = random.sample(index_list, k=number_file_train)
                index_list = list(set(index_list) - set(train_index_list))
                test_index_list = random.sample(index_list, k=number_file_test)
                val_index_list = list(set(index_list) - set(test_index_list))
            else:
                train_index_list = [0]
                test_index_list = [1]
                val_index_list = [1]

            # TRAIN assignment and copy to train directory
            for idx in train_index_list:
                path_file = list_path_mp4_files[idx]
                filename = path_file.split("\\")[-1]
                dst_dir = os.path.join(
                    output_path, VideoTypes.TRAIN.value, dir, filename
                )

                CHECK_FOLDER = os.path.isdir(
                    os.path.join(output_path, VideoTypes.TRAIN.value, dir)
                )
                if not CHECK_FOLDER:
                    os.makedirs(
                        os.path.join(output_path, VideoTypes.TRAIN.value, dir),
                        exist_ok=True,
                    )

                shutil.copy2(path_file, dst_dir)
            print(
                "Number of files into dir '{}': {}".format(
                    os.path.join(output_path, VideoTypes.TRAIN.value, dir),
                    len(
                        os.listdir(
                            os.path.join(output_path, VideoTypes.TRAIN.value, dir)
                        )
                    ),
                )
            )

            # TEST assignment and copy to test directory
            for idx in test_index_list:
                path_file = list_path_mp4_files[idx]
                filename = path_file.split("\\")[-1]
                dst_dir = os.path.join(
                    output_path, VideoTypes.TEST.value, dir, filename
                )

                CHECK_FOLDER = os.path.isdir(
                    os.path.join(output_path, VideoTypes.TEST.value, dir)
                )
                if not CHECK_FOLDER:
                    os.makedirs(os.path.join(output_path, VideoTypes.TEST.value, dir))

                shutil.copy2(path_file, dst_dir)
            print(
                "Number of files into dir '{}': {}".format(
                    os.path.join(output_path, VideoTypes.TEST.value, dir),
                    len(
                        os.listdir(
                            os.path.join(output_path, VideoTypes.TEST.value, dir)
                        )
                    ),
                )
            )

            # VAL assignment and copy to val directory
            for idx in val_index_list:
                path_file = list_path_mp4_files[idx]
                filename = path_file.split("\\")[-1]
                dst_dir = os.path.join(output_path, VideoTypes.VAL.value, dir, filename)

                CHECK_FOLDER = os.path.isdir(
                    os.path.join(output_path, VideoTypes.VAL.value, dir)
                )
                if not CHECK_FOLDER:
                    os.makedirs(os.path.join(output_path, VideoTypes.VAL.value, dir))

                shutil.copy2(path_file, dst_dir)


video_folder = VIDEO_MOD_FOLDER
ttv_folder = VIDEO_MOD_FOLDER

# split_dataset(os.path.join(video_folder, VideoTypes.ALL.value), ttv_folder)

# # Step 3.5: Make more augmented videos for each of the types (train, val)

# print("Making augmented Videos for Train")
# make_augmented_videos(video_folder, ttv_folder, VideoTypes.TRAIN, NUMBER_VIDEO_TRAIN)
# print("Making augmented Videos for Test")
# make_augmented_videos(video_folder, ttv_folder, VideoTypes.TEST, 0, False)
# print("Making augmented Videos for Val")
# make_augmented_videos(video_folder, ttv_folder, VideoTypes.VAL, NUMBER_VIDEO_VAL)


# Step 4: Create csv files for train, test and validation


def create_dict_class2label(dir_path_augmented):
    class2label = dict()
    label = 0
    dir_path = os.path.join(dir_path_augmented, "train")
    for subdir, dirs, files in os.walk(dir_path):
        for dir in dirs:
            path_subdir = os.path.join(dir_path, dir)
            CHECK_FOLDER = os.path.isdir(path_subdir)
            if CHECK_FOLDER:
                class_name = path_subdir.split("\\")[-1]
                print("class_name: ", class_name)

                if class_name not in class2label:
                    class2label[class_name] = label
                    label += 1
    return class2label


class2label = create_dict_class2label(VIDEO_MOD_FOLDER)
print("class2label: ", class2label)
print("")
print("list of classes: ", [key for key in class2label.keys()])

print("----------------------------------")


def create_csv(dir_path_augmented, type, class2label):

    df = pd.DataFrame(
        columns=["CLASS", "LABEL", "PATH", "NUM_FRAMES", "NUM_SEC", "FPS"]
    )
    dir_path = os.path.join(dir_path_augmented, type)

    for subdir, dirs, files in os.walk(dir_path):
        for dir in dirs:
            path_subdir: str = os.path.join(dir_path, dir)
            CHECK_FOLDER = os.path.isdir(path_subdir)
            if CHECK_FOLDER:
                class_name = path_subdir.split("\\")[-1]
                for name_file in os.listdir(path_subdir):
                    if name_file.endswith(".mp4"):
                        video_path = os.path.join(dir_path, dir, name_file)
                        _, num_frames, fps, video_duration = load_video(video_path)
                        relative_path = os.path.join(type, class_name, name_file)
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    [
                                        {
                                            "CLASS": class_name,
                                            "LABEL": class2label[class_name],
                                            "PATH": relative_path,
                                            "NUM_FRAMES": num_frames,
                                            "NUM_SEC": video_duration,
                                            "FPS": fps,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
    return df


print("Create train csv")
df_train = create_csv(VIDEO_MOD_FOLDER, "train", class2label)
df_train.to_csv(os.path.join(BSL_DATASET_FOLDER, "train.csv"), index=False)
print(df_train.info())

print("---------------------------------")

print("Create test csv")
df_test = create_csv(VIDEO_MOD_FOLDER, "test", class2label)
df_test.to_csv(os.path.join(BSL_DATASET_FOLDER, "test.csv"), index=False)
print(df_test.info())

print("---------------------------------")

print("Create val csv")
df_val = create_csv(VIDEO_MOD_FOLDER, "val", class2label)
df_val.to_csv(os.path.join(BSL_DATASET_FOLDER, "val.csv"), index=False)
print(df_val.info())
