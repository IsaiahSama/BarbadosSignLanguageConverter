"""This script will be used to download all of the videos from the dataset."""

import json
from os.path import exists
from threading import Thread
from pytube import YouTube

MS_ASL_DIR = "../../data/datasets/MS_ASL/"
LOCAL_MS_ASL_DIR = "./data/"
MS_ASL_VIDEO_DIR = "../../data/datasets/videos/MS_ASL/"

TRAIN_FILE = "MSASL_train.json" 
TEST_FILE = "MSASL_test.json"
VAL_FILE = "MSASL_val.json"

types = ["train", "test", "val"]

def download_and_save(url: str, save_path: str, filename: str):
    print("Attempting to download", filename)
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
        if stream:
            stream.download(save_path, filename)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}")
    except Exception as e:
        print(e)
        
threads = []

for file in types:
    # Go through the different json files
    print("Hello")
    with open(MS_ASL_DIR + f"MSASL_{file}.json", "r") as f:
        samples = json.load(f)
        # Load each sample
        for i, sample in enumerate(samples[:10]):
            video_url = sample["url"]
            video_label = sample["label"]
            filename = f"{sample['clean_text']}.mp4"
            sample["path"] = filename
            
            # Check if the video isn't already stored
            
            if not exists(MS_ASL_VIDEO_DIR + filename):
                # If not, then we download it
                t = Thread(target=download_and_save, args=(video_url, MS_ASL_VIDEO_DIR, filename))
                t.start()
            else:
                print(f"Video {filename} already exists")
                
    break

for t in threads:
    t.join()
