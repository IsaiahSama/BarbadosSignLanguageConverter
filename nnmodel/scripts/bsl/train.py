"""This will be responsible for running training of the model."""

import os
import torch
import torch.nn as nn
import pandas as pd

from classification_dataset import create_loaders

VIDEO_BASE_FOLDER = "..\\..\\data\\datasets\\videos\\BSL"
VIDEO_ORIGINAL_FOLDER = os.path.join(VIDEO_BASE_FOLDER, "original")
VIDEO_MOD_FOLDER = os.path.join(VIDEO_BASE_FOLDER, "modified")
BSL_DATASET_FOLDER = "..\\..\\data\\datasets\\BSL"

path_dataset_test_csv = "test.csv"
path_dataset_train_csv = "train.csv"
path_dataset_val_csv = "val.csv"

dataset_path = BSL_DATASET_FOLDER

# load dataset csv
path_dataset_train_csv = os.path.join(dataset_path, path_dataset_train_csv)
df_dataset_train = pd.read_csv(path_dataset_train_csv)
path_dataset_val_csv = os.path.join(dataset_path, path_dataset_val_csv)
df_dataset_val = pd.read_csv(path_dataset_val_csv)
path_dataset_test_csv = os.path.join(dataset_path, path_dataset_test_csv)
df_dataset_test = pd.read_csv(path_dataset_test_csv)

# create the dataloaders
classification_dataloader_train, classification_dataloader_val, classification_dataloader_test = create_loaders(df_dataset_train, df_dataset_val, df_dataset_test, cfg)

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# load the model and set the classes
model = create_model(cfg, device)
checkpoint_dir = saving_dir_model


# set Loss, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)