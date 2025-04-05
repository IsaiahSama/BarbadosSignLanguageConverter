# Barbados Sign Language

Within the /nnmodel/data/datasets/videos/BSL folder, there are a number of "base" videos that show different signs, generally only 1 video clip per handsign.

The purpose of the code in this folder is to take the videos, perform processing, and then train a neural network on it in order to perform classification.

In accordance with [Enrico Randellini's article](https://medium.com/@enrico.randellini/hands-on-video-classification-with-pytorchvideo-dc9cfcc1eb5f), this will be broken down into the following steps.

1. Use the videos we have as the test validation set.
1. Perform data augmentation transformations to create synthetic videos. These will be used in the train and validation sets
1. Define the VideoClassifcationModel and ClassificationDataset
1. Perform training and testing using the dataframes and dataloaders.
