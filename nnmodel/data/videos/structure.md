# Videos folder

The `videos` folder will be used to store the raw videos used for training and validating.
This folder will contain a subfolder for each sign, of which may contain one or more clips for that sign.

A script will then be used to go through each of these subfolders, and then, using the folder name as the label, convert the videos into numpy arrays, and then create the bsl_training.csv and bsl_testing.csv files.

For each sign folder, training and testing will be decided as follows:

- If the number of files in the sign folder is 1, then use the same clip for training and testing.
- Use `random.sample` and select 1 clip to use for validation, and the others for training.

> [!NOTE]
> This file exists so the folder is visible. The data will be visibile as a zip folder.
