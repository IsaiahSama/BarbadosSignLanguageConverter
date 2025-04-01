The entries for the dataaset are stored in the MSASL_test.json, MSASL_train.json and MSASL_val.json files.
The format for each entry of the dataset is as follows:

```py
{
    "org_text": str,
    "clean_text": str, 
    "start_time": float, 
    "signer_id": int, 
    "signer": int, 
    "start": int,
    "end": int,
    "file": str,
    "label": int,
    "height": float,
    "fps": float,
    "end_time": float,
    "url": str,
    "text": str,
    "box": List[float],
    "width": float
}
```

The video files can be downloaded from the "url" field.
The labels are indexes, which are obtained from the MSASL_classes.json file.

With this information, we can use the data to train a model.

Training will be done in the following steps:

1. Download the test, train and val datasets and update the json files, by adding an extra "path" field to each entry.
1. Increase the number of train and validation examples with data augmentation transformations.
1. Create the VideoClassificationDataset to load the data.
1. Create the VideoClassificationModel to define the network architecture.
1. Train and evaluate the model.