# BarbadosSignLanguageConverter

This repository hosts the code for my research project, and will be broken down into three (3) parts.

These parts are:

- **Machine Learning Model**
: The model responsible for being able to recognize hand gestures and produce speech and text from them. 

- **Application Programming Interface**
: This will be the service class that defines the methods to interact with the onnx model.

- **Application Programming Interface Server**
: Primarily will take the form of a web server, capable of providing access to the features of the model in a standardized, documented way. 

- **User-Interface Application**
: A website or mobile application which will communicate with the server in order to get access to the features that the model provides. 

## Components

### Machine Learning Model

This will be a model that (at it's core) will be trained to detect and recognize Barbadian Sign Language (the sign language variant specific to Barbados).

This will serve to "immortalize" the data set and assist in digitizing the language.

Completing this model requires (in general) the following items:

- Data that has been cleaned, formatted, and converted to a CSV format.
- Both training and testing data.
- A neural network (designed using PyTorch) capable of learning and recognizing the gestures.
- An Onnx model for easy exporting and use for other layers.

#### Development Tools

- Python (PyTorch)

### Application Programming Interface

The API will be a class that wraps around an onnx model, and provides methods for getting recognition information from it.

This API will contain two (2) components. These being:

- Image Classification
- Video Classification

These will provide 2 different interfaces for classifying some input.
Image Classification will expect a single image or frame as input, and then perform classification on it. As such, the onnx model provided to it is expected to be one which is designed to work on images.

Subsequentlly, the Video classification will expect a video stream or otherwise sequence of frames, in order to perform classification. As such, the onnx model provided to it is expected to be one which was trained on videos.

#### Image Classification

This class which will be defined as `ImageNNApi`, will have 2 main methods.

- recognize_image_from_ndarray
- recognize_image 

The `recognize_image_from_ndarray` method, will accept a numpy n-dimensional array, of an image that was already pre-processed to meet the requirements based on the model the user is using. This method will simply run the onnx session, pass in the input, and output the result, which would be a list of likelihoods for the time being.

The `recognize_image` method, will accept a path, or some image object (still need to figure this out), and perform the conversion to an ndarray, and then return the results from `recognize_image_from_ndarray`

### Application Programming Interface Server

The application programming interface (API) will be used to interface and control the model. 

This API will primarily be responsible for accepting an image or video stream as input, passing it to the model, and then returning some output.

Tentatively, the output should be a string indicating what the recognized hand gesture is.

This API should also suppport websockets, for live and continous communication between a client and the server / model.

#### Development Tools

- Python (FastAPI)

### User-Interface Application

This will be any client side application desgined to interact with the API provided in that layer.

Therefore, this can take numerous forms, such as a website, mobile app, perhaps even just a simple script using OpenCV.

#### Development Tools

- Python (OpenCV)
- Svelte + JavaScript (Website)
- Flutter + Dart (Mobile App)


## Folder Structure

As mentioned, the app will be split into three (3) components, that (for now) will all reside here in this repository.

As such, at the root level we will have:

- README.md (This file)
- nnmodel (The folder to store the neural network and its related data)
- bslapi (This folder will store the code for the API, that makes use of the lighter weight onnx model.)
- clients (This folder will store some of the clients developed. Each will communicate with the API through HTTP requests)

> [!NOTE]
> The code structure for each will be explained in more detail in the System Design section.

### nnmodel 

This folder will have the following items:

- code responsible for training the neural network
- the data that will be used to train it
- the code used to convert the raw images to csv for training
- the code used to export the onnx model
- and anything else required for the successful training.

The `data` folder will have the following structure:

- videos/
- bsl_training.csv
- bsl_testing.csv

The `videos` folder will be used to store the raw videos used for training and validating.
This folder will contain a subfolder for each sign, of which may contain one or more clips for that sign.

A script will then be used to go through each of these subfolders, and then, using the folder name as the label, convert the videos into numpy arrays, and then create the bsl_training.csv and bsl_testing.csv files.

For each sign folder, training and testing will be decided as follows:

- If the number of files in the sign folder is 1, then use the same clip for training and testing.
- Use `random.sample` and select 1 clip to use for validation, and the others for training.

### bslapi 

This folder will store the code that will be used to be an API around the model.

This will consist of two (2) main components.

1. An API class
1. A WebServer.

The API class will be what actually wraps the created output model, providing an interface with which can then be used by another service (for example a web server) to perform recognition of the gestures.

The WebServer, will be responsible for using the API class and providing a way of accessing the underlying model by way of HTTP requests.

The structure of both of these will be eexamined in the "System Design > bslapi" section.

### clients

This folder will contain subfolders of the various clients.

Recall, these clients will be responsible for providing a user interface for the model, and will make use of the API wrapper or the Web Server to communicate with it.

## System Design 

This section will have more in depth details about how each component of the system will be designed.

### The Model

"The Model" refers to the neural network which will be trained and tested on the data.

The entire process is broken down into the following components.

1. Data Preparation
1. Dataset Design
1. Model Training
1. Model Evaluation

Each phase will be explored in their subsections below.


####  Data Preparation

We have been provided with several long form videos.
Data preparation will involve going through each video, and performing the following actions.

- [ ] Segmenting the video using video editing tools.
- [ ] Performing any neccessary modifications to the clips / images to ensure that the hands are the focus.
- [ ] Adding noise to each of the clips, ensuring that there is a variety of backgrounds and colors. Either that, or just have it always use GreyScaled data.
- [ ] Labelling the data correctly.
- [ ] Separating the data into training and validation sets.
- [ ] Exporting the data into CSV format so that the model can be trained on it.

####  Dataset Design

Dataset Design refers to the design of the dataset class that PyTorch will use.
This involves creating a class which inherits from Dataset, and will be responsible for loading the Sign Language Dataset that was created in the previous phase into Pytorch.

This class will be responsible for getting the labels, their mappings, and associated values.

This phase will also involve the creation of the train and test data loaders.

####  Model Training

In this phase, the model will undergo training.

This will involve using the loaders that were created, alongside creating the class to represent our Neural Network.

This class will contain the init method to initialize the network with the correct data types, and a forward method to do the training.

We will also need a `train` method that will be used for training the neural network using the trainloader for a given number of epochs.

This stage can be itemized with the following.

####  Model Evaluation

At this stage, the following will occur.

- [ ] Test the model using the validation data.
- [ ] Export the model to onnx and the run the validation data on it to ensure.

### Model API

This section refers to the API that will wrap the onnx model using the onnxruntime, and provide access to the evaluation features.

The API should support:

- Single Image Detection
- Parsing Video Stream

The process will be examined in further detail in the below IPO section.

#### Input

For input, the API will have the following methods.

`recognize_image(image: Image|MatLike) -> dict`

This method will accept an image as input, attempt to recognize it, then return a dictionary as shown in the Output section.

`recognize_stream(stream) -> dict`

I have no idea how this will work as yet.

#### Processing 

The processing simply involves creating the onnxruntime session using the previously created onnx model, and then calling run with the provided input to get an output.

The output can then be evaluated, formatted, and then returned.

#### Output

The model will output a response containing the following structure:

```json

{

    "status": int, // Will typically either be 200, 404 or 500
    "label": str | List[str] | null, // This will be the recognized sign

}

```

The aim is that the client / server will then decide what to do with the returned label

### Model API Server 

This refers to the web server (built with FastAPI), that will act as a web wrapper around the underlying API model created in the previous step.

This server will allow clients to communicate with the model using HTTP methods.

Ideally, it will support:

- [ ] Single Image Processing 
- [ ] Multi Image Processing
- [ ] Streaming using WebSockets

### Clients

Clients refer to user-facing interfaces which will either interact with the API directly, or to the web server wrapper wrapped around it.

These clients for the most part will be:

- [ ] A simple Open CV
- [ ] A website with web cam permissions
- [ ] A mobile application.

The details of these aren't very important.
The web and mobile application will use the web server provided by the API layer, in order to communicate with the model, while the OpenCV will use the API directly.

They will communicate with the layer in the most appropriate way, and will handle the output accordingly.

