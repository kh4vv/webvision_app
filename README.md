# Welcome to Computer Vision World

### This is a web application (Image Classification, Object Detection, Instant Segmentation)
#### By K.W Hong and Y.W. Cho
##### Any question and comment, email here: kh4vv@virginia.edu (K.W Hong) and choyoungwoon@gmail.com (Y. W Cho)

#### Index of Contents

1. [Getting Started](#getting-Started)
2. [Weight File](#weight-file)

## Getting Started

#### Frontend

**First**, you need to `git clone` the project and go to the directory.
```
$ git clone https://github.com/kh4vv/webvision_app.git
$ cd webvision_app
```
**Second**, in order to run this application, you need to install `node.js` and `npm`.
Here is the instruction where and how to install those: [node.js](https://nodejs.org/en/download/)

Once you have `node.js`, you will automatically get `npm` installed in your machine.

**Third**, you need to install `yarn` through the `npm package manager`. You can install it by writing this command:
```
$ npm install -g npm 
$ npm install --global yarn
```
Then, you need to download the node-modules to start the application. 
```
$ npm install react-script
```
Once it is done, you can start the application:
```
$ yarn start
```

Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

#### Backend

You can open the another terminal. Before you lauch the backend, you need to download\
few libraries and [flask](https://flask.palletsprojects.com/en/1.1.x/).

Let's create a virtual environment called `venv` in the project directory `webvision_app/`:\
If you don't have the virtual environment was not created successfully, you can download by
`apt-get install python3-venv`

```
$ cd api
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ _
```
Now, let's download all neccessary libraries before starting the backend server.
All the requriement libraries are in `requirements.txt`. 
```
$ pip install -r requirements.txt
```
For now we need to install individually before we fix the problem.

```
$ pip install flask
$ pip install numpy
$ pip install pillow
$ pip install torch
$ pip install torchvision
$ pip install opencv-python
$ pip install albumentations
$ pip install efficientnet_pytorch
$ pip install flask-cors
$ pip install quickdraw
```

After that, you can start it by using `yarn`:
```
$ yarn start-api
```

## Weight File

Now we have to create the directory called `weights` and download the pre-trained weight.
Go to `api` directory and create it.
```
$ mkdir weights
$ cd weights
```
Weighs can be downloaded from [Google Cloud](https://drive.google.com/drive/folders/1E8wspdt9aGRzGrCeyQlsOYMHJ3ompQtR?usp=sharing)

In Google Drive, it should be contained:
```
effnet_448_512_34_190000.pth  
mnist.pth  
ResNext101_448_300_141_390000.pth  
yolov3.pth
```
After you downloaded the pre-trained weight files and moved to the correct directory, 
you need to restart the backend server by `$ yarn start-api` from the `/webvision_app` location

