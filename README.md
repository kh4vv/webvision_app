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
$ pip install -r requirements.txt  <- 이거 지금 안되는데 왜 안되는지 모르겠네요. 나중에 형이 한번 봐주세요
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

프리트레인드 파일 어떻게 다운로드 링크 걸지 생각해봐야할거같아요 (google drive

<이 밑은 아직 지우시 마세요. 

## Available Scripts
.W. Cho
In the project directory, you can run:

### `npm start`


### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.

You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
