import time
import os
from flask import Flask, flash, request, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json
import base64
import cv2
import io

import torch
from inference import mnist_evaluation, quickdraw_evaluation, landmark_evaluation, transform_landmark
from objdec_inference import yolov3_evaluation
from model import LeNet, ResNext101Landmark

# Initialize the useless part of the base64 encoded image.
init_Base64 = 22

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
result = {}


@app.route('/')
def backendmain():
    return 'hello'


@app.route('/time')
def get_current_time():
    return {'time': time.time()}


@app.route('/mnist_upload', methods=['GET', 'POST'])
def fileUpload():
    if request.method == 'POST':
        path = '../public/img'
        mnist_f = request.files['file']
        mnist_fname = secure_filename(mnist_f.filename)
        print(mnist_fname)

        mnist_f.save(os.path.join('static', mnist_fname))
        mnist_img = Image.open(mnist_f, 'r')
        mnist_img.save(os.path.join(path, mnist_fname))

        weight_path = './weights/mnist.pth'
        mnist_model = LeNet().to(device)

        mnist_img, mnist_preds = mnist_evaluation(
            mnist_img, weight_path, mnist_model)
        mnist_preds = int(mnist_preds)
        print(mnist_preds)
        return {'filename': mnist_fname, 'pred': mnist_preds}
    else:
        return {}


@app.route('/mnist_pad', methods=['GET', 'POST'])
def mnist_predict():
    result = {}
    if request.method == 'POST':

        mnist_draw = request.form['url']
        print(mnist_draw)
        mnist_draw = mnist_draw[init_Base64:]
        print(mnist_draw)
        mnist_draw_decoded = base64.b64decode(mnist_draw)
        # Fix later(to PIL version)
        # Conver bytes array to PIL Image
        imageStream = io.BytesIO(base64.b64decode(mnist_draw))
        img = Image.open(imageStream)
        I = np.asarray(img)
        print(I)

        mnist_img = np.asarray(bytearray(mnist_draw_decoded), dtype="uint8")
        mnist_img = cv2.imdecode(mnist_img, cv2.IMREAD_GRAYSCALE)
        mnist_img = cv2.resize(mnist_img, (28, 28),
                               interpolation=cv2.INTER_AREA)
        print(mnist_img)
        mnist_img = Image.fromarray(mnist_img)
        print(mnist_img)

        weight_path = './weights/mnist.pth'
        mnist_model = LeNet().to(device)
        mnist_img, mnist_pred = mnist_evaluation(
            mnist_img, weight_path, mnist_model)

        mnist_pred = int(mnist_pred)
        print(mnist_pred)

        return mnist_draw_decoded

    return result


@app.route('/landmark', methods=['GET', 'POST'])
def landmark_upload_image():
    with open('./dataset/landmark_classmap.json', 'r', encoding='UTF-8-sig') as json_file:
        landmark_classmap = json.load(json_file)

    if request.method == 'POST':
        landmark_f = request.files['file']
        landmark_fname = secure_filename(landmark_f.filename)
        path = '../public/img'
        landmark_f.save(os.path.join('static', landmark_fname))

        landmark_img = Image.open(landmark_f, 'r')
        landmark_img.save(os.path.join(path, landmark_fname))

        num_classes = 1049
        # weight_path = './weights/effnet_448_512_34_190000.pth'
        weight_path = './weights/ResNext101_448_300_141_390000.pth'

        # model = EfficientNetLandmark(1, num_classes)
        model = ResNext101Landmark(num_classes).to(device)

        landmark_img, landmark_preds = landmark_evaluation(
            landmark_img, weight_path, model)
        landmark_preds = landmark_classmap[str(int(landmark_preds))]
        print(landmark_preds, landmark_fname)
        return {"pred": landmark_preds, "filename": landmark_fname}

    else:
        return {}


@app.route('/quickdraw', methods=['GET', 'POST'])
def quickdraw_predict():
    quickdraw_animal_map = ['ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'dog', 'dolphin', 'dragon', 'duck', 'elephant', 'fish', 'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion',
                            'lobster', 'mermaid', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda', 'penguin', 'pig', 'rabbit', 'raccoon', 'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'teddy-bear', 'tiger', 'whale', 'zebra']

    if request.method == 'POST':
        quick_draw = request.form['url']
        quick_draw = quick_draw[init_Base64:]
        quick_draw_decoded = base64.b64decode(quick_draw)

        # Fix later(to PIL version)
        # Conver bytes array to PIL Image
        # imageStream = io.BytesIO(draw_decoded)
        # img = Image.open(imageStream)

        quick_img = np.asarray(bytearray(quick_draw_decoded), dtype="uint8")
        quick_img = cv2.imdecode(quick_img, cv2.IMREAD_GRAYSCALE)
        quick_img = cv2.resize(quick_img, (28, 28),
                               interpolation=cv2.INTER_AREA)
        quick_img = Image.fromarray(quick_img)

        weight_path = './weight/quickdraw_90_animal.pth'
        quick_model = LeNet(num_classes=44).to(device)
        quick_img, quick_pred = quickdraw_evaluation(
            quick_img, weight_path, quick_model)

        quick_pred = int(quick_pred)
        quick_label = quickdraw_animal_map[quick_pred]
    return render_template('draw_quickdraw.html')


if __name__ == '__main__':
    app.run(host='localhost', port=9000, debug=False)
