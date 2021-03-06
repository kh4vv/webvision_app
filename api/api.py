import time
import os
from flask import Flask, flash, request, redirect, url_for, session, jsonify, escape
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json
import base64
import cv2
from io import BytesIO

import torch

from classification.evaluation import mnist_evaluation, quickdraw_evaluation, landmark_evaluation, transform_landmark
from classification.models.models import LeNet, ResNext101Landmark, EfficientNetLandmark
from object_det.yolov3.evaluation import yolov3_evaluation
from instant_seg.maskrcnn.evaluation import maskrcnn_evaluation

# Initialize the useless part of the base64 encoded image.
init_Base64 = 22

app = Flask(__name__, static_folder='outputs')
CORS(app)

app.secret_key = "u of virginia"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('./data/coco.names', 'r') as coco_name:
    coco_classmap = coco_name.read().split("\n")[:-1]


@app.route('/mnist_upload', methods=['GET', 'POST'])
def mnist_upload():
    mnist_f = request.files['file']
    mnist_fname = secure_filename(mnist_f.filename)
    print(mnist_fname)

    mnist_f.save(os.path.join('outputs/', mnist_fname))
    mnist_img = Image.open(mnist_f, 'r')

    weight_path = './weights/lenet.pth'
    mnist_model = LeNet().to(device)

    mnist_img, mnist_preds = mnist_evaluation(
        mnist_img, weight_path, mnist_model)
    mnist_preds = int(mnist_preds)
    print(mnist_preds)

    result = {'filename': mnist_fname, 'pred': mnist_preds}
    return jsonify(result)


@app.route('/mnist_pad', methods=['GET', 'POST'])
def mnist_predict():
    mnist_draw = request.form['url']
    mnist_draw = mnist_draw[init_Base64:]
    mnist_img = Image.open(BytesIO(base64.b64decode(mnist_draw)))

    weight_path = './weights/lenet.pth'
    mnist_model = LeNet().to(device)
    mnist_img, mnist_pred = mnist_evaluation(
        mnist_img, weight_path, mnist_model, pad=True)

    mnist_pred = int(mnist_pred)
    result = {'pred': mnist_pred}
    return jsonify(result)


@app.route('/landmark', methods=['GET', 'POST'])
def landmark_upload():
    with open('./classification/dataset/landmark_classmap.json', 'r', encoding='UTF-8-sig') as json_file:
        landmark_classmap = json.load(json_file)

    if request.method == 'POST':
        landmark_f = request.files['file']
        landmark_fname = secure_filename(landmark_f.filename)
        landmark_f.save(os.path.join('outputs/', landmark_fname))
        landmark_img = Image.open(landmark_f, 'r')

        num_classes = 1049
        # weight_path = './object_det/weights/effnet_448_512_34_190000.pth'
        weight_path = './weights/ResNext101_448_300_141_390000.pth'

        # model = EfficientNetLandmark(1, num_classes)
        model = ResNext101Landmark(num_classes).to(device)

        landmark_img, landmark_preds = landmark_evaluation(
            landmark_img, weight_path, model)
        landmark_preds = landmark_classmap[str(int(landmark_preds))]
        print(landmark_preds, landmark_fname)
        return {"pred": landmark_preds, "filename": landmark_fname}


@app.route('/quickdraw', methods=['GET', 'POST'])
def quickdraw_predict():
    quickdraw_animal_map = ['ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'dog', 'dolphin', 'dragon', 'duck', 'elephant', 'fish', 'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion',
                            'lobster', 'mermaid', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda', 'penguin', 'pig', 'rabbit', 'raccoon', 'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'teddy-bear', 'tiger', 'whale', 'zebra']

    if request.method == 'POST':
        quick_draw = request.form['url']
        quick_draw = quick_draw[init_Base64:]
        quick_draw_decoded = base64.b64decode(quick_draw)

        quick_img = np.asarray(bytearray(quick_draw_decoded), dtype="uint8")
        quick_img = cv2.imdecode(quick_img, cv2.IMREAD_GRAYSCALE)
        quick_img = cv2.resize(quick_img, (28, 28),
                               interpolation=cv2.INTER_AREA)
        quick_img = Image.fromarray(quick_img)

        weight_path = './weights/quickdraw_90_animal.pth'
        quick_model = LeNet(num_classes=44).to(device)
        quick_img, quick_pred = quickdraw_evaluation(
            quick_img, weight_path, quick_model)

        quick_pred = int(quick_pred)
        quick_label = quickdraw_animal_map[quick_pred]
    return render_template('draw_quickdraw.html')


@app.route('/yolov3', methods=['GET', 'POST'])
def yolov3_upload():
    yolov3_f = request.files['file']
    yolov3_fname = secure_filename(yolov3_f.filename)

    yolov3_img = Image.open(yolov3_f, 'r')
    weight_path = './weights/yolov3.pth'
    yolov3_img = yolov3_evaluation(
        yolov3_img, weight_path, coco_classmap, yolov3_fname)

    result = {'filename': yolov3_fname}
    return jsonify(result)


@app.route('/maskrcnn', methods=['GET', 'POST'])
def maskrcnn():

    maskrcnn_f = request.files['file']
    maskrcnn_fname = secure_filename(maskrcnn_f.filename)
    maskrcnn_f.save(os.path.join('outputs/', maskrcnn_fname))

    maskrcnn_img = Image.open(maskrcnn_f, 'r')
    maskrcnn_img = maskrcnn_evaluation(
        maskrcnn_img, coco_classmap, maskrcnn_fname)
    maskrcnn_img['filename'] = maskrcnn_fname
    #print(maskrcnn_img)

    return jsonify(maskrcnn_img)


@app.route('/outputs', methods=['GET', 'POST'])
def output():
    filename_id = request.args.get('filename')
    return app.send_static_file(filename_id+'.png')


if __name__ == '__main__':
    app.run(host='localhost', port=9000, debug=False, threaded=True)
