import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from gazenet import GazeNet
import ssd_detector.tester as facedetect
from skimage import io, transform
import time
import glob, os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import loadmat
import logging
import matplotlib.pyplot as plt
import frame_helper as ef
import frame_to_video as fv
import multiprocessing as mp
import imageio

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

net = GazeNet()
net = DataParallel(net)
net.cpu()
torch.device('cpu')
pretrained_dict = torch.load('./savedmodels/pretrained_model.pkl', map_location=lambda storage, loc: storage)
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid


def preprocess_image(image_path, eye):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # crop face
    x_c,y_c = eye

    x_0 = x_c - 0.15
    y_0 = y_c - 0.15
    x_1 = x_c + 0.15
    y_1 = y_c + 0.15
    if x_0 < 0:
        x_0 = 0
    if y_0 < 0:
        y_0 = 0
    if x_1 > 1:
        x_1 = 1
    if y_1 > 1:
        y_1 = 1

    h, w = image.shape[:2]
    face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]

    # process face_image for face net
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = data_transforms['test'](face_image)
    # process image for saliency net
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    # generate gaze field
    gaze_field = generate_data_field(eye_point=eye)
    sample = {'image': image,
              'face_image': face_image,
              'eye_position': torch.FloatTensor(eye),
              'gaze_field': torch.from_numpy(gaze_field)}

    return sample


def test(net, test_image_path, eyes):
    net.eval()
    heatmaps = []

    data = preprocess_image(test_image_path, eyes)

    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data[
        'eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cpu(), volatile=True),
                                                      [image, face_image, gaze_field, eye_position])

    _, predict_heatmap = net([image, face_image, gaze_field, eye_position])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])

    return  f_point[0], f_point[1]


def draw_result(gazeim, eye, gaze_point,i):
    color=[0,0,255]
    x1, y1 = eye
    x2, y2 = gaze_point

    image_height, image_width = gazeim.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    #cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    #cv2.circle(im, (x2, y2), 5, [0, 0, 0], -1)
    cv2.arrowedLine(gazeim, (x1, y1), (x2, y2), color, 3)

    # heatmap visualization
    #heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    #heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    #heatmap = cv2.resize(heatmap, (image_width, image_height))

    #heatmap = (0.8 * heatmap.astype(np.float32) + 0.2 * im.astype(np.float32)).astype(np.uint8)
    return gazeim


def inference(imname):
    im = io.imread(imname)
    gazeim = cv2.imread(imname)
    head, tail = os.path.split(imname)
    faces, eye_coords, eyes = facedetect.getFaces(im)
    if eyes.__len__():
        for i in range(len(eyes)):
            p_x, p_y = test(net, imname, (eyes[i][0], eyes[i][1]))
            gazeim=draw_result(gazeim, (eyes[i][0], eyes[i][1]), (p_x, p_y),i)
        cv2.imwrite('./output/' + tail, gazeim)



def main():
    video_name = 'yuyu.mp4'
    vid = imageio.get_reader(video_name, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    frame_list = []
    for i, im in enumerate(vid):
        frame_list.append(im)

    print('Frame List Created')
    ef.clear_frames('./imgs/test/')
    ef.clear_frames('./output/')
    ef.extract_frames('./imgs/test/')
    pool=mp.Pool()
    pool.map(inference,glob.glob('imgs/test/*'))
    pool.close()

    fv.convert_frames_to_video('./output/*.png', './output/videos/output.mp4', 5)


if __name__ == '__main__':
    main()