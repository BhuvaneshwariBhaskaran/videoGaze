
import sys

from argparse import ArgumentParser
import cv2

import numpy as np
import pickle


import time
import os
from scipy.special import expit

import csv

from mtcnn.mtcnn import MTCNN

tiny_faces_program_path = os.path.join(os.getcwd(),'./')

sys.path.insert(0, tiny_faces_program_path)

import tensorflow as tf
import tiny_face_model
import util


def frange(start, end=None, inc=None):

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)

    return L


MAX_INPUT_DIM_TINYFACES = 5000.0


def showImg(img, waitTime):
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)


def PV(arr):
    for x in arr:
        print(x);


def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;


def get_csv_data(csvFile):
    with open(csvFile, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data




# MTCNN





def getFaceBoundingBoxesMTCNN(frame, MTCNN_detector):


    result = MTCNN_detector.detect_faces(frame)

    faces = []

    for ind in range(len(result)):
        bounding_box = result[ind]['box']


        faces.append(
            [bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]])

    return faces


def detectFacesMTCNN(videoFile, sampling_interval, detectionFrames, MTCNN_detector):

    vid = cv2.VideoCapture(videoFile);
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    total_frames = int(vid.get(7))
    FPS = vid.get(cv2.CAP_PROP_FPS)

    sampling_rate = int(FPS * sampling_interval)
    # TODO : may set adaptive sampling rate based on shot length

    print(videoFile, " : total_frames:", total_frames, ", FPS:", FPS, ' SR:', sampling_rate)

    detections = [];  # face detection bounding boxes
    indices = [];  # frame number for detection


    for ind in range(0, total_frames):

        if (ind%(int(total_frames/10) ) == 0):
            print( str(int(ind/int(total_frames/10)) )+'0 %' )

        ret, frame = vid.read()

        if (ret == False):
            continue


        if (ind - 2 in detectionFrames):
            faces = getFaceBoundingBoxesMTCNN(frame, MTCNN_detector)
            detections.append(faces)
            indices.append(ind)
            continue;


        if (ind % sampling_rate != 0):
            continue;

        faces = getFaceBoundingBoxesMTCNN(frame, MTCNN_detector)
        detections.append(faces)
        indices.append(ind)

    return detections, indices;



# Tiny Faces Face detection methods 



def overlay_bounding_boxes(raw_img, refined_bboxes, lw):

  for r in refined_bboxes:
    _score = expit(r[4])
    cm_idx = int(np.ceil(_score * 255))
    rect_color = [int(np.ceil(x * 255)) for x in util.cm_data[cm_idx]]  # parula
    _lw = lw
    if lw == 0:  # line width of each bounding box is adaptively determined.
      bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
      _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
      _lw = int(np.ceil(_lw * _score))

    _r = [int(x) for x in r[:4]]
    cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), rect_color, _lw)
    
    
def faceDetection_TinyFaces(weight_file_path, videoFile, sampling_interval, detectionFrames,  prob_thresh=0.5, nms_thresh=0.1, lw=3, display=False, newScale = 360):


    RESIZED_IMAGE_HEIGHT = newScale

    print('Resizing images to height ', RESIZED_IMAGE_HEIGHT);

    vid = cv2.VideoCapture(videoFile);
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    total_frames = int(vid.get(7))
    FPS = vid.get(cv2.CAP_PROP_FPS)

    sampling_rate = int(FPS * sampling_interval)
    # TODO : may set adaptive sampling rate based on shot length

    print(videoFile, " : total_frames:", total_frames, ", FPS:", FPS, ' SR:', sampling_rate)


    detections = [];  # face detection bounding boxes
    indices = [];  # frame number for detection


    # placeholder of input images. Currently batch size of one is supported.
    x = tf.placeholder(tf.float32, [1, None, None, 3]) # n, h, w, c

    # Create the tiny face model which weights are loaded from a pretrained model.
    model = tiny_face_model.Model(weight_file_path)
    score_final = model.tiny_face(x)


    # Load an average image and clusters(reference boxes of templates).
    with open(weight_file_path, "rb") as f:
        _, mat_params_dict = pickle.load(f)

    average_image = model.get_data_by_key("average_image")
    clusters = model.get_data_by_key("clusters")
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    normal_idx = np.where(clusters[:, 4] == 1)

    # main
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for frameInd in range(0, total_frames):

            if (frameInd%(int(total_frames/10) ) == 0):
                print( str(int(frameInd/int(total_frames/10)) )+'0 %' )

            ret, frame = vid.read()

            if (ret == False):
                continue

            # subsample frames
            if (frameInd % sampling_rate != 0 and (frameInd-2) not in detectionFrames):
                continue;

           
            raw_img = frame
            org_raw_img = frame[:]

            myScale = (float(RESIZED_IMAGE_HEIGHT)/raw_img.shape[0]);

            # print ('org:', raw_img.shape)
            raw_img = cv2.resize(raw_img, ( int(raw_img.shape[1]*myScale), RESIZED_IMAGE_HEIGHT),interpolation=cv2.INTER_CUBIC) 
            # print ('res:',raw_img.shape)

            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            raw_img_f = raw_img.astype(np.float32)

            def _calc_scales():
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
                min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                                np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
                max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM_TINYFACES))
                scales_down = frange(min_scale, 0, 1.)
                scales_up = frange(0.5, max_scale, 0.5)
                scales_pow = np.hstack((scales_down, scales_up))
                scales = np.power(2.0, scales_pow)
                return scales

            scales = _calc_scales()
            start = time.time()

            # initialize output
            bboxes = np.empty(shape=(0, 5))

            # process input at different scales
            for s in scales:
                # print("Processing {} at scale {:.4f}".format(str(frameInd), s))
                img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                img = img - average_image
                img = img[np.newaxis, :]

                # we don't run every template on every scale ids of templates to ignore
                tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
                ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

                # run through the net
                score_final_tf = sess.run(score_final, feed_dict={x: img})

                # collect scores
                score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
                prob_cls_tf = expit(score_cls_tf)
                prob_cls_tf[0, :, :, ignoredTids] = 0.0

                def _calc_bounding_boxes():
                  # threshold for detection
                  _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)

                  # interpret heatmap into bounding boxes
                  cy = fy * 8 - 1
                  cx = fx * 8 - 1
                  ch = clusters[fc, 3] - clusters[fc, 1] + 1
                  cw = clusters[fc, 2] - clusters[fc, 0] + 1

                  # extract bounding box refinement
                  Nt = clusters.shape[0]
                  tx = score_reg_tf[0, :, :, 0:Nt]
                  ty = score_reg_tf[0, :, :, Nt:2*Nt]
                  tw = score_reg_tf[0, :, :, 2*Nt:3*Nt]
                  th = score_reg_tf[0, :, :, 3*Nt:4*Nt]

                  # refine bounding boxes
                  dcx = cw * tx[fy, fx, fc]
                  dcy = ch * ty[fy, fx, fc]
                  rcx = cx + dcx
                  rcy = cy + dcy
                  rcw = cw * np.exp(tw[fy, fx, fc])
                  rch = ch * np.exp(th[fy, fx, fc])

                  scores = score_cls_tf[0, fy, fx, fc]
                  tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
                  tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
                  tmp_bboxes = tmp_bboxes.transpose()
                  return tmp_bboxes

                tmp_bboxes = _calc_bounding_boxes()
                bboxes = np.vstack((bboxes, tmp_bboxes)) # <class 'tuple'>: (5265, 5)


            print("Took {:.2f} secs for Frame {}".format(time.time() - start, str(frameInd)))

            # non maximum suppression
            # refind_idx = util.nms(bboxes, nms_thresh)
            refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                       tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                       max_output_size=bboxes.shape[0], iou_threshold=nms_thresh)
            refind_idx = sess.run(refind_idx)
            refined_bboxes = bboxes[refind_idx]

            finalBboxes = []

            for ind_bb in range(len(refined_bboxes)):
                refined_bboxes[ind_bb][0]/=myScale
                refined_bboxes[ind_bb][1]/=myScale
                refined_bboxes[ind_bb][2]/=myScale
                refined_bboxes[ind_bb][3]/=myScale
                nbbox = list( refined_bboxes[ind_bb][0:4].astype(int) )
                finalBboxes.append( nbbox )


            detections.append(finalBboxes)
            indices.append(frameInd)



    return detections, indices;



def perform_face_detection(filePath, input_folder, output_folder, interval, tinyFaces_weights = '', detectionMethod = 'mtcnn', tinyFaces_scale = 360, MTCNN_detector = None):


    shots = get_csv_data(filePath);
    stage1_data = shots[0:2];
    shots = shots[4:];
    for ss in range(len(shots)):
        shots[ss] = list(map(int, shots[ss]));

    PV(shots)

    videoPath = stage1_data[1][1];

    detectionFrames = {};
    for shot in shots:
        detectionFrames[shot[0]] = 1

 #perform face detection

    if (detectionMethod == 'mtcnn'):
        print(detectionFrames)
        detections, indices = detectFacesMTCNN(videoPath, interval, detectionFrames, MTCNN_detector)

    elif(detectionMethod == 'tiny_faces'):

      print('Using Tiny Faces model : ',tinyFaces_weights )
      with tf.Graph().as_default():
        print(detectionFrames)
        detections, indices = faceDetection_TinyFaces(tinyFaces_weights, videoPath, interval, detectionFrames, newScale = tinyFaces_scale)
    
#creating output file

    outFileName = videoPath.split('/')[-1] + '_faceDetections.csv'
    outFile = open(os.path.join(output_folder, outFileName), 'w')

    # write process information to output file
    outFile.write(','.join(['StageName', 'InputFile', 'OutputFile']) + '\n');

    outFile.write(
        ','.join([' face_detection', videoPath, os.path.join(output_folder, outFileName)]) + '\n');

    # write face detection output
    outFile.write('\n');
    outFile.write(','.join(['Frame number', 'Shot number', 'Shot end frame', 'Bounding boxes (x1_y1_x2_y2)']) + '\n');


    currShot = 1
    currShotEnd = shots[0][1];
    for ind in range(len(detections)):

        if (indices[ind] > currShotEnd):
            currShot += 1;

            currShotEnd = shots[currShot-1][1]

        outFile.write(','.join([str(indices[ind]), str(currShot), str(currShotEnd)]));

        for dd in detections[ind]:
            bb = list(map(str, dd));
            outFile.write(',' + '_'.join(bb));

        outFile.write('\n');


# invocation begins here

input_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'shot/output'))
out_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'detect/output'))

print('Begin  face detection ')


argparse = ArgumentParser()

stage_folder = '/'.join(out_folder.split('/')[:-1])

argparse.add_argument('--method', type=str, help='Algorithm for face detection. Options: \'mtcnn\' or \'tiny_faces\' ', default="mtcnn")
argparse.add_argument('--tinyFaces_weights', type=str, help='Path to pretrained weights for Tiny Faces algorithm', default=os.path.join(os.getcwd(),'weights/hr_res101.pkl'))
argparse.add_argument('--detection_interval', type=float, help='at what interval(in seconds) should we perform face detection.(default: 0.5).', default=0.5)
argparse.add_argument('--tinyFaces_scale', type=int, help='Width to which image should be resized before detection. Smaller values (240) are faster.', default=360)

args = argparse.parse_args()

# check arguments
if (args.method == 'tiny_faces'):
    assert os.path.exists(args.tinyFaces_weights), "Tiny_faces weight file: " + args.tinyFaces_weights + " not found."

assert os.path.exists(input_folder), " input folder : " + input_folder + " not found."
assert os.path.exists(out_folder), " output folder : " + out_folder + " not found."


MTCNN_detector = None;
# process videos
for f in os.listdir(input_folder):


    filePath = os.path.join(input_folder, f)
    
    # process csv files
    if (filePath[-4:] == '.csv'):
    
        print('Processing: ', filePath)

        
        if (args.method == 'mtcnn' and MTCNN_detector == None):
          MTCNN_detector = MTCNN();

        perform_face_detection(filePath, input_folder, out_folder, args.detection_interval,
          tinyFaces_weights = args.tinyFaces_weights, detectionMethod = args.method, tinyFaces_scale = args.tinyFaces_scale, MTCNN_detector = MTCNN_detector);


print('End of face detection')


