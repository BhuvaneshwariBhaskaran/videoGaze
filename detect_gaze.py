import cv2
import glob
import os
from mtcnn.mtcnn import MTCNN
import time
import torch
import math
from torch.nn import DataParallel
from torch.autograd import Variable
from gazenet import GazeNet
import numpy as np
from utils import data_transforms
from PIL import Image, ImageOps

from detector import detect_faces
from visualization_utils import show_results




net = GazeNet()
net = DataParallel(net)
net.cpu()
torch.device('cpu')
pretrained_dict = torch.load('./savedmodels/pretrained_model.pkl', map_location=lambda storage, loc: storage)
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

def draw_result(frame,eye, gaze_point,color):

    x1, y1 = eye
    x2, y2 = gaze_point

    image_height, image_width = frame.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    itbuffer= createLineIterator((x1,y1),(x2,y2),frame)

    if(len(itbuffer)):
        x2,y2=itbuffer[int(itbuffer.shape[0]/2)][0],itbuffer[int(itbuffer.shape[0]/2)][1]

    cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, 3)
    #cv2.circle(frame,(x2,y2),10,color,-1)



def createLineIterator(P1, P2, img):

   #define local variables for readability
   imageH = img.shape[0]
   imageW = img.shape[1]
   P1X = P1[0]
   P1Y = P1[1]
   P2X = P2[0]
   P2Y = P2[1]

   #difference and absolute difference between points
   #used to calculate slope and relative location between points
   dX = P2X - P1X
   dY = P2Y - P1Y
   dXa = np.abs(dX)
   dYa = np.abs(dY)

   #predefine numpy array for output based on distance between points
   itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
   itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
   negY = P1Y > P2Y
   negX = P1X > P2X
   if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
   elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
   else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX/dY
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY/dX
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

   #Remove points outside of image
   colX = itbuffer[:,0]
   colY = itbuffer[:,1]
   itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

   #Get intensities from img ndarray
   #itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

   return itbuffer


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


def detect_target(frame,eyes):
    net.eval()
    data = preprocess_image(frame, eyes)
    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data[
        'eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cpu(), volatile=True),
                                                      [image, face_image, gaze_field, eye_position])

    _, predict_heatmap = net([image, face_image, gaze_field, eye_position])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])

    return f_point[0], f_point[1]


def preprocess_image(frame, eyes):
    # crop face
    x_c, y_c = eyes

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

    h, w = frame.shape[:2]
    face_image = frame[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]

    # process face_image for face net
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = data_transforms['test'](face_image)
    # process image for saliency net
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    # generate gaze field
    gaze_field = generate_data_field(eye_point=eyes)
    sample = {'image': image,
              'face_image': face_image,
              'eye_position': torch.FloatTensor(eyes),
              'gaze_field': torch.from_numpy(gaze_field)}

    return sample


def face_orientation(frame, landmarks):
    size = frame.shape  # (height, width, color_channel)

    image_points = np.array([
        (landmarks[4], landmarks[5]),  # Nose tip
        (landmarks[0], landmarks[1]),  # Left eye left corner
        (landmarks[2], landmarks[3]),  # Right eye right corne
        (landmarks[6], landmarks[7]),  # Left Mouth corner
        (landmarks[8], landmarks[9])  # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals

    center = (size[1] / 2, size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[4], landmarks[5])

def normalize_eyeCords(x1,y1,x2,y2,width,height):
    eyecoords = []
    ey_x = (x1 + x2) / 2
    ey_y = (y1 + y2) / 2

    # print("Face: ", x1, x2, y1, y2)
    ey_x = int(np.floor((ey_x / float(width)) * 13.0))
    ey_y = int(np.floor((ey_y / float(height)) * 13.0))
    if ey_x >= 12:
        ey_x = 12
    if ey_y >= 12:
        ey_y = 12
    eye = np.zeros((13, 13))
    eye[ey_y][ey_x] = 1
    eye = torch.from_numpy(eye)
    eye = eye.view(1, 1, 13, 13)
    ix = (x1 + x2) / (2.0 * float(width))
    iy = (y1 + y2) / (2.0 * float(height))
    eyecoords.append((ix, iy))
    return eyecoords

def enormalize(leye,reye,width,height):
    l_x, l_y = leye
    r_x,r_y =  reye
    leyecoords = []
    reyecoords=[]

    # print("Face: ", x1, x2, y1, y2)
    ey_x = int(np.floor((l_x / float(width)) * 13.0))
    ey_y = int(np.floor((l_y / float(height)) * 13.0))
    if ey_x >= 12:
        ey_x = 12
    if ey_y >= 12:
        ey_y = 12
    eye = np.zeros((13, 13))
    eye[ey_y][ey_x] = 1
    eye = torch.from_numpy(eye)
    eye = eye.view(1, 1, 13, 13)
    lx = (l_x) *2/ (2.0 * float(width))
    ly = (l_y)*2 / (2.0 * float(height))
    leyecoords.append((lx, ly))

    ey_x = int(np.floor((r_x / float(width)) * 13.0))
    ey_y = int(np.floor((r_y / float(height)) * 13.0))
    if ey_x >= 12:
        ey_x = 12
    if ey_y >= 12:
        ey_y = 12
    eye = np.zeros((13, 13))
    eye[ey_y][ey_x] = 1
    eye = torch.from_numpy(eye)
    eye = eye.view(1, 1, 13, 13)
    rx = (r_x) * 2 / (2.0 * float(width))
    ry = (r_y) * 2 / (2.0 * float(height))
    reyecoords.append((rx, ry))

    return leyecoords,reyecoords

def r_normalize_eyeCords(x1,y1,x2,y2,width,height):
    eyecoords = []
    ey_x = (x1 + x2) / 2
    ey_y = (y1 + y2) / 2

    # print("Face: ", x1, x2, y1, y2)
    ey_x = int(np.floor((ey_x / float(width)) * 13.0))
    ey_y = int(np.floor((ey_y / float(height)) * 13.0))
    if ey_x >= 12:
        ey_x = 12
    if ey_y >= 12:
        ey_y = 12
    eye = np.zeros((13, 13))
    eye[ey_y][ey_x] = 1
    eye = torch.from_numpy(eye)
    eye = eye.view(1, 1, 13, 13)
    ix = (x1 + x2) / (2.0 * float(width))
    iy = (y1 + y2) / (2.0 * float(height))
    eyecoords.append((ix, iy))
    return eyecoords


def image_gaze(img_dir):
    for image in glob.glob(img_dir):

        head, tail = os.path.split(image)
        # stage1
        img = Image.open(image)  # modify the image path to yours
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        bounding_boxes, landmarks = detect_faces(img)  # detect bboxes and landmarks for all faces in the image
        h, w = image.shape[:2]
        for i in range(len(bounding_boxes)):
            #imgpts, modelpts, rotate_degree, nose = face_orientation(image, landmarks[i])
            #random_color = tuple(np.random.random_integers(0, 255, size=3))
            cv2.rectangle(image, (int(bounding_boxes[i][0]), int(bounding_boxes[i][1])),(int(bounding_boxes[i][2]),int(bounding_boxes[i][3])), (0, 155, 255), 2)
            headpose_center_cords = normalize_eyeCords(int(bounding_boxes[i][0]), int(bounding_boxes[i][1]),int(bounding_boxes[i][2]) , int(bounding_boxes[i][3]), w,h)
            px, py = detect_target(image, (headpose_center_cords[0][0], headpose_center_cords[0][1]))
            draw_result(image, (headpose_center_cords[0][0], headpose_center_cords[0][1]), (px, py),[0, 0, 255])
        #stage2
        result = detector.detect_faces(image)

        if result != []:
            for person in result:
                #imgpts, modelpts, rotate_degree, nose = face_orientation(image,person['keypoints'])
                #cv2.line(image, nose, tuple(imgpts[1].ravel()), (0, 255, 0), 2)  # GREEN
                #cv2.line(image, nose, tuple(imgpts[0].ravel()), (255, 0,), 2)  # BLUE
                #cv2.line(image, nose, tuple(imgpts[2].ravel()), (0, 0, 255), 2)  # RED
                confidence = person['confidence']
                if (confidence > 0.90):
                    bounding_box = person['box']
                    keypoints = person['keypoints']
                    #commented cv2 drawing
                    #cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),(0, 155, 255),2)
                    #cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
                    #cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)

                    leyecoords, reyecoords = enormalize(keypoints['left_eye'], keypoints['right_eye'], w,h)
                    lpx, lpy = detect_target(image, (leyecoords[0][0], leyecoords[0][1]))
                    rpx, rpy = detect_target(image, (reyecoords[0][0], reyecoords[0][1]))
                    draw_result(image, (leyecoords[0][0], leyecoords[0][1]), (lpx, lpy),[255, 0, 0])
                    draw_result(image, (reyecoords[0][0], reyecoords[0][1]), (rpx, rpy),[0, 255, 255])


        cv2.imwrite("./output/"+tail, image)
        cv2.waitKey(0)


def video_gaze(vid_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(vid_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Current video fps {0}, WxH {1}'.format(fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('./output/videos/result.mp4', cv2.VideoWriter_fourcc('X', '2', '6', '4'), 30, (frame_width,
                                                                                          frame_height))

    while (True):
        ret, frame = cap.read()

        if ret == True:
            result = detector.detect_faces(frame)

            if result != []:
                for person in result:
                    #imgpts, modelpts, rotate_degree, nose = face_orientation(frame, person['keypoints'])
                    #cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
                    #cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255, 0,), 3)  # BLUE
                    #cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED
                    confidence = person['confidence']
                    if(confidence>0.93):
                        bounding_box = person['box']
                        keypoints = person['keypoints']
                        cv2.rectangle(frame,
                                      (bounding_box[0], bounding_box[1]),
                                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                      (0, 155, 255),
                                      2)
                        cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 2)
                        cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 2)
                        headpose_center_cords = normalize_eyeCords(bounding_box[0], bounding_box[1],bounding_box[0] + bounding_box[2],
                                                       bounding_box[1] + bounding_box[3],frame_width,frame_height)
                        leyecoords,reyecoords = enormalize(keypoints['left_eye'],keypoints['right_eye'], frame_width, frame_height)
                        lpx, lpy = detect_target(frame, (leyecoords[0][0], leyecoords[0][1]))
                        rpx, rpy = detect_target(frame, (reyecoords[0][0], reyecoords[0][1]))
                        px,py=detect_target(frame,(headpose_center_cords[0][0],headpose_center_cords[0][1]))
                        draw_result(frame, (leyecoords[0][0], leyecoords[0][1]), (lpx, lpy), [255, 0, 0])
                        draw_result(frame, (reyecoords[0][0], reyecoords[0][1]), (rpx, rpy), [0, 255, 255])
                        #draw_result(frame,(headpose_center_cords[0][0],headpose_center_cords[0][1]),(px,py))

            out.write(frame)
            # Display the resulting frame
            # cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def main():
    global detector
    global pbar
    start = time.time()
    detector = MTCNN()
    img_dir = 'imgs/test/*'
    vid_path='./videos/1.mp4'
    #video_gaze(vid_path)
    image_gaze(img_dir)
    print("Process time: " + str((time.time() - start)))

if __name__ == '__main__':
    main()