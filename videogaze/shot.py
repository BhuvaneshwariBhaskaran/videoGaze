
# Stage 1 - Segment shots from video


import os
import sys
import subprocess
import cv2


def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;




from pyannote.video import Video
from pyannote.video import Shot, Thread

def pyannote_shot(videoPath):

    video = Video(videoPath)
    FPS = video._fps
    shots = Shot(video)
    shotBoundaries = []

    for x in shots:
        shotBoundaries.append([int(FPS * x.start), int(FPS * x.end)])

    return shotBoundaries

def perform_shot_segmentation(filename, filePath, input_folder, output_folder):
#shot segment output
    output = pyannote_shot(filePath)


#create a output file csv
    outFileName = filename + '_shots.csv'
    outFile = open(os.path.join(output_folder, outFileName), 'w')

#appends headers
    outFile.write(','.join(['StageName', 'InputFile', 'OutputFile']) + '\n');
#appends data
    outFile.write(','.join(['shot_segmentation', os.path.join(input_folder, filename),
                            os.path.join(output_folder, outFileName)]) + '\n');


    outFile.write('\n');
    outFile.write(','.join(['Shot start', 'Shot end']) + '\n');
    for segment in output:
        outFile.write(','.join([str(segment[0]), str(segment[1])]) + '\n');

#Begins here

in_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'videoin'))
out_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'shot/output'))

print('Begin  shot segmentation ')

assert os.path.exists(in_folder), "shot input folder : " + in_folder + " not found."
assert os.path.exists(out_folder), "shot output folder : " + out_folder + " not found."


for f in os.listdir(in_folder):
    filePath = os.path.join(in_folder, f)
    print('Processing: ', filePath)

    perform_shot_segmentation(f, filePath, in_folder, out_folder);

print('End of shot segmentation ')
