
import os
import sys
import subprocess
import cv2
import csv



openface_FaceLandmarkVidMulti_path = '/Users/bhuvaneshwaribhaskaran/PycharmProjects/oo/OpenFace/build/bin/FaceLandmarkVidMulti'

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


def do_feature_extraction(input_folder, output_folder):
    """
    Execute feature extraction program using OpenFace
    """
    program_location = os.path.join(os.getcwd(),'extract/featureExtract.py')
    c1 = 'python \'' + program_location + '\' \'' + openface_FaceLandmarkVidMulti_path + '\' \'' + input_folder + '\' \'' + output_folder + '\'';

    # execute command

    print(c1)
    subprocess.call(c1, shell=True)

input_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'crop/output'))
out_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'extract/output'))



print('Begin feature extraction ')


assert os.path.exists(input_folder), " input folder : " + input_folder + " not found."
assert os.path.exists(out_folder), " output folder : " + out_folder + " not found."


# process files
do_feature_extraction(input_folder, out_folder);

print('End of feature extraction ')
