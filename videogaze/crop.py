
import os
import sys
import subprocess
import cv2
import csv




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


def do_face_cropping(filename, filePath, output_folder):
    """
    Execute face cropping  for face detection file
    """
    program_location = os.path.join(os.getcwd(), 'crop/faceCrop.py')
    c1 = 'python \'' + program_location + '\' \'' + filePath + '\' ' + output_folder;

    # execute command

    print(c1)
    subprocess.call(c1, shell=True)


# invocation begins here

input_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'track/output'))
out_folder = removeTrailingBackslash(os.path.join(os.getcwd(),'crop/output'))



print('Begin face cropping')

assert os.path.exists(input_folder), "input folder : " + input_folder + " not found."
assert os.path.exists(out_folder), "output folder : " + out_folder + " not found."


# process files
for f in os.listdir(input_folder):
    # print (i);
    filePath = os.path.join(input_folder, f)

    # process csv files
    if (filePath[-4:] == '.csv'):
        print('Processing: ', filePath)

        do_face_cropping(f, filePath, out_folder);

print('End of face cropping')
