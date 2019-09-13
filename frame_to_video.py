import cv2

import glob

def convert_frames_to_video(vipathIn, vipathOut, fps):
    img_array = []
    for filename in sorted(glob.glob(vipathIn)):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(vipathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def main():
    convert_frames_to_video('./output/*.png', './output/videos/output.mp4', 5)

if __name__ == '__main__':
        main()


