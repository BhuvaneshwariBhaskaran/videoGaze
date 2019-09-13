import os
import glob


def clear_frames(filepath):
    files = os.listdir(filepath)
    for f in files:
        if (f.endswith(".mp4") or f.endswith(".png")or f.endswith(".jpg")or f.endswith(".JPG")or f.endswith(".jpeg")):
            os.remove(filepath + f)

def extract_frames(output_dir):
    for file_name in glob.glob('videos/*'):
        if not (file_name.endswith(".mp4") or file_name.endswith(".mov")):
            continue
        video = file_name.split('/')[-1]
        path = os.path.join(output_dir)

        if not os.path.exists(path):
            os.makedirs(path)
            os.remove()
        os.system('ffmpeg -i '+ '\"'+ file_name+'\"' + ' -r 2 ' + '\"' +path + '/%05d.png' + '\"')

def main():
    output_dir = './imgs/test/'
    clear_frames(output_dir)
    clear_frames('./output/')
    extract_frames(output_dir)

if __name__ == '__main__':
        main()

