import cv2
import glob
import numpy as np
from moviepy.editor import VideoFileClip
import sys

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

image_folder = "frames/"
video_file = 'processed_video.avi'
writer = cv2.VideoWriter("{}".format(video_file), cv2.VideoWriter_fourcc(*"MJPG"), 25., (1280,720))

images = glob.glob('{}*jpg'.format(image_folder))
print('converting images to video..')
for idx, image in enumerate(sorted(images)):
    progress(idx+1, len(images))
    img = cv2.imread(image)
    try:
        # video.write(img)
        writer.write(np.asarray(img))
    except Exception as e:
        print(e)
print('')

writer.release()
