import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import glob
import os
import ntpath
from utils import show_images
from threshold import lane_mask
from camera_cal import Camera


class Transform:
    DEFAULT_SRC=np.float32([[685, 450], [1075, 705], [230, 705], [600, 450]])
    DEFAULT_DEST=np.float32([[960, 0], [960, 720], [320, 720], [320, 0]])

    def __init__(self, camera, src_points, dest_points):
        self.camera = camera
        self.src = src_points
        self.dst = dest_points

    def birdseye(self, image, thresh=True):
        #get undistorted image
        undistorted_image = self.camera.undistort(image)

        img = undistorted_image

        # get mask
        if thresh is True:
            img, _ = lane_mask(undistorted_image)

        #get perspective transform matrix
        perspective_transform = cv2.getPerspectiveTransform(self.src, self.dst)

        #warp image
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, perspective_transform, img_size, flags=cv2.INTER_LINEAR)

        return warped, img


def draw_lines(image, points, color=(255,0,0)):
    cv2.polylines(image, [np.asarray(points, np.int32)], True, color, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caldir', type=str, default='camera_cal', 
                        help='directory to read calibration image files from')
    parser.add_argument('--dir', type=str, default='test_images', 
                        help='directory to read test image files from')
    parser.add_argument('--debug', type=int, default=0, 
                        help='print images to screen')
    parser.add_argument('--outputdir', type=str, default='output_images', 
                        help='directory to write images to')

    FLAGS, unparsed = parser.parse_known_args()

    camera = Camera(FLAGS.caldir)
    transform = Transform(camera, Transform.DEFAULT_SRC, Transform.DEFAULT_DEST)

    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image = mpimg.imread(fname)
        thresh_image = image.copy()

        warped, undist = transform.birdseye(image, False)

        draw_lines(undist, transform.src)
        draw_lines(warped, transform.dst)

        thresh_warped, mask = transform.birdseye(thresh_image)

        if FLAGS.debug == 1:
            show_images(undist, warped, 'Warped', fname)
            show_images(mask, thresh_warped, 'Warped', fname)
        else:
            head, tail = ntpath.split(fname)
            name = tail or ntpath.basename(head)
            print('writing {}/perspective_normal_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/perspective_normal_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
            print('writing {}/perspective_warped_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/perspective_warped_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

            print('writing {}/perspective_thresh_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/perspective_thresh_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            print('writing {}/perspective_normal_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/perspective_thresh_warped_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(thresh_warped, cv2.COLOR_BGR2RGB))
