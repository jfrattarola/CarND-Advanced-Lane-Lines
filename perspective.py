import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import glob
import os
import ntpath
from utils import lane_mask, show_images
from camera_cal import camera_cal_init

def transform_perspective(image, 
                          src_points,
                          dest_points,
                          object_points=None, 
                          image_points=None):
    if object_points == None or image_points == None:
        object_points, image_points = camera_cal_init(FLAGS.caldir)

    #calibrate camera
    image_size = (image.shape[1], image.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None ,None)
    
    #get undistorted image
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

    #get perspective transform matrix
    perspective_transform = cv2.getPerspectiveTransform(src_points, dest_points)

    #warp image
    warped_image = cv2.warpPerspective(image, perspective_transform, image_size, flags=cv2.INTER_LINEAR)

    return warped_image

def get_warped_binary(image_name, src_points, dest_points, object_points=None, image_points=None):
    image = mpimg.imread(image_name)
    mask = lane_mask(image)

    warped = transform_perspective(mask, 
                                   src_points,
                                   dest_points,
                                   object_points, 
                                   image_points)
    return image, warped

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

    #calibrate camera
    object_points, image_points = camera_cal_init(FLAGS.caldir)

    s = np.float32([[685, 450], [1075, 705], [230, 705], [600, 450]])
    d = np.float32([[960, 0], [960, 720], [320, 720], [320, 0]])

    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image, warped = get_warped_binary( fname, s, d )
        draw_lines(image, s)
        draw_lines(warped, d)
        if FLAGS.debug == 1:
            show_images(image, warped, 'Warped', fname)
        else:
            head, tail = ntpath.split(fname)
            name = tail or ntpath.basename(head)
            print('writing {}/perspective_normal_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/perspective_normal_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.imwrite('{}/perspective_warped_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
