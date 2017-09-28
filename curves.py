import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import glob
import os
import ntpath
from utils import gradient_mask, hls_mask, show_images
from camera_cal import camera_cal_init
from perspective import get_warped_binary
from lanes import Lane, get_lanes, get_curves_and_offset, get_curves_in_pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caldir', type=str, default='camera_cal', 
                        help='directory to read calibration image files from')
    parser.add_argument('--dir', type=str, default='test_images', 
                        help='directory to read test image files from')
    parser.add_argument('--outputdir', type=str, default='output_images', 
                        help='directory to write images to')
    FLAGS, unparsed = parser.parse_known_args()

    #calibrate camera
    object_points, image_points = camera_cal_init(FLAGS.caldir)

    s = np.float32([[685, 450], [1075, 705], [230, 705], [600, 450]])
    d = np.float32([[960, 0], [960, 720], [320, 720], [320, 0]])

    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image, warped = get_warped_binary(fname, s, d, object_points, image_points)
        lanes = get_lanes(warped)

        out = get_curves_and_offset(lanes, warped)
        left_curverad, right_curverad, center_offset_meters, center_offset_pixels = out
        print('left_curverad {:.2f} meters - right_curverad: {:.2f} meters'.format(left_curverad, right_curverad))
        if center_offset_meters < 0:
            print('offset: {:.3f} meters to the left'.format(abs(center_offset_meters)))
        else:
            print('offset: {:.3f} meters to the right'.format(abs(center_offset_meters)))

        out = get_curves_in_pixels(lanes, warped)
        left_curverad_pixels, right_curverad_pixels = out
        print('left_curverad_pixels {:.2f} pixels - right_curverad_pixels: {:.2f} pixels'.format(left_curverad_pixels, right_curverad_pixels))
        if center_offset_pixels < 0:
            print('offset: {:.3f} pixels to the left'.format(abs(center_offset_pixels)))
        else:
            print('offset: {:.3f} pixels to the right'.format(abs(center_offset_pixels)))

        print('Pixel to meters ration in radii: {}'.format(left_curverad_pixels/left_curverad))
        print('Pixel to meters ration in radii: {}'.format(right_curverad_pixels/right_curverad))

        print('')
