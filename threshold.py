import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import glob
import os
import ntpath
from utils import gradient_mask, hls_mask, show_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='test_images', 
                        help='directory to read calibration image files from')
    parser.add_argument('--debug', type=int, default=0, 
                        help='print images to screen')
    parser.add_argument('--outputdir', type=str, default='output_images', 
                        help='directory to write images to')

    FLAGS, unparsed = parser.parse_known_args()

    #test png
    image = mpimg.imread('signs_vehicles_xygrad.png')
    # Apply each of the thresholding functions
    gradx = gradient_mask(image, orient='x', sobel_kernel=9, thresh=(20, 100))
    if FLAGS.debug == 1:
        show_images(image, gradx, 'Sobel X')
    s_binary = hls_mask( image, thresh=(90,255), channel=2 )
    if FLAGS.debug == 1:
        show_images(image, s_binary, 'Saturation (HLS)')
    h_binary = hls_mask( image, thresh=(20,30), channel=0 )
    if FLAGS.debug == 1:
        show_images(image, h_binary, 'Hue (HLS)')
    hsg_channels = np.dstack((gradx, h_binary, s_binary)).astype('uint8') * 255
    hsg_binary = np.zeros_like(gradx)
    hsg_binary[(h_binary == 1) & (s_binary == 1) | (gradx == 1)] = 1
    hsg_mask = np.dstack((hsg_binary, hsg_binary, hsg_binary)).astype('uint8') * 255
    if FLAGS.debug == 1:
        show_images( hsg_channels, hsg_mask, 'HSG Mask', 'HSG Channels')
    else:
        name = 'signs_vehicles_xygrad.png'
        print('writing {}/thresh_channels_{}'.format(FLAGS.outputdir, name))
        cv2.imwrite('{}/thresh_channels_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(hsg_channels, cv2.COLOR_BGR2RGB))
        cv2.imwrite('{}/thresh_mask_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(hsg_mask, cv2.COLOR_BGR2RGB))        

    #print calibration image differences
    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image = mpimg.imread(fname)
        # Apply each of the thresholding functions
        gradx = gradient_mask(image, orient='x', sobel_kernel=9, thresh=(20, 100))
        if FLAGS.debug == 1:
            show_images(image, gradx, 'Sobel X')
        s_binary = hls_mask( image, thresh=(90,255), channel=2 )
        if FLAGS.debug == 1:
            show_images(image, s_binary, 'Saturation (HLS)')
        h_binary = hls_mask( image, thresh=(20,30), channel=0 )
        if FLAGS.debug == 1:
            show_images(image, h_binary, 'Hue (HLS)')
        hsg_channels = np.dstack((gradx, h_binary, s_binary)).astype('uint8') * 255
        hsg_binary = np.zeros_like(gradx)
        hsg_binary[(h_binary == 1) & (s_binary == 1) | (gradx == 1)] = 1
        hsg_mask = np.dstack((hsg_binary, hsg_binary, hsg_binary)).astype('uint8') * 255
        if FLAGS.debug == 1:
            show_images( hsg_channels, hsg_mask, 'HSG Mask', 'HSG Channels')
        else:
            head, tail = ntpath.split(fname)
            name = tail or ntpath.basename(head)
            print('writing {}/thresh_channels_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/thresh_channels_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(hsg_channels, cv2.COLOR_BGR2RGB))
            cv2.imwrite('{}/thresh_mask_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(hsg_mask, cv2.COLOR_BGR2RGB))
