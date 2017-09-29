import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import glob
import os
import ntpath
from utils import show_images

def gradient_mask(img, orient='x', sobel_kernel=3, thresh=(0, 255), should_gray=True):
    #set x/y params based on orient arg
    x = 1 if orient is 'x' else 0
    y = 1 if x == 0 else 0

    #convert to grayscale
    if should_gray is True:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #take derivitive in x or y, given orient
    sobel = cv2.Sobel(img, cv2.CV_64F, x, y)

    #take the absolute value of the derivative
    sobel_abs = np.absolute(sobel)

    #scale to 8-bit (0-255) and convert to uint8
    scaled_sobel = np.uint8(255 * sobel_abs / np.max(sobel_abs)) if sobel_abs.dtype != 'uint8' else sobel_abs

    #create a mask when the scaled gradient (derivative) is between the min/max thresholds
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #take derivitives of x and y, separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    #calculate the magnitude
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))

    #scale the magnitude to 8bit (0-255)
    scaled_sobel_mag = np.uint8( 255 * sobel_mag / np.max(sobel_mag) )

    #create a mask when the scaled magnitude is between the min/max thresholds
    mag_binary = np.zeros_like( scaled_sobel_mag )
    mag_binary[(scaled_sobel_mag >= mag_thresh[0]) & (scaled_sobel_mag <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #take derivitives of x and y, separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # take the absolute value of the x and y gradients
    abs_sobelx = np.sqrt(np.square(sobelx))
    abs_sobely = np.sqrt(np.square(sobely))

    #get the arctan (inverse tangent) to calculate the direction of the gradient
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)

    #create a binary mask where the thresholds are met
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

def _hls_mask(channel, thresh):
    #convert to 8 bit (0-255)
    scaled_S = np.uint8( 255 * channel / np.max(channel) ) if channel.dtype != 'uint8' else channel
    
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(scaled_S)
    binary_output[(scaled_S > thresh[0]) & (scaled_S <= thresh[1])] = 1
    return binary_output

def hls_mask(img, thresh=(0, 255), channel=2):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # 2) Apply a threshold to the S channel
    S = hls[:,:,channel]

    binary_output = _hls_mask( S, thresh )

    return binary_output

def combined_sgray(img, sobel_kernel=9, grad_thresh=(20,100), s_thresh=(170,255)):
    #get gradient mask for x
    sxbinary = gradient_mask(img, orient='x', sobel_kernel=sobel_kernel, thresh=grad_thresh)

    #get saturation mask
    s_binary = hls_mask(img, thresh = s_thresh, channel=2)

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary

def lane_mask(img):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # 2) Apply a threshold to the S channel
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    l_output = _hls_mask( l_channel, (0,255) )

#    gradx = gradient_mask(l_output, orient='x', sobel_kernel=9, thresh=(20, 255), should_gray=False)
    gradx = gradient_mask(img, orient='x', sobel_kernel=9, thresh=(20,100))
    s_binary = _hls_mask( s_channel, thresh=(120,255))
    l_binary = _hls_mask( l_channel, thresh=(40,255) )
    lsg_binary = np.zeros_like(gradx)
    
    lsg_binary[(l_binary == 1) & (s_binary == 1) | (gradx == 1)] = 1
    thresh_binary = np.dstack((lsg_binary, lsg_binary, lsg_binary)).astype('uint8') * 255
    color_binary = np.dstack((s_binary, lsg_binary, l_binary)).astype('uint8') * 255

    return thresh_binary, color_binary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='test_images', 
                        help='directory to read calibration image files from')
    parser.add_argument('--debug', type=int, default=0, 
                        help='print images to screen')
    parser.add_argument('--outputdir', type=str, default='output_images', 
                        help='directory to write images to')

    FLAGS, unparsed = parser.parse_known_args()
    #print calibration image differences
    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image = mpimg.imread(fname)
        mask, channels = lane_mask(image)
        if FLAGS.debug == 1:
            show_images( channels, mask, 'Mask', 'Channels')
        else:
            head, tail = ntpath.split(fname)
            name = tail or ntpath.basename(head)
            print('writing {}/thresh_channels_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/thresh_channels_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(channels, cv2.COLOR_BGR2RGB))
            print('writing {}/thresh_mask_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/thresh_mask_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

