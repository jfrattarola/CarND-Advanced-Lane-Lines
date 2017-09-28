import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import glob
import os
from all_funcs import gradient_mask, hls_mask, show_images
from camera_cal import camera_cal_init
from perspective import transform_perspective

def get_lanes_full_search(binary_warped):
    # Take a histogram of the bottom half of the image binary birdeye image
    # NB: binary_warped is a 3 channel image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :, 0], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = binary_warped.copy()
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to re-center window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

        # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low ,win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return lefty, leftx, righty, rightx, out_img

def get_lanes_from_previous(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return lefty, leftx, righty, rightx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caldir', type=str, default='camera_cal', 
                        help='directory to read calibration image files from')
    parser.add_argument('--dir', type=str, default='test_images', 
                        help='directory to read test image files from')
    FLAGS, unparsed = parser.parse_known_args()

    #calibrate camera
    object_points, image_points = camera_cal_init(FLAGS.caldir)

    s = np.float32([[685, 450], [1075, 705], [230, 705], [600, 450]])
    d = np.float32([[960, 0], [960, 720], [320, 720], [320, 0]])

    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for test_image, fname in enumerate(images):
        image = mpimg.imread(fname)
        gradx = gradient_mask(image, orient='x', sobel_kernel=9, thresh=(20, 100))
        s_binary = hls_mask( image, thresh=(90,255), channel=2 )
        h_binary = hls_mask( image, thresh=(20,30), channel=0 )
        hsg_binary = np.zeros_like(gradx)
        hsg_binary[(h_binary == 1) & (s_binary == 1) | (gradx == 1)] = 1
        thresh_bin = np.dstack((hsg_binary, hsg_binary, hsg_binary)).astype('uint8') * 255

        warped = transform_perspective(thresh_bin, 
                                       src_points=s,
                                       dest_points=d,
                                       object_points=object_points, 
                                       image_points=image_points)

        # Full search
        lefty, leftx, righty, rightx, out_img = get_lanes_full_search(warped)
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Use data from previous frame
        lefty, leftx, righty, rightx = get_lanes_from_previous(warped, left_fit, right_fit)
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Draw curve on figure
        left_lane_dots = zip(list(ploty), list(left_fitx))
        right_lane_dots = zip(list(ploty), list(right_fitx))
        for l in list(left_lane_dots):
            cv2.circle(out_img, (int(l[1]), int(l[0])), 2, (0, 255, 255))
        for r in list(right_lane_dots):
            cv2.circle(out_img, (int(r[1]), int(r[0])), 2, (0, 255, 255))

        cv2.imwrite('output_images/test{}_with_lanes.jpg'.format(test_image), out_img)
