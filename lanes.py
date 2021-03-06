import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import glob
import os
import ntpath
from utils import show_images
from camera_cal import Camera
from perspective import Transform

class Lane:

    def __init__(self, camera, transform, image, alpha=0.1, beta=0.9):
        self.camera = camera
        self.transform = transform
        self.img = image
        self.alpha = alpha
        self.beta = beta
        self.leftx=None
        self.lefty=None
        self.rightx=None
        self.righty=None
        self.left_fit=None
        self.right_fit=None
        self.prev_left_fit=None
        self.prev_right_fit=None
        self.out_img=None
        self.nwindows=9        # Choose the number of sliding windows
        self.margin = 100 # Set the width of the windows +/- margin
        self.minpix = 50 # Set minimum number of pixels found to re-center window
        self.left_curverad=None
        self.right_curverad=None 
        self.left_curverad_pixels=None
        self.right_curverad_pixels=None 
        self.center_offset_meters=None 
        self.prev_center_offset_pixels=None
        self.center_offset_pixels=None

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0]/2):, :, 0], axis=0)

        # Create an output image to draw on and  visualize the result
        self.out_img = image.copy()
    
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(image.shape[0]/self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.img.shape[0] - (window+1) * window_height
            win_y_high = self.img.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low ,win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > self.minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        # Create vizualization
        self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    def advance_next_lane(self, image, smooth=True):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "warped")
        # It's now much easier to find line pixels!
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy +
                                       self.left_fit[2] - self.margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) +
                                                                                  self.left_fit[1]*nonzeroy + self.left_fit[2] + self.margin)))

        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy +
                                        self.right_fit[2] - self.margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) +
                                                                                    self.right_fit[1]*nonzeroy + self.right_fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.prev_left_fit = self.left_fit
        self.prev_right_fit = self.right_fit
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        if smooth is True:
            self.left_fit = self.alpha * self.prev_left_fit + self.beta * self.left_fit
            self.right_fit = self.alpha * self.prev_right_fit + self.beta * self.right_fit
            

    def draw_curve(self):
        out_img = self.out_img.copy()

        # get x and y plot values
        ploty = np.linspace(0, self.img.shape[0]-1, self.img.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # draw curve
        left_lane_dots = zip(list(ploty), list(left_fitx))
        right_lane_dots = zip(list(ploty), list(right_fitx))
        for l in list(left_lane_dots):
            cv2.circle(out_img, (int(l[1]), int(l[0])), 2, (0, 255, 255))
        for r in list(right_lane_dots):
            cv2.circle(out_img, (int(r[1]), int(r[0])), 2, (0, 255, 255))

        return out_img

    def get_curves_and_offset(self):
        ploty = np.linspace(0, self.img.shape[0]-1, self.img.shape[0])
        y_eval = np.max(ploty)
        
        # Generate points
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.lefty*ym_per_pix, self.leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty*ym_per_pix, self.rightx*xm_per_pix, 2)

        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Calcualate vehicle offset
        camera_position = self.img.shape[1] / 2.
        lane_center = (right_fitx[-1] + left_fitx[-1]) / 2.
        self.center_offset_pixels = camera_position - lane_center
        self.center_offset_meters = xm_per_pix * self.center_offset_pixels


    def get_curves_in_pixels(self):
        ploty = np.linspace(0, self.img.shape[0]-1, self.img.shape[0])
        y_eval = np.max(ploty)

        # Calculate the new radii of curvature
        self.left_curverad_pixels = ((1 + (2*lanes.left_fit[0]*y_eval + lanes.left_fit[1])**2)**1.5) / np.absolute(2*lanes.left_fit[0])
        self.right_curverad_pixels = ((1 + (2*lanes.right_fit[0]*y_eval + lanes.right_fit[1])**2)**1.5) / np.absolute(2*lanes.right_fit[0])



    def draw_lane(self, img):
        self.get_curves_and_offset()

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.img.shape[0]-1, self.img.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Create an image to draw the lines on
        draw_img = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(draw_img, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix
        ip_matrix = self.transform.get_inverse_perspective_matrix()
        new_warped = cv2.warpPerspective(draw_img, ip_matrix, (img.shape[1], img.shape[0]))

        # Combine the result with the original image
        out_img = cv2.addWeighted(img, 1, new_warped, 0.3, 0)

        # Add info about radius and offset
        font = cv2.FONT_HERSHEY_DUPLEX
        text1 = 'left_curverad {:5.2f} meters - right_curverad: {:5.2f}'.format(self.left_curverad, self.right_curverad)
        if self.center_offset_meters < 0:
            text2 = 'offset: {:2.3f} meters to the left'.format(abs(self.center_offset_meters))
        else:
            text2 = 'offset: {:2.3f} meters to the right'.format(abs(self.center_offset_meters))

        cv2.putText(out_img, text1, (50, 30), font, 1, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(out_img, text2, (50, 70), font, 1, (0,0,255), 1, cv2.LINE_AA)

        return out_img



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caldir', type=str, default='camera_cal', 
                        help='directory to read calibration image files from')
    parser.add_argument('--dir', type=str, default='test_images', 
                        help='directory to read test image files from')
    parser.add_argument('--outputdir', type=str, default='output_images', 
                        help='directory to write images to')
    parser.add_argument('--debug', type=int, default=0, 
                        help='print images to screen')
    FLAGS, unparsed = parser.parse_known_args()

    camera = Camera(FLAGS.caldir)
    transform = Transform(camera, Transform.DEFAULT_SRC, Transform.DEFAULT_DEST)

    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image = mpimg.imread(fname)
        warped, mask = transform.birdseye(image)

        lane = Lane(camera, transform, warped)

        # use data from previous frame
        if FLAGS.debug == 1:
            lane.advance_next_lane(warped)

        head, tail = ntpath.split(fname)
        name = tail or ntpath.basename(head)

        curve_image = lane.draw_curve()
        if FLAGS.debug == 1:
            show_images(warped, curve_image, 'Curves', fname)
        else:
            print('writing {}/lane_detect_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/lane_detect_{}'.format(FLAGS.outputdir, name), curve_image)

        lane_image = lane.draw_lane( image )
        if FLAGS.debug == 1:
            show_images(image, lane_image)
        else :
            print('writing {}/color_lane_{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/color_lane_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
