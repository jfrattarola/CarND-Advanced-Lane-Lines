from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import cv2
import glob
import argparse
import numpy as np
import sys
from utils import show_images
from camera_cal import Camera
from perspective import Transform
from lanes import Lane

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

def draw(warped, left_fit, right_fit, Minv, undist):
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

def add_text(img, left_curverad, right_curverad, center_offset_meters):
    # Add info about radius and offset
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'left_curverad {:5.2f} meters - right_curverad: {:5.2f}'.format(left_curverad, right_curverad)
    if center_offset_meters < 0:
        text2 = 'offset: {:2.3f} meters to the left'.format(abs(center_offset_meters))
    else:
        text2 = 'offset: {:2.3f} meters to the right'.format(abs(center_offset_meters))

    cv2.putText(img, text1, (50, 30), font, 1, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, text2, (50, 70), font, 1, (0,0,255), 1, cv2.LINE_AA)

    return result

if __name__ == '__main__':
    """
    PARAMETERS, UTILS AND PATHS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--caldir', type=str, default='camera_cal', 
                        help='directory to read calibration image files from')
    parser.add_argument('--dir', type=str, default='test_images', 
                        help='directory to read test image files from')
    parser.add_argument('--outputdir', type=str, default='frames', 
                        help='directory to write images to')
    parser.add_argument('--debug', type=int, default=0, 
                        help='print images to screen')
    FLAGS, unparsed = parser.parse_known_args()

    clip = VideoFileClip("project_video.mp4")
    frames = int(clip.fps * clip.duration)
    image_folder = FLAGS.outputdir
    video_file = 'processed_video.mp4'

    # pixel to meters conversion
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    pxl_to_meters_radius_ratio = 3.05

    camera = Camera(FLAGS.caldir)
    transform = Transform(camera, Transform.DEFAULT_SRC, Transform.DEFAULT_DEST)

    """
    Loop over all frames:
        - Convert image to birdseye view (includes undistort, threshold mask binary)
        - Detect lanes and smooth between frames (Fit a second order polynomial to each lane,  Calculate curves and vehicle offset)
        - Draw lanes and add text
        - Save frame

    """
    lane = None
    print('Processing video...')
    for idx, frame in enumerate(clip.iter_frames()):
        progress(idx+1, frames)
        warped, mask = transform.birdseye(frame)

        # Detect lanes and smooth between frames
        if lane is None:
            lane = Lane(camera, transform, warped)
        else:
            lane.advance_next_lane(warped)


        # Draw_lanes
        lane_image = lane.draw_lane(frame)

        # save frame
        if FLAGS.debug == 1:
            show_images(frame, lane_image)
        else :
            cv2.imwrite('{}/frame_{:010d}.jpg'.format(image_folder, idx), cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
    print('')
