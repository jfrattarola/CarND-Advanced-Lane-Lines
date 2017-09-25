import argparse
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from all_funcs import show_images

def camera_cal_init(cal_images_path):
    object_points=[]
    image_points=[]

    #create object points for each intersection on the chessboard
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

    #import all images
    images = glob.glob(os.path.join(cal_images_path, '*.jpg'))

    #iterate over images to get corner and image points
    for fname in images:
        #read in and convert to grayscale
        image = mpimg.imread(fname)
        
        #Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        #find the corners
        ret, corners = cv2.findChessboardCorners( gray, (8,6), None)

        #if corners are found, add them to the image and object points array
        if ret == True:
            image_points.append(corners)
            object_points.append(objp)

    return object_points, image_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='camera_cal', 
                        help='directory to read calibration image files from')
    FLAGS, unparsed = parser.parse_known_args()

    object_points, image_points = camera_cal_init(FLAGS.dir)

    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image = mpimg.imread(fname)
        image_size = (image.shape[1], image.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None ,None)
        undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

        show_images(image, undistorted_image, alt_text='Undistorted Image')
        
