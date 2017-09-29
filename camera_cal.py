import argparse
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import ntpath
from utils import show_images

class Camera:    
    def __init__(self, cal_images_path):
        self.cal_path=cal_images_path
        self.objp=[]
        self.imgp=[]
        self._image_size=None

        #create object points for each intersection on the chessboard
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        #import all images
        images = glob.glob(os.path.join(cal_images_path, '*.jpg'))

        #iterate over images to get corner and image points
        for fname in images:
            #read in and convert to grayscale
            image = mpimg.imread(fname)
        
            #Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            #find the corners
            ret, corners = cv2.findChessboardCorners( gray, (9,6), None)

            #if corners are found, add them to the image and object points array
            if ret == True:
                self.imgp.append(corners)
                self.objp.append(objp)
            else:
                print('ERROR reading {}'.format(fname))

    def image_size(self):
        return (0,0) if self._image_size is None else self._image_size

    def calibrate(self):
        ret = 0
        if self._image_size is not None:
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objp, self.imgp, self._image_size, None ,None)
            ret = 1
        return ret

    def undistort(self, image):
        image_size = (image.shape[1], image.shape[0])
        ret = 0 if self._image_size != image_size else 1

        if ret == 0:
            self._image_size = image_size
            ret = self.calibrate()
        undistort = None
        if ret == 1:
            undistort = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return undistort

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='camera_cal', 
                        help='directory to read calibration image files from')
    parser.add_argument('--testdir', type=str, default='test_images', 
                        help='directory to read calibration image files from')
    parser.add_argument('--debug', type=int, default=0, 
                        help='print images to screen')
    parser.add_argument('--outputdir', type=str, default='output_images', 
                        help='directory to write images to')
    FLAGS, unparsed = parser.parse_known_args()

    camera = Camera(FLAGS.dir)

    #print calibration image differences
    images = glob.glob(os.path.join(FLAGS.dir, '*.jpg'))
    for fname in images:
        image = mpimg.imread(fname)
        undistorted_image = camera.undistort(image)
        if FLAGS.debug == 1:
            show_images(image, undistorted_image, 'Undistorted Image', 'Distorted Calibration Image')
        else:
            head, tail = ntpath.split(fname)
            name = tail or ntpath.basename(head)
            print('writing {}/{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/{}'.format(FLAGS.outputdir, name), image)
            cv2.imwrite('{}/undistorted_{}'.format(FLAGS.outputdir, name), undistorted_image)
    #print test image differences
    images = glob.glob(os.path.join(FLAGS.testdir, '*.jpg'))
    for fname in images:
        image = mpimg.imread(fname)
        undistorted_image = camera.undistort(image)
        if FLAGS.debug == 1:
            show_images(image, undistorted_image, 'Undistorted Image', 'Distorted Test Image')
        else:
            head, tail = ntpath.split(fname)
            name = tail or ntpath.basename(head)
            print('writing {}/{}'.format(FLAGS.outputdir, name))
            cv2.imwrite('{}/{}'.format(FLAGS.outputdir, name), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.imwrite('{}/undistorted_{}'.format(FLAGS.outputdir, name), cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))

