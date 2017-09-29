import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image = mpimg.imread('signs_vehicles_xygrad.png')

def show_images(orig_image, alt_image, alt_text='Alt', orig_text='Original Image'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(orig_image)
    ax1.set_title(orig_text, fontsize=30)
    ax2.imshow(alt_image, cmap='gray')
    ax2.set_title(alt_text, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


if __name__ == '__main__':
    # Choose a Sobel kernel size
    ksize = 9 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = gradient_mask(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    show_images(image, gradx, 'Sobel X')
    grady = gradient_mask(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    show_images(image, grady, 'Sobel Y')
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    show_images(image, mag_binary, 'Magnitude of Gradient')
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    show_images(image, dir_binary, 'Direction of Gradient')
    hls_binary = hls_mask( image, thresh=(90,255), channel=2 )
    show_images(image, hls_binary, 'Saturation (HLS)')
    combined_color, combined_binary = combined_sgray(image)
    show_images(combined_color, combined_binary, 'Combined Gradients + Sat', 'Green Gradients; Blue Sat')
