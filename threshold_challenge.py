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

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    #set x/y params based on orient arg
    x = 1 if orient is 'x' else 0
    y = 1 if x == 0 else 0

    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #take derivitive in x or y, given orient
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y)

    #take the absolute value of the derivative
    sobel_abs = np.absolute(sobel)

    #scale to 8-bit (0-255) and convert to uint8
    scaled_sobel = np.uint8(255 * sobel_abs / np.max(sobel_abs))

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

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]

    #convert to 8 bit (0-255)
    scaled_S = np.uint8( 255 * S / np.max(S) )

    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(scaled_S)
    binary_output[(scaled_S > thresh[0]) & (scaled_S <= thresh[1])] = 1

    return binary_output

def combined_sgray(img, grad_thresh=(20,100), s_thresh=(170,255)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    scaled_s = np.uint8( 255 * s_channel / np.max(s_channel) )

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(scaled_s)
    s_binary[(scaled_s > s_thresh[0]) & (scaled_s <= s_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary

# Choose a Sobel kernel size
ksize = 9 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
show_images(image, gradx, 'Sobel X')
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
show_images(image, grady, 'Sobel Y')
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
show_images(image, mag_binary, 'Magnitude of Gradient')
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
show_images(image, dir_binary, 'Direction of Gradient')
hls_binary = hls_select( image, thresh=(90,255) )
show_images(image, hls_binary, 'Saturation (HLS)')
combined_color, combined_binary = combined_sgray(image)
show_images(combined_color, combined_binary, 'Combined Gradients + Sat', 'Green Gradients; Blue Sat')
