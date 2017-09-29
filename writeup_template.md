**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[dist_img]: ./examples/distort_output.jpg "Distorted"
[undist_img]: ./examples/undistort_output.jpg "Undistorted"
[dist_cars]: ./examples/distorted_cars.jpg "Distorted Cars"
[undist_cars]: ./examples/undistorted_cars.jpg "Undistorted Cars"
[thresh_channels]: ./examples/thresh_channels.jpg "Threshold Channels Binary"
[thresh_mask]: ./examples/thresh_mask.jpg "Threshold Mask Binary"
[straight]: ./examples/straight_lines1.jpg "Straight undistorted with src lines"
[warped]: ./examples/warped_straight_lines1.jpg "Warped undistorted with dest lines"
[thresh_warped]: ./examples/perspective_thresh_warped.jpg "Masked image with perspective transform"
[lanes]: ./examples/lanes.png "Colored/Curved lane lines"
[color_lane]: ./examples/color_lane.jpg "Colored Curved Lane"
[video1]: ./project_video.mp4 "Video"
[final_video]: ./final_video.m4v "Final Project Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

Aside from images displayed, I ran tests on all images in `test_images/`, placing the output in `output_images/`

### Camera Calibration

The code for this step is contained in the file `camera_cal.py`

This file produces the class object `Camera`. The init function takes in the path for the calibration images. I used 20 different perspectives of a chess board.  The Camera object returned contains the object points (x,y,z=0 coordinates of the chessboard corners) and image points (x,y pixel position of each of the corners in the image plane with each successful chessboard detection). These are used to calibrate the camera once per image_size change, producing distortion coefficients.

This class also has an `undistort()` function which will use the distortion coefficients and a provided image to obtain an undistorted version of the image, as seen here:

#### Distorted Original Calibration Image

![Distorted Original][dist_img] 

#### Undistorted Output

![Undistorted Image][undist_img]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

#### Distorted Cars Original

![Distorted Cars][dist_cars]

#### Undistorted Cars

![Undistorted Cars][undist_cars]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of gradient and color thresholds to generate a binary image. The function I used in the end is found in `threshold.py`, called `lane_mask()`.  Instead of grayscaling the image for the Sobel operator, I used the L channel from HSL along the x axis.

#### Colorized Threshold Binary

![threshold binary][thresh_channels]

#### Threshold Mask Binary

![threshold_mask][thresh_mask]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in `perspective.py` in the class `Transform`, which takes in a Camera object, src points and dest points. The warped/transformed image is created by the function birdseye() in Transform object.  I chose to hardcode the source and destination points manually:

```python
SRC=np.float32([[685, 450], [1075, 705], [230, 705], [600, 450]])                                                                 
DEST=np.float32([[960, 0], [960, 720], [320, 720], [320, 0]])    
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![straight][straight]

![warped][warped]

Here, I apply the method to a threshold/masked binary image

![thresh_mask][thresh_mask]

![thresh_warped][thresh_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I found the pixels in each lane from the birdseye view.  The code is found in `lanes.py` class Lane, init function and advance_next_lane, which is the optimized non-full-search function given in the lesson.

I drew curved lines and colored the lane lines from the mask using numpy's polyfit method. Example:

![lanes][lanes]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is done in `lanes.py` in the Lane class function `get_curves_and_offset()`, which uses the formula given in the lesson (based off the formula from the Lane() init method): `(1 + (2 * Ay + B)^2)^(3/2)]/|2*A|`

The offset of the car from the road's center assumes the camera is mounted in the middle of the car. We find the distance of the two lanes via: 
```python
camera_position = self.img.shape[1] / 2.                                                                                  
lane_center = (right_fitx[-1] + left_fitx[-1]) / 2.                                                                       
self.center_offset_pixels = camera_position - lane_center                                                                 
self.center_offset_meters = xm_per_pix * center_offset_pixels 
```
        
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Using the `draw_lane()` method of the `Lane` class, I draw the lane area back onto the lane:

![color_lane][color_lane]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Using this [source video](./project_video.mp4) I ran each frame through my pipeline and converted the frames back into an m4v video: [Here](./final_video.m4v)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project was by far the hardest.  It took me so much time to complete. I learned that a **lot** of the success is dependant on getting the earlier threshold binaries correct.  I had a weird bug where it kept crashing halfway through because the right lane just went all over the place. I had been using the L channel for sobel gradient instead of grayscale (people were saying they had good success with this). I went over and over with debugging statements trying to find the problem. Eventually, I switched back to grayscale for my sobel x gradient and everything was fixed.  Insane!
