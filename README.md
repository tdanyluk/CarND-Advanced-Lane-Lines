## Writeup

---

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

[undistort]: ./output_images/undistort.png "Chessboard Undistorted"
[undistort2]: ./output_images/undistort2.png "Road Undistorted"
[threshold]: ./output_images/thresholded.png "Thresholded image"
[warped]: ./output_images/warped.png "Warp Example"
[lane_finding]: ./output_images/lane_finding.png "Lane finding"
[unwarped]: ./output_images/unwarped.png "Unwarped image"
[video1]: ./project_video_out.mp4 "Project Video"
[video2]: ./challenge_video_out.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `camera.py` starting from line 48.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Image undistortion][undistort]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Distortion correction is implemented in `camera.py` starting from line 48 and 101.
OpenCV can simply undistort the image, given the camera matrix and the distortion coefficients.

![Image undistortion][undistort2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a threshold based on "color", in HLS color space. (`thresholding.py`).  Here's an example of my output for this step.
I tried more sophisticated methods also, but this one proved good for the "project" and the "challenge" video.

![Thresholded image][threshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 104 through 105 in the file `camera.py`.  The `warp()` function takes as inputs an image (`img`), and uses the perspective transformation matrix. The transformation matrix is calculated by get_perspective_transform (`camera.py` line 55) based on the image size and the following hardcoded source and destination points:

```
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
print(src)

dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped image][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

My lane finding code is in `lane_finding.py`.

I used a histogram to find the approximate x position of lane lines at the bottom of the image.
I erased the leftmost and rightmost 100 pixel part of the histogram, to avoid finding the side of the road instead of lane lines.
I used the simple windowed search mentioned in the lessons to identify which are the lane pixels.
I fit my lane lines with a 2nd order polynomial for each lane.
f(y) = Ay^2 + By + C

![Lane finding][lane_finding]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did the radius of curvature calculation in lines 12 through 24 in `lane.py`.
First I transformed the points of the polynomial segments to world size, then I fit a new polynomial over them.
I did the radius calculation at the lowest points of the lanes (that point is the closest to the car).
I display the avarage of the left and right curvatures. (And I only change the displayed text about 3 times / second to make it more readable.)

The position of vehicle position calculation is in line 26 through 27 in `lane.py`. It is done based on the distance of the lane center from the image center at the bottom of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in lines 4 through 17 in my code in `lane_drawer.py` in the function `draw_lane()`.  Here is an example of my result on a test image:

![Unwarped image][unwarped]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

And also to the [challenge video result](./challenge_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First I faced a problem: I did not optimize the thresholding on hard enough examples and I later discovered that my (previous) method was not good enough on some frames of the project video.

It might fail in very bright weather conditions, or when looking out of a tunnel, because the roud could all seem white. More robust thresholding would be needed.

It might fail when there are very sharp bends, because the window based lane line finder might not find the next lane line segment if it is at a very different place than the current. It would have to move the window more horizontally. I think that the convolution based sliding window search would handle it.