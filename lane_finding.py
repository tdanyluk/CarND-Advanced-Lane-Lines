import cv2
import matplotlib.pyplot as plt
import numpy as np

import config

def find_lane_polynomials(binary_warped, output_image_file_name=None):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Avoid lines on the side
    histogram[0:100] = 0
    histogram[binary_warped.shape[1] - 100:binary_warped.shape[1]] = 0

    if config.VISUALIZE_LANE_FINDING_HISTOGRAM:
        plt.plot(histogram)
        plt.show()

    # Create an output image to draw on and visualize the result
    if config.VISUALIZE_LANE_FINDING or output_image_file_name is not None:
        out_img = np.uint8(
            np.dstack((binary_warped, binary_warped, binary_warped)) * 255)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if config.VISUALIZE_LANE_FINDING or output_image_file_name is not None:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean
        # position
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

    # Fit a second order polynomial to each
    if leftx.shape[0] == 0 or lefty.shape[0] == 0 or rightx.shape[0] == 0 or righty.shape[0] == 0:
        raise Exception('No good data')
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, 30)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    if config.VISUALIZE_LANE_FINDING or output_image_file_name is not None:
        out_img[nonzeroy[left_lane_inds],
                nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds],
                nonzerox[right_lane_inds]] = [0, 0, 255]

    left_points = np.stack([left_fitx, ploty]).T
    right_points = np.stack([right_fitx, ploty]).T

    if config.VISUALIZE_LANE_FINDING or output_image_file_name is not None:
        cv2.polylines(out_img, [np.int32(left_points)], False, (0, 255, 0), 4)
        cv2.polylines(out_img, [np.int32(right_points)], False, (0, 255, 0), 4)
        if config.VISUALIZE_LANE_FINDING:
            cv2.imshow("debug", out_img)
            if cv2.waitKey() % 256 == 27:
                exit()
        if output_image_file_name is not None:
            cv2.imwrite(output_image_file_name, out_img)

    return (left_fit, right_fit, left_fitx, right_fitx, ploty)
