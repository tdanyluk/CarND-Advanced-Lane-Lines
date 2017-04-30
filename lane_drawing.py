import cv2
import numpy as np

def draw_lane(undist_img, pts, camera):
    # Create an image to draw the lines on
    warp_zero = np.zeros(undist_img.shape[0:2], np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space
    newwarp = camera.unwarp(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
    return result

def draw_all(undist_img, lane, camera):
    if not lane.is_valid():
        return undist_img

    img = draw_lane(undist_img, lane.points, camera)
    curvature = lane.radius_of_curvature_for_display_m
    diff_m = lane.relative_car_position_for_display_m

    color = (255, 255, 255) # if lane.is_up_to_date() else (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Radius: {0:5.0f}m, Position: {1:0.2f}m".format(curvature, diff_m),
                (10, 70), font, 2, color, 2, cv2.LINE_AA)
    return img
