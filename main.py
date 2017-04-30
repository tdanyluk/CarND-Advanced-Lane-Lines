import glob

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip

# My modules
from camera import Camera
from lane import Lane
import thresholding
import lane_finding
import lane_drawing


def plot_side_by_side(title1, img1, title2, img2, targetFileName=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.tight_layout()
    if len(img1.shape) == 2:
        ax1.imshow(img1, cmap='gray')
    else:
        ax1.imshow(img1)
    ax1.set_title(title1, fontsize=24)
    if len(img2.shape) == 2:
        ax2.imshow(img2, cmap='gray')
    else:
        ax2.imshow(img2)
    ax2.set_title(title2, fontsize=24)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.0)
    if targetFileName is None:
        plt.show()
    else:
        plt.savefig(targetFileName)


def calibration_demo(camera, src, dst):
    img = mpimg.imread(src)
    undist_img = camera.undistort(img)
    plot_side_by_side("Original", img, "Undistorted", undist_img, dst)


def create_demo_images(camera):
    img = mpimg.imread('test_images/test1.jpg')
    calibration_demo(camera, 'camera_cal/calibration1.jpg',
                     "output_images/undistort.png")
    calibration_demo(camera, 'test_images/test1.jpg',
                     "output_images/undistort2.png")

    img = mpimg.imread('test_images/test1.jpg')
    img = process_image(img, camera)
    mpimg.imsave("output_images/unwarped.png", img)

    img = mpimg.imread('test_images/test3.jpg')
    undistorted = camera.undistort(img)
    thresholded = thresholding.threshold(undistorted)
    plot_side_by_side("Undistorted", undistorted, "Thresholded",
                      thresholded, "output_images/thresholded.png")
    warped = camera.warp(thresholded)
    lane_finding.find_lane_polynomials(warped, "output_images/lane_finding.png")


    img = mpimg.imread('test_images/straight_lines1.jpg')
    undistorted = camera.undistort(img)
    thresholded = thresholding.threshold(undistorted)
    warped = camera.warp(undistorted)
    cv2.polylines(undistorted, [np.int32(
        camera.warp_source_points)], True, (0, 0, 255), thickness=4)
    cv2.polylines(warped, [np.int32(camera.warp_target_points)],
                  True, (0, 0, 255), thickness=4)
    plot_side_by_side("Undistorted", undistorted, "Warped",
                      warped, "output_images/warped.png")

def process_image(img, camera, lane_object=None):
    if lane_object is None:
        lane_object = Lane(img.shape)

    img = camera.undistort(img)
    undist = np.copy(img)

    img = thresholding.threshold(img)

    img = camera.warp(img)

    try:
        (left_poly, right_poly, left_fitx, right_fitx,
         ploty) = lane_finding.find_lane_polynomials(img)
        lane_object.update(left_fitx, right_fitx, ploty, left_poly, right_poly)
    except Exception as e:
        print('Exception:', e)
        lane_object.update_insane()

    return lane_drawing.draw_all(undist, lane_object, camera)


def get_out_file_name(file_name):
    file_name_parts = file_name.split('.')
    file_name_parts[0] = file_name_parts[0] + '_out'
    return '.'.join(file_name_parts)


def process_video_file(file_name, camera):
    lane_object = Lane(camera.image_shape)

    src_clip = VideoFileClip(file_name)
    dst_clip = src_clip.fl_image(
        lambda frame: process_image(frame, camera, lane_object))
    dst_clip.write_videofile(get_out_file_name(file_name), audio=False)


def process_test_images(camera):
    test_images = glob.glob('test_images/*.jpg')
    for file_name in test_images:
        img = process_image(mpimg.imread(file_name), camera)
        cv2.imshow("debug", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey() % 256 == 27:
            exit()


CAMERA = Camera([720, 1280, 3], 'camera_cal/calibration*.jpg',
                9, 6, 'camera_calibration.p')

create_demo_images(CAMERA)
#process_test_images(CAMERA)
process_video_file('project_video.mp4', CAMERA)
process_video_file('challenge_video.mp4', CAMERA)
