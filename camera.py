import os
import glob
import pickle

import cv2
import numpy as np


def generate_grid(nx, ny):
    grid = np.zeros((ny * nx, 3), np.float32)
    for y in range(0, ny):
        for x in range(0, nx):
            grid[y * nx + x] = [x, y, 0]
    return grid


def calculate_object_and_image_points(chessboard_images_filename_pattern, nx, ny):
    objp = generate_grid(nx, ny)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(chessboard_images_filename_pattern)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        print(
            "Finding chessboard patterns {0}/{1}".format(idx + 1, len(images)), end='\r')
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print()
            print('Failed: ' + fname)

    print()
    return objpoints, imgpoints

def calculate_camera_matrix_and_distortion(objpoints, imgpoints, target_image_shape):
    target_image_size = (target_image_shape[1], target_image_shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, target_image_size, None, None)
    return mtx, dist


def get_perspective_transform(img_shape):
    img_size = (img_shape[1], img_shape[0])

    src = np.float32([[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
                      [((img_size[0] / 6) - 10), img_size[1]],
                      [(img_size[0] * 5 / 6) + 60, img_size[1]],
                      [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])

    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    return cv2.getPerspectiveTransform(src, dst), src, dst

def warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)


class Camera:
    def __init__(self, image_shape, calibration_images_pattern, nx_chessboard, ny_chessboard, camera_pickle_file_name):
        if not os.path.isfile(camera_pickle_file_name):
            objpoints, imgpoints = calculate_object_and_image_points(
                calibration_images_pattern, nx_chessboard, ny_chessboard)
            matrix, distortion = calculate_camera_matrix_and_distortion(
                objpoints, imgpoints, image_shape)
            pickle.dump((matrix, distortion), open(
                camera_pickle_file_name, "wb"))
        else:
            print('Using pickled camera parameters.')
            (matrix, distortion) = pickle.load(
                open(camera_pickle_file_name, "rb"))

        perspective_matrix, src, dest = get_perspective_transform(
            image_shape)

        self.image_shape = image_shape
        self.matrix = matrix
        self.distortion = distortion
        self.warp_matrix = perspective_matrix
        self.unwarp_matrix = cv2.invert(perspective_matrix)[1]
        self.warp_source_points = src
        self.warp_target_points = dest

    def undistort(self, img):
        return cv2.undistort(img, self.matrix, self.distortion, None, self.matrix)

    def warp(self, img):
        return cv2.warpPerspective(img, self.warp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.unwarp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)