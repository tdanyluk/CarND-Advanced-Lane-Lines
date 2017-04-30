import numpy as np
import cv2

# my own modules
import config


def boolean_to_binary(boolean_array):
    binary = np.zeros_like(boolean_array, float)
    binary[boolean_array] = 1
    return binary

def simple_thresh(img, thresh):
    return boolean_to_binary((img >= thresh[0]) & (img <= thresh[1]))

def abs_sobel_thresh(der, thresh):
    der = np.absolute(der)
    der = np.uint8(der * 255 / np.max(der))
    return simple_thresh(der, thresh)

def mag_thresh(dx, dy, thresh):
    magnitude = np.sqrt(dx**2 + dy**2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))
    return simple_thresh(magnitude, thresh)

def dir_thresh(dx, dy, thresh):
    dir_grad = np.arctan2(dy, dx)
    return simple_thresh(dir_grad, thresh)

def combined_gradient_threshold(img, min_dx, min_dy, min_magnitude):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    dx_binary = abs_sobel_thresh(dx, (min_dx, 255))
    dy_binary = abs_sobel_thresh(dy, (min_dy, 255))

    mag_binary = mag_thresh(dx, dy, (min_magnitude, 255))
    dir_binary = dir_thresh(dx, dy, (0.7, 1.3))

    if config.VISUALIZE_COMBINED_GRADIENT_THRESHOLD:
        gradxy = boolean_to_binary((dx_binary == 1) & (dy_binary == 1))
        magdir = boolean_to_binary((mag_binary == 1) & (dir_binary == 1))
        color = np.dstack((np.zeros_like(gradxy), gradxy, magdir))
        cv2.imshow('debug', color)
        if cv2.waitKey() % 256 == 27:
            exit()

    return boolean_to_binary(
        ((dx_binary == 1) & (dy_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))
    )

def threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    l_grad = combined_gradient_threshold(l_channel, 30, 30, 40)
    ls_combo = boolean_to_binary((l_channel >= 100) & (s_channel >= 140))

    if config.VISUALIZE_THRESHOLD:
        color = np.dstack((np.zeros_like(l_grad), l_grad, ls_combo))
        cv2.imshow('debug', color)
        if cv2.waitKey() % 256 == 27:
            exit()

    return boolean_to_binary((l_grad == 1) | (ls_combo == 1))

def threshold2(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    l_channel = l_channel * (255.0 / np.percentile(l_channel[int(l_channel.shape[0]*0.6):, :], 99.9))
    yellow = (h_channel >= 20) & (h_channel <= 40) & (s_channel >= 20)
    white = (l_channel >= 230)

    white_binary = boolean_to_binary(white)
    white_or_yellow_binary = boolean_to_binary(white | yellow)

    if config.VISUALIZE_THRESHOLD:
        color = np.dstack((white_binary, white_or_yellow_binary, white_or_yellow_binary))
        cv2.imshow('debug', color)
        if cv2.waitKey() % 256 == 27:
            exit()

    return white_or_yellow_binary
