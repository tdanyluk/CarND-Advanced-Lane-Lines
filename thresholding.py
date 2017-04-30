import numpy as np
import cv2

import config

def boolean_to_binary(boolean_array):
    binary = np.zeros_like(boolean_array, float)
    binary[boolean_array] = 1
    return binary

def threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    yellow = (h_channel >= 20) & (h_channel <= 40) & (s_channel >= 30)
    white = (l_channel >= 200)

    white_binary = boolean_to_binary(white)
    white_or_yellow_binary = boolean_to_binary(white | yellow)

    if config.VISUALIZE_THRESHOLD:
        color = np.dstack((white_binary, white_or_yellow_binary, white_or_yellow_binary))
        cv2.imshow('debug', color)
        if cv2.waitKey() % 256 == 27:
            exit()

    return white_or_yellow_binary
