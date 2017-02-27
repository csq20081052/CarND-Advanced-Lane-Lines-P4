import os
import cv2
import glob
import pickle
import numpy as np
import pylab as plt


def undistort(image, camera_calibration):
    mtx = camera_calibration['mtx']
    dist = camera_calibration['dist']
    
    return cv2.undistort(image, mtx, dist, None, mtx)


def extract_lanes_by_color(rgb_image, white_lower_th, white_upper_th, yellow_lower_th, yellow_upper_th):
    hls_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    
    white = cv2.inRange(hls_image, white_lower_th, white_upper_th)
    yellow = cv2.inRange(hls_image, yellow_lower_th, yellow_upper_th)
    combined = yellow | white
    
    return combined


def extract_lanes_by_gradient(rgb_image, sobel_mag_th, sobel_ang_th):
    hls_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    
    s_channel = hls_image[...,2]
    
    # Sobel x
    sobel_x = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x) # Absolute x derivative to accentuate lines away from horizontal
    
    # Sobel y
    sobel_y = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    abs_sobel_y = np.absolute(sobel_y) # Absolute x derivative to accentuate lines away from horizontal
    
    mag = np.sqrt(sobel_x*sobel_x + sobel_y*sobel_y)
    mag_sc = (255*mag/np.max(mag)).astype(np.uint8)
    ang = np.arctan2(abs_sobel_x, abs_sobel_y)
    
    # Threshold gradient
    sobel_binary = np.zeros_like(sobel_x, np.uint8)
    sobel_binary[(mag_sc >= sobel_mag_th[0]) & (mag_sc <= sobel_mag_th[1]) & \
                 (ang >= sobel_ang_th[0]) & (ang <= sobel_ang_th[1])] = 1
    
    return sobel_binary


def dilation(image):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(image, kernel, iterations=2)


def combine_color_gradient_postpro(thresholded_color, thresholded_gradient):
    
    thresholded_combined = thresholded_color & dilation(thresholded_gradient)
    
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(thresholded_combined, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    
    return dilated


def threshold(image):
    
    # Final thresholds for white and yellow lane lines:
    white_lower_th = (0, 195, 25)
    white_upper_th = (30, 255, 255)
    yellow_lower_th = (15, 75, 125)
    yellow_upper_th = (35, 200, 255)
    
    # Final thresholds for gradients
    sobel_mag_th = (25, 165)
    sobel_ang_th = (0.31, 1.05)
    
    # Color thresholding
    thresholded_color = extract_lanes_by_color(image, white_lower_th, white_upper_th, \
                                         yellow_lower_th, yellow_upper_th)
    
    # Gradient thresholding
    thresholded_gradient = extract_lanes_by_gradient(image, sobel_mag_th, sobel_ang_th)
    
    # Combining both and post process
    combined = combine_color_gradient_postpro(thresholded_color, thresholded_gradient)

    return combined, thresholded_gradient, thresholded_color


def compute_perspective_mat(image, invert=False):
    h, w = image.shape[:2]
    h_offset = 20 # height offset for dst points
    w_offset = 150 # width offset

    # Source points
    src = points_of_interest[[0, 2, 3, 5],:] # Biggest trapezoid
    #src = points_of_interest[[1, 2, 3, 4],:] # Smallest inferior trapezoid
    #src = points_of_interest[[0, 1, 4, 5],:] # Superior trapezoid

    # Destination points
    dst = np.float32([[w-w_offset, h_offset], 
                     [w-w_offset, h-h_offset], 
                     [w_offset, h-h_offset],
                     [w_offset, h_offset]])
    
    if invert:
        src, dst = dst, src
        
    M = cv2.getPerspectiveTransform(src, dst)
    
    return M


def perspective(image, perspective_matrix):
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(image, perspective_matrix, (w, h))
    
    return warped


def pipeline(image, camera_calibration, perspective_matrix):
    
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # Already done in Camera calibration.ipynb
    
    # Apply a distortion correction to raw images.
    undistorted = undistort(image, camera_calibration)
    
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    thresholded = threshold(undistorted)
    
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    birds_eye = perspective(thresholded, perspective_matrix)
    
    # Detect lane pixels and fit to find the lane boundary.
    detected_lanes, polynomials = detect_lanes(birds_eye)
    
    # Determine the curvature of the lane and vehicle position with respect to center.
    curvature = get_curvature(detected_lanes, polynomials)
    
    # Warp the detected lane boundaries back onto the original image.
    image_lanes = get_original_with_lanes()
    
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    final_image = visualization(image_lanes, curvature, vehicle_position)
    
    # TODO: where to get vehicle_position?
    
    return final_image


def visualize_before_after(image_before, image_after, cmap=None):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1,2,1)
    plt.imshow(image_before)
    plt.title("Image before")
    
    plt.subplot(1,2,2)
    if cmap:
        plt.imshow(image_after, cmap=cmap)
    else:
        plt.imshow(image_after)
    plt.title("Image after")