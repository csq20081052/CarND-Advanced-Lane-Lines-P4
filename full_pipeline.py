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
    
    points_of_interest = np.float32([(736, 466), # 1st point 
                                 (837, 527), # 2nd point
                                 (1039, 651), # 3rd point
                                 (336, 651), # 4th point
                                 (496, 527), # 5th point
                                 (586, 466)]) # 6th point

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


def perspective(image, M):
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(image, M, (w, h))
    
    return warped


def detect_lanes(image, base=None):
    
    if base:
        leftx_base = base[0]
        rightx_base = base[1]
        
    else:     
        # Let's find a starting point
        histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(image.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
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
    
    l_windows = list()
    r_windows = list()

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1)*window_height
        win_y_high = image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Store this window for visualization purposes
        l_windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
        r_windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) \
                          & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) \
                           & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
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
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    windows = [l_windows, r_windows]
    pixels = [left_lane_inds, right_lane_inds]
    poly = [left_fit, right_fit]
    
    return windows, pixels, poly


def visualize_detected_lines(image, windows, pixels, poly):
    
    l_windows = windows[0]
    r_windows = windows[1]
    
    left_lane_inds = pixels[0]
    right_lane_inds = pixels[1]
    
    left_fit = poly[0]
    right_fit = poly[1]
    
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image))*255
    
    # Draw windows
    for l_window, r_window in zip(l_windows, r_windows):
        cv2.rectangle(out_img,l_window[0], l_window[1],(0,255,0), 2) 
        cv2.rectangle(out_img,r_window[0], r_window[1],(0,255,0), 2) 
    
    # Draw in-windows pixels' lanes
    nonzero = image.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = blue
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = red
    
    """
    for l_window, r_window in zip(l_windows, r_windows):
        out_image[l_window][ out_image[l_window] == 0] = blue
        out_image[r_window][ out_image[r_window] == 0] = red
    """
    
    
    (win_xleft_low,win_y_low),(win_xleft_high,win_y_high)
    
    # Dray fitted curves
    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    plt.figure(figsize=(10, 8))
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow', linewidth=6)
    plt.plot(right_fitx, ploty, color='yellow', linewidth=6)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    
def get_curvature_in_meters(image, pixels, xm_per_pix, ym_per_pix):
    
    left_lane_inds = pixels[0]
    right_lane_inds = pixels[1]
    
    nonzero = image.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    y_eval = image.shape[0]
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2*right_fit_cr[0])

    poly = [left_fit_cr, right_fit_cr]
    curvature = [left_curverad, right_curverad]
    
    return curvature, poly


def get_deviation_from_center(image_shape, poly_meters, xm_per_pix):
    
    left_fit = poly_meters[0]
    right_fit = poly_meters[1]
    
    # We will measure the center of the car in the closest point to the car
    y_eval = image_shape[0]
    
    left_lane_center = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_lane_center = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]


    lane_center = left_lane_center + (right_lane_center - left_lane_center)/2
    image_center = image_shape[1]/2 * xm_per_pix
    offset_in_metters = image_center - lane_center # Negative if we are in the left and positive if we are in the right
    # TODO check that this is happening
    
    return offset_in_metters


def draw_lane_unwarped(original_image, warped, poly, pixels, inv_M, plot=False):
    
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    left_fit = poly[0]
    right_fit = poly[1]
    
    left_lane_inds = pixels[0]
    right_lane_inds = pixels[1]
    
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Draw lane lines
    nonzero = warped.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    color_warp_lines = np.zeros_like(color_warp)
    color_warp_lines[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = blue
    color_warp_lines[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = red
    
    
    # Warp the blank back to original image space using inverse perspective matrix
    newwarp_lane = cv2.warpPerspective(color_warp, inv_M, (original_image.shape[1], original_image.shape[0])) 
    newwarp_lines = cv2.warpPerspective(color_warp_lines, inv_M, (original_image.shape[1], original_image.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp_lane, 0.3, 0)
    result = cv2.addWeighted(result, 0.95, newwarp_lines, 1, 0)
    
    if plot:
        plt.figure(figsize=(10,5))
        plt.imshow(result)
    
    return result


def draw_stats(unwarped, curvature, deviation, plot=False):
    
    final_im = unwarped.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(final_im, 'Curvature: Left = %.2lf m, Right = %.2lf m' % (curvature[0], curvature[1]), \
                (70, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(final_im, 'Lane deviation: %.2lf m' % deviation, \
                (70, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if plot:
        plt.figure(figsize=(10,5))
        plt.imshow(final_im)
    
    return final_im
    
    
def load_parameters():
    
    path2calib_im = "./camera_cal/"
    
    camera_calibration = pickle.load(open(path2calib_im + "calibration_data.pkl", "rb" ))
    perspective_matrix = pickle.load(open("./perspective_mat.pkl", "rb"))
    meters_per_pix = pickle.load(open("./meters_per_pix.pkl", "rb"))
    
    return camera_calibration, perspective_matrix, meters_per_pix


def pipeline(image, camera_calibration, perspective_matrix, meters_per_pix):
    
    xm_per_pix, ym_per_pix = meters_per_pix
    M = perspective_matrix['M']
    inv_M = perspective_matrix['inv_M']
    
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # Already done in Camera calibration.ipynb
    
    # Apply a distortion correction to raw images.
    undistorted = undistort(image, camera_calibration)
    
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    thresholded, _, _ = threshold(undistorted)
    
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    birds_eye = perspective(thresholded, M)
    
    # Detect lane pixels and fit to find the lane boundary.
    windows, pixels, polynomials = detect_lanes(birds_eye)
    
    # Determine the curvature of the lane and vehicle position with respect to center.
    curvature, polynomials_in_meters = get_curvature_in_meters(birds_eye, pixels, xm_per_pix, ym_per_pix)
    deviation = get_deviation_from_center(birds_eye.shape, polynomials_in_meters, xm_per_pix)
    
    # TODO it lacks the definition of parameters xm_per_pix, ym_per_pix
    
    # Warp the detected lane boundaries back onto the original image.
    image_lane = draw_lane_unwarped(image, birds_eye, polynomials, pixels, inv_M)
    
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    final_image = draw_stats(image_lane, curvature, deviation)

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