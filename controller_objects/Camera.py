import cv2
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, argrelextrema, find_peaks
import threading
"""
0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
5. CV_CAP_PROP_FPS Frame rate.
6. CV_CAP_PROP_FOURCC 4-character code of codec.
7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by
"""

def get_red(img):
    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    mask1 = mask1+mask2
    return mask1

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

class Camera:
    _height = 1944
    _width = 2592

    def __init__(self, verbose=False):
        self.cam = cv2.VideoCapture(-1)
        self.frame = None
        if not (self.cam.isOpened()):
            print("No camera detected...")
        # Random sleeps to verify that the settings work

        time.sleep(1)
        self.cam.set(3,self._width)
        time.sleep(1)
        self.cam.set(4,self._height)
        time.sleep(1)
        self.cam.set(10, -1)
        time.sleep(1)
        self.cam.set(38,1)
        time.sleep(1)
        #self.cam.set(12,1200)
        time.sleep(1)
        #print(self.cam.get(10))
        #print(self.cam.get(38))
        #print(self.cam.get(14))
        got_first_frame = False
        while not got_first_frame:
            ret, frame = self.cam.read()
            if not (frame is None):
                got_first_frame = True



    def grab_picture(self):
        ret, frame = self.cam.read()
        frame = None
        while frame is None:
            ret, frame = self.cam.read()
        self.frame = frame

    def get_display(self, crop_x=[500,2000], crop_y=[800,1300]):
        img_crop = np.copy(self.frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]])

        output = get_red(np.copy(img_crop))

        #----------------------------#
        #------ Finding gauge -------#
        #----------------------------#

        # Remove noise from image using cv2.MORPH_OPEN
        kernel = np.ones((5,5),np.uint8)
        out = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)

        # Find larges object in image
        cnts, _ = cv2.findContours(out, 1, 2)
        cnts.sort(key=lambda x: cv2.contourArea(x))
        gauge = cnts[-1]

        # Calculate the approximated circle that is the gague
        gauge_area = cv2.contourArea(gauge)
        approx_circle_raduis = int(np.sqrt(gauge_area / np.pi))

        M = cv2.moments(gauge)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])   
        approx_gauge_center = (cX, cY)

        # Apply a shirnikg factor to remove border noise
        shrik_factor = 0.8
        approx_circle_raduis_shrinked = int(approx_circle_raduis * shrik_factor)
        
        # Use the circle to create a mask
        shrinked_gauge_mask = np.zeros(out.shape)
        cv2.circle(shrinked_gauge_mask, approx_gauge_center, approx_circle_raduis_shrinked, (255,0,0), -1)

        # Use mask on originale image to crop out relevant area and have more data to work with
        shrinked_gauge_mask_3d = np.repeat(shrinked_gauge_mask[:, :, np.newaxis], 3, axis=2)
        originale_image_masked = np.multiply(img_crop, shrinked_gauge_mask_3d / 255).astype(np.uint8)
        
        
        # Crop the image to fit only the mask
        circle_coordinates = np.where(shrinked_gauge_mask == 255)
        x_min = np.min(circle_coordinates[0])
        x_max = np.max(circle_coordinates[0])
        y_min = np.min(circle_coordinates[1])
        y_max = np.max(circle_coordinates[1])
        originale_image_masked_cropped = cv2.cvtColor(originale_image_masked[x_min:x_max, y_min:y_max], cv2.COLOR_BGR2GRAY)
        shrinked_gauge_mask_cropped = shrinked_gauge_mask[x_min:x_max, y_min:y_max]
        
        # filter out noise and find the largest object
        kernel = np.ones((3,3),np.uint8)
        out = cv2.morphologyEx(originale_image_masked_cropped, cv2.MORPH_OPEN, kernel)    
        cv2.imwrite(f"debug_images/manipulated_image_cropped_filt.png", out)
        cv2.imwrite(f"debug_images/manipulated_image_cropped.png", originale_image_masked_cropped)
        
        # Normalie to use the entire spectrum
        lower_thres = 20
        out = (out / np.max(out) * 255).astype(np.uint8)
        
        # Itterativly force peaks at start and end
        hist_stop = 255
        bin_size = 50
        step_size = int(hist_stop / bin_size)
        (n, bins, patches) = plt.hist(out.ravel(), bins=50, range=(1, 255))
        plt.savefig(f"debug_images/max_min_thres_original")
        start_max = np.max(n[1:])
        hist_stop = hist_stop + step_size + 1
        n_stop_value = 255
        while n_stop_value <= start_max:
            hist_stop -= step_size
            out = np.where((out > hist_stop) & (out != 0), hist_stop, out)
            plt.clf()
            (n, bins, patches) = plt.hist(out.ravel(), bins=50, range=(1, 255))
            n_stop_value = np.max(n[1:])
            #plt.savefig(f"max_thres_{hist_stop}")

        # Itterete until we have a new maxima in the start og the histogram
        start_max = np.max(n[1:])
        hist_start = - step_size + 1
        n_start_value = 1
        while n_start_value <= start_max:
            hist_start += step_size
            out = np.where((out < hist_start) & (out != 0), hist_start, out)
            plt.clf()
            (n, bins, patches) = plt.hist(out.ravel(), bins=50, range=(1, 255))
            n_start_value = np.max(n[1:])
            #plt.savefig(f"min_thres_{hist_start}")
        #out = np.where(out < lower_thres, lower_thres, out)
        
        #print(circles)

        # Find threshold value thrugh polynomial fitting and local minima

        
        idx_start = np.where(n == n_start_value)[0][0]
        idx_stop = np.where(n == n_stop_value)[0][0]

        #print(idx_start, idx_stop)

        bins_cropped = bins[idx_start:idx_stop+1]
        n_cropped = n[idx_start:idx_stop+1]
        z = np.polyfit(bins_cropped, n_cropped, 2)
        p = np.poly1d(z)
        x = np.linspace(hist_start,hist_stop,1000)
        pol = p(x)
        plt.plot(x,pol)
        plt.savefig("debug_images/asd")
        minimum = argrelextrema(pol, np.less)
        threshold = int(x[minimum])
        #print(threshold)
        plt.savefig("debug_images/hist.png")
        
        #Threshold image, ivert and mask again to extract only mask
        out_inv = np.where(out > threshold, 0, 255)
        out_inv_masked = np.multiply(shrinked_gauge_mask_cropped / 255, out_inv).astype(np.uint8)
        
        # Find larges object in the image
        cnts, _ = cv2.findContours(out_inv_masked, 1, 2)
        cnts.sort(key=lambda x: cv2.contourArea(x))
        display = cnts[-1]

        display_contour = np.zeros(out_inv_masked.shape).astype(np.uint8)
        cv2.drawContours(display_contour, [display], 0, (255,255,255), 1)

        x_values = display[:,0,0]
        y_values = display[:,0,1]
        buffer_val = int((max(y_values) - min(y_values)) * 0.3)
        
        west_crop = np.copy(display_contour)
        west_crop[:,min(x_values) + buffer_val:] = 0
        
        east_crop = np.copy(display_contour)
        east_crop[:,:max(x_values) - buffer_val] = 0

        north_crop = np.copy(display_contour)
        north_crop[min(y_values) + buffer_val:, :] = 0
        
        south_crop = np.copy(display_contour)
        south_crop[:max(y_values) - buffer_val,:] = 0

        #First two is vertical, last two is horizontal
        crops = [west_crop, east_crop, north_crop, south_crop]

        lines = []
        print("Finding lines")
        for crop in crops:
            thresh = 100
            line = None
            while True:
                line = cv2.HoughLines(crop, 1, np.pi / 180, thresh)
                if not line is None:
                    lines.append(line[0][0])
                    #print(line)
                    break
                thresh -= 1

        points = []
        for line in lines[:2]:
            points.append(intersection(line, lines[2]))
            points.append(intersection(line, lines[3]))



        orig_img = img_crop[x_min:x_max, y_min:y_max]
        #lines = cv2.HoughLines(display_contour, 1, np.pi / 180, 80)
        print(len(lines))
        # Draw the lines
        if lines is not None:
            for line in lines:
                rho = line[0]
                theta = line[1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(orig_img, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)

        # Aprroximate corners of shape found
        epsilon = 0.1*cv2.arcLength(display,True)
        approx = cv2.approxPolyDP(display,epsilon,True)

        for point in points:
            #print(point)
            cv2.circle(orig_img, (point[0], point[1]), 2, (255,255,255), 1)
        cv2.imwrite(f"debug_images/manipulated_image_display_contour.png", display_contour)
        cv2.imwrite(f"debug_images/manipulated_image_thres.png", out_inv_masked)
        cv2.imwrite(f"debug_images/manipulated_image_originale.png", img_crop)
        cv2.imwrite(f"debug_images/manipulated_image_output.png", orig_img)

        # Measured values
        w = 43
        h = 19

        dest_w = 150
        dest_h = int((h/w) * dest_w)

        dst = np.array([
            [0,0],
            [0,dest_h],
            [dest_w,0],
            [dest_w,dest_h]
        ], dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        
        print(points)
        print(dst)
        print(type(points[0,0]))
        print(type(dst[0,0]))

        M = cv2.getPerspectiveTransform(points, dst)
        warped = cv2.warpPerspective(orig_img, M, (dest_w, dest_h))    

        return warped

    def save_frame_to_file(self, filename="img.jpg"):
        if self.frame is not None:
            cv2.imwrite(filename, self.frame)
        else:
            print("No frame to save...")

    def load_frame_from_file(self, filename="img.jpg"):
        self.frame = cv2.imread(filename)

    def __del__(self):
        self.cam.release()
