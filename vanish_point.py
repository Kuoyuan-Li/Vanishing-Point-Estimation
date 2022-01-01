"""
**Author:** Kuoyuan Li
"""

# Import needed libraries
import matplotlib.pyplot as plt
import cv2  
import os
import numpy as np
import pandas as pd
import random
import math
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

# Helper functions

# Show images given a list of images
def show_images(image):
    plt.figure()
    plt.imshow(image,cmap='gray')

# Load images from a folder given their filenames
def load_images(filename):
    try:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        return img
    except IOError:
        print("File is not an image\n")
        exit()

# Plot lines on original images 
def show_lines(image,lines):
    # Implementation is based on workshop material
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1 = (int(x0 + 1000*(-b)),int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)),int(y0 - 1000*(a)))
        # Draws a line segment connecting two points, colour=(255,0,0) and thickness=2.
        cv2.line(image,pt1,pt2,(255,0,0),1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
        
# Plot lines and points on original images 
def show_point(image, point):
    # Implementation is based on workshop material
    cv2.circle(image,point,3,(0,255,0), thickness=3)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

## 1. Detect lines in the image
## Use the Canny edge detector and Hough transform to detect lines in the image.

def detect_lines(image):
    """
    Use Canny edge detection and Hough transform to get selected lines 
    (which are useful for locating vanishing point) for all images
    
    Args: images: a list of original images
    
    Return: blur_images: Blurred images (for report)
    edge_images: Edge images (for report)
    valid_lines_all: Detected lines
    """
    # Do blurry to smooth the image, try to remove edges from textures
    gau_kernel = cv2.getGaussianKernel(70,4)# 1d gaussian kernel (size, sigma)
    gau_kern2d = np.outer(gau_kernel, gau_kernel)
    gau_kern2d = gau_kern2d/gau_kern2d.sum() # 2d gaussian kernel to do blurry
    # Apply blurry filter
    blur_image = cv2.filter2D(image,-1,gau_kern2d)
    # Canny edge detection with OpenCV for all blurry images
    edge_image = cv2.Canny(blur_image,40,70,apertureSize=3,L2gradient=True)
    # Use hough transform to detect all lines
    lines=cv2.HoughLines(edge_image, 1, np.pi/120, 55)
    valid_lines = []
    # Remove horizontal and vertical lines as they would not converge to vanishing point
    for line in lines:
        rho,theta = line[0]
        if (theta>0.4 and theta < 1.47) or (theta > 1.67 and theta < 2.74):
            valid_lines.append(line)
    
    return blur_image,edge_image,valid_lines

### 2. Locate the vanishing point
### Use RANSAC to locate the vanishing point from the detected lines.


#### 2.1 RANSAC functions
#### Define two fuctions required by RANSAC: a function to find the point where lines intersect, and a function to compute the distance from a point to a line.

# Find the intersection point
def find_intersection_point(line1,line2):
    """Implementation is based on code from https://stackoverflow.com/questions/46565975
    Original author: StackOverflow contributor alkasm 
    Find an intercept point of 2 lines model
    
    Args: line1,line2: 2 lines using rho and theta (polar coordinates) to represent
    
    Return: x0,y0: x and y for the intersection point
    """
    # rho and theta for each line
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    # Use formula from https://stackoverflow.com/a/383527/5087436 to solve for intersection between 2 lines 
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ]) 
    b = np.array([[rho1], [rho2]])
    det_A = np.linalg.det(A)
    if det_A != 0:
        x0, y0 = np.linalg.solve(A, b)
        # Round up x and y because pixel cannot have float number
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return x0, y0
    else:
        return None
        
    
# Find the distance from a point to a line
def find_dist_to_line(point,line):
    """Implementation is based on Computer Vision material, owned by the University of Melbourne
    Find an intercept point of the line model with a normal from point to it, to calculate the
    distance betwee point and intercept
    
    Args: point: the point using x and y to represent
    line: the line using rho and theta (polar coordinates) to represent
    
    Return: dist: the distance from the point to the line
    """
    x0,y0 = point
    rho, theta = line[0]
    m = (-1*(np.cos(theta)))/np.sin(theta)
    c = rho/np.sin(theta)
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c
    dist = math.sqrt((x - x0)**2 + (y - y0)**2)
    return dist


#### 2.2 RANSAC loop
#### Define the main RANSAC loop

def RANSAC(lines,ransac_iterations,ransac_threshold,ransac_ratio):
    """Implementation is based on code from Computer Vision material, owned by the University of Melbourne
    Use RANSAC to identify the vanishing points for all images
    
    Args: lines_all: The lines for all images
    ransac_iterations,ransac_threshold,ransac_ratio: RANSAC hyperparameters
    
    Return: vanishing_points: Estimated vanishing points for all images
    """
    # Store vanishing point for the image
    inlier_count_ratio = 0.
    vanishing_point = (0,0)
    # perform RANSAC iterations for each set of lines
    for iteration in range(ransac_iterations):
        # randomly sample 2 lines
        n = 2
        selected_lines = random.sample(lines,n)
        line1 = selected_lines[0]
        line2 = selected_lines[1]
        intersection_point = find_intersection_point(line1,line2)
        if intersection_point is not None:
            # count the number of inliers num
            inlier_count = 0
            # inliers are lines whose distance to the point is less than ransac_threshold
            for line in lines:
                # find the distance from the line to the point
                dist = find_dist_to_line(intersection_point,line)
                # check whether it's an inlier or not
                if dist < ransac_threshold:
                    inlier_count += 1

            # If the value of inlier_count is higher than previously saved value,
            # save it, and save the current point
            if inlier_count/float(len(lines)) > inlier_count_ratio:
                inlier_count_ratio = inlier_count/float(len(lines))
                vanishing_point = intersection_point

            # We are done in case we have enough inliers
            if inlier_count > len(lines)*ransac_ratio:
                break
    return vanishing_point


### 3. Main function
### Run your vanishing point detection method on a folder of images, return the (x,y) locations of the vanishing points


# RANSAC parameters:
ransac_iterations,ransac_threshold,ransac_ratio = 350,13,0.93

if __name__ == "__main__":
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    """For the image, Use Canny+Hough to detect edges, use RANSAC to identify the vanishing points
    """
    # Read images from folder
    image = load_images(filename)
    # Task1: Detect lines using Canny + Hough
    blur_image, edge_image, lines = detect_lines(image)
    # Show lines on the original images
    # show_lines(images,lines_all)
    # Task2: Use RANSAC to find all vanishing points
    vanishing_point = RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio)
    show_point(image, vanishing_point)
    #show_lines(image, lines)



"""
Hyper parameters tunning
Note: the following code is not expected to run while marking. It will take hours to run.
In addition, tuning for Canny edge detection and Hough Transform are done as well. They are described in the report.
"""

"""
# Use grid search to tune hyperparameters for RANSAC
best_result = 10000
best_model = None
for ransac_iterations in range (150,351,50):
    for ransac_threshold in range (10,15,1):
        for ransac_ratio in np.arange(0.81,1,0.02):
            sum_mse = 0
            # Run 10 times to alleviate the effect of stochasticity
            for it in range(0,10):
                vanishing_points = RANSAC(lines_all,ransac_iterations,ransac_threshold,ransac_ratio)
                single_mse = MSE(ground_truth,vanishing_points)
                sum_mse += single_mse
            mse_avg = sum_mse/10
            if mse_avg < best_result:
                best_result = mse_avg
                best_model = ransac_iterations,ransac_threshold,ransac_ratio
print("Best model is:"+
      "ransac_iterations="+str(best_model[0])+
      ", ransac_threshold="+str(best_model[1])+
      ", ransac_ratio="+str(best_model[2])+
      ", which give MSE "+str(best_result))
"""