# -*- coding: utf-8 -*-
"""dshah05.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FyQ7zjymL_2Syp9RbVq2vIY0UucKcDHc
"""

# Importing Google Drive
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

# Importing Dependencies
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

# Importing Video
video_path = "/content/drive/MyDrive/ENPM673/Project_1 /object_tracking.mp4"
vid = cv.VideoCapture(video_path)

# Initializing Variables

frame_count = 0
threshold = 5 # Threshold value for dark
pix = []  # Pixel information of all Centroids.
c_x = []  # Pixel X Co-ordinate
c_y = []  # Pixel Y Co-ordinate

"""**Step 1,2 & 3**

The below code block handles frames retrieved from a video. It first determines whether the video was successfully opened. Then it scans each frame, transforms it to grayscale, and uses a threshold to detect dark pixels. If it finds dark pixels, it calculates their centroids and adds them to the list. The loop runs until all frames have been handled or there is an error. Finally, it stops the video capture and prints the centroids' x-coordinates and count. This code is most likely part of a larger project that includes video analysis, such as object tracking or motion detection.
"""

# Extracting and Processing Frames
if not vid.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

while vid.isOpened():
    ret, frame = vid.read()

    if ret:
        file_path = f'/content/drive/MyDrive/ENPM673/Project_1 /Frames/frame_{frame_count}.jpg' # Saving Image to temporary file path
        # cv.imwrite(file_path, frame)  # Used to extract all frames.
        image = cv.imread(file_path)
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Converting to Grayscale

        arr = np.where(image_gray < threshold,0,255).astype(np.uint8)
        cord = np.where(arr == 0)  # Selecting all Dark Pixels
        if len(cord[0]>0):
          centroid_coord = [ np.mean(cord[1]), np.mean(cord[0])]
          centroid_x = (np.mean(cord[1]))
          centroid_y = (np.mean(cord[0]))
          pix.append(centroid_coord)
          c_x.append(centroid_x)
          c_y.append(centroid_y)

        frame_count += 1
    else:
        break

vid.release()
print(c_x)
print(len(c_x))

"""**Step 4**"""

# Curve Fitting
c_x = np.array(c_x) # Converting to Array
c_y = np.array(c_y) # Converting to Array

X = np.column_stack([np.ones_like(c_x), c_x, (c_x)**2]) # Standard Least Square Formula
B = np.linalg.inv(X.T @ X) @ X.T @ c_y  # Standard Least Mean Square Formula
c,b,a = B
Y = c + b*c_x + a*(c_x**2)

print(Y)
print(len(Y))

"""**Step 4  - Continued**"""

#Printing Pixels using MatPLotlib

pix = np.array(pix)  # Converting to array for better manipulation
plt.scatter(pix[:, 0], pix[:, 1], c='r', marker='o') #Plot before Curve Fitting
plt.plot(c_x, Y)  # Plot after Curve Fitting
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Centroid Coordinates')
plt.gca().invert_yaxis()  # Used to invert Y-axis to match original Video
plt.grid(True)
plt.show()

"""**Step 6**"""

# Drawing circles for each centroid coordinate

image_path = "/content/drive/MyDrive/ENPM673/Project_1 /Frames/frame_355.jpg"
image = cv.imread(image_path)
trajectory_plot = image.copy()

# Writing a for loop to plot each point on the image for the fitted curve. (As a line is just a collection of points)

for i in range(len(c_x)):
    cv.circle(trajectory_plot, (int(c_x[i]), int(Y[i])), radius=5, color=(0, 255, 0), thickness=-1)

# Writing a for loop to plot each point on the image for the calculated points.

for coord in pix:
    if not np.isnan(coord).any():
        cv.circle(trajectory_plot, (int(coord[0]), int(coord[1])), 6, (0, 0, 255), -1)

# Display the image with trajectory plot
plt.imshow(cv.cvtColor(trajectory_plot, cv.COLOR_BGR2RGB))
plt.show()

"""**Step 5**"""

# Inputing the value of X as 1000 as Y is already a quadratic curve and will give output of the curve.

# Calculating the value of Y when the x value is 1000

x_value = 1000
Y = c + b*x_value + a*(x_value**2)
print("Y value when c_x is 1000:", int(Y))