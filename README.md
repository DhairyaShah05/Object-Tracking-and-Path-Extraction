# Object Tracking and Centroid Detection

This project uses a Python script to process video frames, detect dark pixels, calculate their centroids, and perform curve fitting for object tracking. The script is implemented using OpenCV, NumPy, and Matplotlib libraries. It also integrates with Google Colab for handling files stored in Google Drive.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dependencies](#dependencies)
3. [File Structure](#file-structure)
4. [Project Workflow](#project-workflow)
5. [Script Breakdown](#script-breakdown)
   - [1. Mounting Google Drive](#1-mounting-google-drive)
   - [2. Importing Dependencies](#2-importing-dependencies)
   - [3. Loading the Video](#3-loading-the-video)
   - [4. Extracting and Processing Frames](#4-extracting-and-processing-frames)
   - [5. Curve Fitting](#5-curve-fitting)
   - [6. Visualization and Results](#6-visualization-and-results)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Getting Started

To get started with this project, you need to have access to Google Colab and a Google Drive account where the necessary video files are stored. This project is designed to run in the Colab environment due to its integration with Google Drive and other Colab-specific functionalities.

## Dependencies

The script requires the following Python libraries:

- `cv2`: OpenCV library for computer vision tasks.
- `numpy`: For numerical computations.
- `matplotlib`: For data visualization.
- `google.colab.patches`: For displaying images within Google Colab.

These libraries can be installed using the following commands:

```bash
pip install opencv-python-headless
pip install numpy
pip install matplotlib
```

## File Structure

The file structure is organized as follows:

```
Project Directory/
├── dshah05.ipynb                # The main script file
├── object_tracking.mp4          # Input video file (stored in Google Drive)
└── Frames/                      # Directory to store extracted frames
    ├── frame_0.jpg
    ├── frame_1.jpg
    └── ...
```

## Project Workflow

1. **Mount Google Drive**: The script mounts the Google Drive to access the video and save frames.
2. **Video Processing**: Each frame of the video is analyzed to find dark pixels, and their centroids are calculated.
3. **Curve Fitting**: A quadratic curve is fitted to the centroids to model the trajectory.
4. **Visualization**: The centroids and the fitted curve are plotted to visualize the object's motion.

## Script Breakdown

### 1. Mounting Google Drive

```python
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
```

This section mounts Google Drive to access files stored in the user's drive. It uses the `force_remount=True` parameter to ensure the drive is remounted even if it was previously mounted.

### 2. Importing Dependencies

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
```

This section imports necessary libraries for video processing (`cv2`), numerical operations (`numpy`), and plotting (`matplotlib`).

### 3. Loading the Video

```python
video_path = "/content/drive/MyDrive/ENPM673/Project_1 /object_tracking.mp4"
vid = cv.VideoCapture(video_path)
```

The video file is loaded from Google Drive. The script will process this video to detect and track objects.

### 4. Extracting and Processing Frames

```python
if not vid.isOpened():
    print("Error: Couldn't open the video file.")
    exit()
```

This code checks if the video file was successfully opened. If not, it exits the script. The main loop reads each frame, converts it to grayscale, and applies a threshold to detect dark pixels.

- **Centroid Calculation**: The script calculates the centroids of detected dark pixels and stores them for later use in curve fitting.

```python
cord = np.where(arr == 0)  # Selecting all Dark Pixels
centroid_coord = [ np.mean(cord[1]), np.mean(cord[0])]
```

### 5. Curve Fitting

```python
X = np.column_stack([np.ones_like(c_x), c_x, (c_x)**2])
B = np.linalg.inv(X.T @ X) @ X.T @ c_y
```

This section performs quadratic curve fitting using the Least Squares method. The fitted curve is represented as:

\[ Y = c + b \times x + a \times x^2 \]

The coefficients `a`, `b`, and `c` are calculated using the formula above.

### 6. Visualization and Results

```python
plt.scatter(pix[:, 0], pix[:, 1], c='r', marker='o') # Plot before Curve Fitting
plt.plot(c_x, Y)  # Plot after Curve Fitting
```

This section visualizes the centroids and the fitted curve using Matplotlib. It also inverts the y-axis to match the original video orientation.

### 7. Drawing Centroids on Video Frames

```python
for i in range(len(c_x)):
    cv.circle(trajectory_plot, (int(c_x[i]), int(Y[i])), radius=5, color=(0, 255, 0), thickness=-1)
```

Circles are drawn on the original video frame for each calculated and fitted centroid. This allows for visual comparison of the detected centroids and the fitted trajectory.

## Usage

1. Clone or download the script to your local environment or Google Colab.
2. Ensure that the video file is present in the specified Google Drive directory.
3. Run the script in Google Colab to visualize and track the object in the video.

## Contributing

If you wish to contribute to this project, feel free to fork the repository and submit a pull request. Any suggestions for improvements or new features are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
