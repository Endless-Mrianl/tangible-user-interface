import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Step 1: Read image
image = cv2.imread("test5.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Mirror the image horizontally
image = cv2.flip(image, 1)  # 1 for horizontal flip, 0 for vertical flip, -1 for both

# Step 2: Convert to HSV and extract Hue channel
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
hue = hsv[:,:, 0] / 179.0  # Normalize Hue to [0, 1] as in MATLAB
def sub_matrix(hue_channel, num_blocks=5):
    h, w = hue_channel.shape
    # Calculate the size of each submatrix
    sub_row_size = h // num_blocks
    sub_col_size = w // num_blocks
    
    # Initialize lists to store submatrices and sums
    sub_matrices = []
    sum_matrix = np.zeros((num_blocks, num_blocks))
