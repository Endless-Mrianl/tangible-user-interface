import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Step 1: Read image
image = cv2.imread("test3.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Mirror the image horizontally
image = cv2.flip(image, 1)  # 1 for horizontal flip, 0 for vertical flip, -1 for both

# Step 2: Convert to HSV and extract Hue channel
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
hue = hsv[:, :, 0] / 179.0  # Normalize Hue to [0, 1] as in MATLAB

# Step 3: Divide into submatrices and compute sums
def sub_matrix(hue_channel, num_blocks=5):
    h, w = hue_channel.shape
    # Calculate the size of each submatrix
    sub_row_size = h // num_blocks
    sub_col_size = w // num_blocks
    
    # Initialize lists to store submatrices and sums
    sub_matrices = []
    sum_matrix = np.zeros((num_blocks, num_blocks))
    
    # Divide the matrix into submatrices
    for i in range(num_blocks):
        row_blocks = []
        for j in range(num_blocks):
            row_start = i * sub_row_size
            col_start = j * sub_col_size
            
            # Handle the last submatrix to include remaining rows/cols
            if i == num_blocks - 1:
                row_end = h
            else:
                row_end = (i + 1) * sub_row_size
                
            if j == num_blocks - 1:
                col_end = w
            else:
                col_end = (j + 1) * sub_col_size
            
            # Extract submatrix
            block = hue_channel[row_start:row_end, col_start:col_end]
            row_blocks.append(block)
            # Calculate sum of submatrix
            sum_matrix[i, j] = np.sum(block)
        sub_matrices.append(row_blocks)
    
    return sub_matrices, sum_matrix

# Use 30x30 blocks instead of 5x5
sub_matrices, sum_matrix = sub_matrix(hue, num_blocks=5)      #change num_block for change the matrix size 

# Step 4: Normalize to 0–255 like MATLAB code
max_new = sub_matrices[0][0].size  # Size of first submatrix
max_old = 255
new_values_old = (sum_matrix / max_old) * max_new
new_values = np.abs(new_values_old - 255)

# Step 5: Plot all like MATLAB
fig = plt.figure(figsize=(12, 8))

# Top-left: ToF image
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(image)
ax1.set_title("ToF image")
ax1.axis("off")

# Top-right: Hue channel
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(hue, cmap="gray")
ax2.set_title("Hue Channel")
ax2.axis("off")

# Bottom-left: Bar3-style block plot
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
xpos, ypos = np.meshgrid(np.arange(new_values.shape[1]), np.arange(new_values.shape[0]))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = 0.8
dz = new_values.flatten()

colors = cm.bone(dz / dz.max())
ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
ax3.set_title("Result")
ax3.view_init(elev=30, azim=135)

# Bottom-right: Surface plot
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
x, y = np.meshgrid(np.arange(new_values.shape[1]), np.arange(new_values.shape[0]))
ax4.plot_surface(x, y, new_values, cmap="bone", edgecolor='k')
ax4.set_title("Surface Plot")
ax4.view_init(elev=45, azim=135)

plt.tight_layout()
plt.show()

