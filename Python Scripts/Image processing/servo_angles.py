#************************it is the final one for nxm matric no hardware**************************** 
#************************it is the final one for nxm matric no hardware****************************
#************************it is the final one for nxm matric no hardware****************************
#************************it is the final one for nxm matric no hardware****************************
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Step 1: Read image
image = cv2.imread("./Python Scripts/Image processing/test3.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Mirror the image horizontally
image = cv2.flip(image, 1)

# Step 2: Convert to HSV and extract Hue channel
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
hue = hsv[:, :, 0] / 179.0  # Normalize Hue to [0, 1]

# Step 3: Divide into submatrices and compute sums
def sub_matrix(hue_channel, num_blocks=5):
    h, w = hue_channel.shape
    sub_row_size = h // num_blocks
    sub_col_size = w // num_blocks
    
    sub_matrices = []
    sum_matrix = np.zeros((num_blocks, num_blocks))
    
    for i in range(num_blocks):
        row_blocks = []
        for j in range(num_blocks):
            row_start = i * sub_row_size
            col_start = j * sub_col_size
            
            row_end = (i + 1) * sub_row_size if i != num_blocks - 1 else h
            col_end = (j + 1) * sub_col_size if j != num_blocks - 1 else w
            
            block = hue_channel[row_start:row_end, col_start:col_end]
            row_blocks.append(block)
            sum_matrix[i, j] = np.sum(block)
        sub_matrices.append(row_blocks)
    
    return sub_matrices, sum_matrix

# Process for both block sizes
sub_matrices_5, sum_matrix_5 = sub_matrix(hue, num_blocks=3
)

# Corrected normalization
block_size_5 = sub_matrices_5[0][0].size


new_values_5 = np.abs(sum_matrix_5 - block_size_5)

# Plotting
fig = plt.figure(figsize=(15, 10))

# Original image
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis("off")

# Hue channel
ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(hue, cmap="gray")
ax2.set_title("Hue Channel")
ax2.axis("off")

# 20x20 blocks - Bar3D
ax3 = fig.add_subplot(2, 3, 3, projection="3d")
xpos, ypos = np.meshgrid(np.arange(new_values_5.shape[1]), np.arange(new_values_5.shape[0]))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = 0.8
dz = new_values_5.flatten()
# Map bar heights (dz) to servo angles between 0 and 180 degrees
servo_angles = np.interp(dz, [dz.min(), dz.max()], [0, 180])

colors = cm.bone(dz / dz.max())
ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
# Annotate each 3D bar with its corresponding servo angle  **********************for servo anglres on graph ********************************
#for i in range(len(xpos)):
 #   ax3.text(
  #      xpos[i], ypos[i], dz[i] + 3,   # Position slightly above each bar
   ##    ha='center', fontsize=8, color='black'
    #)

# Print servo angles before showing the plot
print("\nServo Angles (in degrees):")
for angle in servo_angles:
    print(f"{int(angle)}°")

ax3.set_title("20x20 Blocks - Bar3D")
ax3.view_init(elev=30, azim=135)


# 20x20 blocks - Surface plot
ax5 = fig.add_subplot(2, 3, 5, projection="3d")
x, y = np.meshgrid(np.arange(new_values_5.shape[1]), np.arange(new_values_5.shape[0]))
ax5.plot_surface(x, y, new_values_5, cmap="bone", edgecolor='k')
ax5.set_title("20x20 Blocks - Surface")
ax5.view_init(elev=45, azim=135)

plt.tight_layout()
plt.show()
