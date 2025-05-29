import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import serial
import time

# === SETTINGS ===
IMAGE_PATH = "test5.png"
NUM_BLOCKS = 3  # Set your NxM matrix size here
SERIAL_PORT = "COM5"  # Change to your ESP32 port
BAUD_RATE = 9600


# === STEP 1: Load and process image ===
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.flip(image, 1)  # Horizontal flip

# === STEP 2: Extract hue channel ===
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
hue = hsv[:, :, 0] / 179.0  # Normalize hue to [0, 1]

# === STEP 3: Divide into NxM blocks and compute sums ===
def sub_matrix(hue_channel, num_blocks):
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

sub_matrices, sum_matrix = sub_matrix(hue, NUM_BLOCKS)
block_size = sub_matrices[0][0].size

# === STEP 4: Normalize and convert to servo angles (0–180 degrees) ===
values = np.abs(sum_matrix - block_size)
flattened = values.flatten()
servo_angles = np.interp(flattened, [flattened.min(), flattened.max()], [0, 180])
servo_angles = [int(a) for a in servo_angles]

print("Servo Angles (degrees):", servo_angles)


# === STEP 5: Send angles to ESP32 via Serial ===
def send_angles_to_esp32(angles, port=SERIAL_PORT, baud=BAUD_RATE):
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            time.sleep(2)  # Wait for ESP32 to reset
            data = ",".join(map(str, angles)) + "\n"
            ser.write(data.encode())
            print(f"Sent to ESP32: {data.strip()}")
    except Exception as e:
        print("Serial send error:", e)

send_angles_to_esp32(servo_angles)


# === STEP 6: Plot for verification ===
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis("off")

ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(hue, cmap="gray")
ax2.set_title("Hue Channel")
ax2.axis("off")

ax3 = fig.add_subplot(2, 2, 3, projection="3d")
xpos, ypos = np.meshgrid(np.arange(values.shape[1]), np.arange(values.shape[0]))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = 0.8
dz = values.flatten()
colors = cm.bone(dz / dz.max())
ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
ax3.set_title("3D Bar Plot")
ax3.view_init(elev=30, azim=135)

ax4 = fig.add_subplot(2, 2, 4, projection="3d")
x, y = np.meshgrid(np.arange(values.shape[1]), np.arange(values.shape[0]))
ax4.plot_surface(x, y, values, cmap="bone", edgecolor='k')
ax4.set_title("Surface Plot")
ax4.view_init(elev=45, azim=135)

plt.tight_layout()
plt.show()