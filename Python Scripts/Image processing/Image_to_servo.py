import cv2
import numpy as np
import serial
import time

# === SETTINGS ===
IMAGE_PATH = "T:/Scripts/Major Project/tangible-user-interface/Python Scripts/test3.png"
NUM_BLOCKS = 3  # NxN matrix size
SERIAL_PORT = "COM12"  # ESP32 port
BAUD_RATE = 115200

# === STEP 1: Load and process image ===
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.flip(image, 1)  # Horizontal flip

# === STEP 2: Extract hue channel ===
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
hue = hsv[:, :, 0] / 179.0  # Normalize hue to [0, 1]

# === STEP 3: Divide into NxN blocks and compute sums ===
def sub_matrix(hue_channel, num_blocks):
    h, w = hue_channel.shape
    sub_row_size = h // num_blocks
    sub_col_size = w // num_blocks
    
    sum_matrix = np.zeros((num_blocks, num_blocks))
    for i in range(num_blocks):
        for j in range(num_blocks):
            row_start = i * sub_row_size
            col_start = j * sub_col_size
            row_end = (i + 1) * sub_row_size if i != num_blocks - 1 else h
            col_end = (j + 1) * sub_col_size if j != num_blocks - 1 else w
            block = hue_channel[row_start:row_end, col_start:col_end]
            sum_matrix[i, j] = np.sum(block)
    return sum_matrix

sum_matrix = sub_matrix(hue, NUM_BLOCKS)
block_size = (image.shape[0] // NUM_BLOCKS) * (image.shape[1] // NUM_BLOCKS)

# === STEP 4: Normalize and convert sums to servo angles (0-180) ===
values = np.abs(sum_matrix - block_size)
flattened = values.flatten()
servo_angles = np.interp(flattened, [flattened.min(), flattened.max()], [0, 180])
servo_angles = [int(a) for a in servo_angles]

print("Servo Angles (degrees):", ','.join(map(str, servo_angles)))

# === STEP 5: Send angles to ESP32 via Serial (working method) ===
data_to_send = ",".join(map(str, servo_angles)) + "\n"

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        time.sleep(1)  # Wait for ESP32 reset

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        ser.write(data_to_send.encode())
        print(f"Sent to ESP32: {data_to_send.strip()}")

        # Optional: read response from ESP32
        time.sleep(0.5)
        while ser.in_waiting:
            response = ser.readline().decode().strip()
            print(f"ESP32 Response: {response}")

except Exception as e:
    print(f"Serial send error: {e}")
