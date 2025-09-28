import serial
import time

SERIAL_PORT = "COM12"
BAUD_RATE = 115200

#angles = [0, 30, 60, 90, 120, 150, 180, 90, 45]
angles = [0,0,0,0,0,0,0,0,0]
data_to_send = ",".join(map(str, angles)) + "\n"

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        time.sleep(4)  # Wait more for ESP32 reset

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        ser.write(data_to_send.encode())
        print(f"Sent: {data_to_send.strip()}")

        # Optional: read response from ESP32
        time.sleep(0.5)
        while ser.in_waiting:
            response = ser.readline().decode().strip()
            print(f"ESP32 Response: {response}")

except Exception as e:
    print(f"Error: {e}")
