import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time
import sys
import os
import serial



# Try to import ArduCAM TOF SDK
ARDUCAM_AVAILABLE = False
ArducamCamera = None
ArducamDepthCamera = None



SERIAL_PORT = "COM3"  # ESP32 port
BAUD_RATE = 115200
# Try multiple import paths for ArduCAM SDK


import_paths = [
    os.path.dirname(__file__),  # Current directory (example/python)
    os.path.join(os.path.dirname(__file__), '..'),  # Parent directory
    os.path.join(os.path.dirname(__file__), '../..'),  # Two levels up
]

for path in import_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
    try:
        import ArducamDepthCamera as ac
        ArducamDepthCamera = ac
        ArducamCamera = ac.ArducamCamera
        ARDUCAM_AVAILABLE = True
        print(f"ArduCAM TOF SDK loaded successfully")
        break
    except ImportError:
        continue

if not ARDUCAM_AVAILABLE:
    print("Warning: ArduCAM TOF SDK not found. Falling back to OpenCV VideoCapture.")
    print("Make sure you're running from the Arducam_tof_camera/example/python directory.")

class VideoProcessor:
    def __init__(self, num_blocks=[3, 5], video_source=0, use_arducam=True):
        """
        Initialize video processor
        num_blocks: Number of blocks to divide the frame into (grid size)
                   Can be a single integer or a list of integers (e.g., [3, 5])
        video_source: 0 for webcam/ArduCAM, or path to video file (only for OpenCV mode)
        use_arducam: If True, use ArduCAM TOF camera, else use OpenCV VideoCapture
        """
        # Convert single value to list if needed
        if isinstance(num_blocks, int):
            self.num_blocks_list = [num_blocks]
        else:
            self.num_blocks_list = num_blocks
        self.video_source = video_source
        self.use_arducam = use_arducam and ARDUCAM_AVAILABLE
        self.cap = None
        self.cam = None
        self.max_distance = 4000  # Default max distance in mm
        
        # Setup matplotlib for real-time plotting
        plt.ion()  # Turn on interactive mode
        # Create larger figure to accommodate multiple block sizes
        num_block_sizes = len(self.num_blocks_list)
        self.fig = plt.figure(figsize=(18, 10))
        
        # Create subplots - adjust layout based on number of block sizes
        if num_block_sizes == 1:
            # Original layout for single block size
            self.ax1 = self.fig.add_subplot(2, 3, 1)
            self.ax2 = self.fig.add_subplot(2, 3, 2)
            self.ax3 = self.fig.add_subplot(2, 3, 3, projection="3d")
            self.ax4 = self.fig.add_subplot(2, 3, 4)
            self.ax5 = self.fig.add_subplot(2, 3, 5, projection="3d")
            self.ax6 = self.fig.add_subplot(2, 3, 6)
        else:
            # Layout for multiple block sizes: 3 rows, 4 columns
            # Row 1: Original frame, Depth/Hue channel, Heatmap for block 1, Heatmap for block 2
            self.ax1 = self.fig.add_subplot(3, 4, 1)  # Original frame
            self.ax2 = self.fig.add_subplot(3, 4, 2)  # Depth/Hue channel
            self.ax3 = self.fig.add_subplot(3, 4, 3)  # Heatmap block 1
            self.ax4 = self.fig.add_subplot(3, 4, 4)  # Heatmap block 2
            # Row 2: 3D plots
            self.ax5 = self.fig.add_subplot(3, 4, 5, projection="3d")  # 3D bar block 1
            self.ax6 = self.fig.add_subplot(3, 4, 6, projection="3d")  # 3D bar block 2
            self.ax7 = self.fig.add_subplot(3, 4, 7, projection="3d")  # Surface block 1
            self.ax8 = self.fig.add_subplot(3, 4, 8, projection="3d")  # Surface block 2
            # Row 3: Text grids
            self.ax9 = self.fig.add_subplot(3, 4, 9)  # Text grid block 1
            self.ax10 = self.fig.add_subplot(3, 4, 10)  # Text grid block 2
        
        # Initialize plots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup initial plot configurations"""
        num_block_sizes = len(self.num_blocks_list)
        
        if num_block_sizes == 1:
            # Original single block size layout
            self.ax1.set_title("Original Frame / Depth Map")
            self.ax1.axis("off")
            
            if self.use_arducam:
                self.ax2.set_title("Depth Channel (Normalized)")
            else:
                self.ax2.set_title("Hue Channel")
            self.ax2.axis("off")
            
            self.ax3.set_title("3D Bar Plot - Servo Angles")
            self.ax4.set_title("Heatmap")
            self.ax5.set_title("Surface Plot")
            self.ax6.set_title("Servo Angles Grid")
            self.ax6.axis("off")
        else:
            # Multiple block sizes layout
            self.ax1.set_title("Original Frame / Depth Map")
            self.ax1.axis("off")
            
            if self.use_arducam:
                self.ax2.set_title("Depth Channel (Normalized)")
            else:
                self.ax2.set_title("Hue Channel")
            self.ax2.axis("off")
            
            # Heatmaps for each block size
            for i, num_blocks in enumerate(self.num_blocks_list):
                if i == 0:
                    self.ax3.set_title(f"Servo Angles Heatmap ({num_blocks}x{num_blocks})")
                elif i == 1:
                    self.ax4.set_title(f"Servo Angles Heatmap ({num_blocks}x{num_blocks})")
            
            # 3D plots
            self.ax5.set_title(f"3D Bar Plot ({self.num_blocks_list[0]}x{self.num_blocks_list[0]})")
            if len(self.num_blocks_list) > 1:
                self.ax6.set_title(f"3D Bar Plot ({self.num_blocks_list[1]}x{self.num_blocks_list[1]})")
            self.ax7.set_title(f"Surface Plot ({self.num_blocks_list[0]}x{self.num_blocks_list[0]})")
            if len(self.num_blocks_list) > 1:
                self.ax8.set_title(f"Surface Plot ({self.num_blocks_list[1]}x{self.num_blocks_list[1]})")
            
            # Text grids
            self.ax9.set_title(f"Servo Angles Grid ({self.num_blocks_list[0]}x{self.num_blocks_list[0]})")
            self.ax9.axis("off")
            if len(self.num_blocks_list) > 1:
                self.ax10.set_title(f"Servo Angles Grid ({self.num_blocks_list[1]}x{self.num_blocks_list[1]})")
                self.ax10.axis("off")
        
        plt.tight_layout()

    def sub_matrix(self, data_channel, num_blocks):
        """Extract submatrices and compute sums"""
        h, w = data_channel.shape
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
                
                block = data_channel[row_start:row_end, col_start:col_end]
                row_blocks.append(block)
                sum_matrix[i, j] = np.sum(block)
            sub_matrices.append(row_blocks)
        
        return sub_matrices, sum_matrix

    def process_frame(self, frame, depth_data=None):
        """Process a single frame for all block sizes
        frame: RGB frame (for OpenCV mode) or depth visualization (for ArduCAM mode)
        depth_data: Depth data array from ArduCAM (if using TOF camera)
        Returns: Dictionary with results for each block size
        """
        # First, get the processing channel (depth or hue)
        if self.use_arducam and depth_data is not None:
            # Process depth data from TOF camera
            depth_normalized = depth_data.astype(np.float32)
            
            # Normalize depth to [0, 1] range based on max_distance
            # Depth values are in millimeters, normalize by max_distance
            depth_normalized = np.clip(depth_normalized / self.max_distance, 0.0, 1.0)
            
            # Handle invalid/zero depth values
            depth_normalized = np.nan_to_num(depth_normalized)
            
            # Create visualization frame (depth map with colormap, same as preview_depth.py)
            result_image = (depth_normalized * 255).astype(np.uint8)
            frame_rgb = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            
            # Use normalized depth as the processing channel (similar to hue)
            processing_channel = depth_normalized
            
        else:
            # Original RGB processing for webcam/video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror horizontally
            
            # Convert to HSV and extract Hue
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            processing_channel = hsv[:, :, 0] / 179.0  # Normalize Hue to [0, 1]
        
        # Process for each block size
        results = {}
        for num_blocks in self.num_blocks_list:
            # Process submatrices
            sub_matrices, sum_matrix = self.sub_matrix(processing_channel, num_blocks)
            block_size = sub_matrices[0][0].size
            new_values = np.abs(sum_matrix - block_size)
            
            # Calculate servo angles
            servo_angles = np.interp(new_values.flatten(), 
                                   [new_values.min(), new_values.max()], 
                                   [0, 180])
            servo_matrix = servo_angles.reshape(new_values.shape)
            
            results[num_blocks] = {
                'new_values': new_values,
                'servo_matrix': servo_matrix,
                'servo_angles': servo_angles
            }
        
        return frame_rgb, processing_channel, results

    def update_plots(self, frame_rgb, processing_channel, results):
        """Update all plots with new data for all block sizes
        results: Dictionary with results for each block size
        """
        num_block_sizes = len(self.num_blocks_list)
        
        # Clear all axes
        if num_block_sizes == 1:
            # Single block size - original layout
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            self.ax5.clear()
            self.ax6.clear()
            
            num_blocks = self.num_blocks_list[0]
            result = results[num_blocks]
            new_values = result['new_values']
            servo_matrix = result['servo_matrix']
            
            # Plot 1: Original frame or depth visualization
            self.ax1.imshow(frame_rgb)
            if self.use_arducam:
                self.ax1.set_title("Depth Map Visualization")
            else:
                self.ax1.set_title("Original Frame")
            self.ax1.axis("off")
            
            # Plot 2: Processing channel (hue or depth)
            self.ax2.imshow(processing_channel, cmap="gray")
            if self.use_arducam:
                self.ax2.set_title("Depth Channel (Normalized)")
            else:
                self.ax2.set_title("Hue Channel")
            self.ax2.axis("off")
            
            # Plot 3: 3D Bar plot
            xpos, ypos = np.meshgrid(np.arange(new_values.shape[1]), 
                                    np.arange(new_values.shape[0]))
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)
            dx = dy = 0.8
            dz = new_values.flatten()
            
            colors = cm.bone(dz / dz.max() if dz.max() > 0 else 1)
            self.ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
            self.ax3.set_title("3D Bar Plot - Heights")
            self.ax3.view_init(elev=30, azim=135)
            
            # Plot 4: Heatmap
            im = self.ax4.imshow(servo_matrix, cmap='viridis', vmin=0, vmax=180)
            self.ax4.set_title("Servo Angles Heatmap")
            for i in range(servo_matrix.shape[0]):
                for j in range(servo_matrix.shape[1]):
                    self.ax4.text(j, i, f'{int(servo_matrix[i,j])}°', 
                                ha='center', va='center', color='white', fontweight='bold')
            
            # Plot 5: Surface plot
            x, y = np.meshgrid(np.arange(new_values.shape[1]), 
                              np.arange(new_values.shape[0]))
            self.ax5.plot_surface(x, y, new_values, cmap="bone", edgecolor='k')
            self.ax5.set_title("Surface Plot")
            self.ax5.view_init(elev=45, azim=135)
            
            # Plot 6: Servo angles as text grid
            servo_text = "\n".join([" | ".join([f"{int(angle):3d}°" for angle in row]) 
                                   for row in servo_matrix])
            self.ax6.text(0.1, 0.5, f"Servo Angles:\n{servo_text}", 
                         transform=self.ax6.transAxes, fontsize=12, 
                         verticalalignment='center', fontfamily='monospace')
            self.ax6.set_title("Servo Angles Grid")
            
        else:
            # Multiple block sizes - new layout
            # Clear all axes
            axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, 
                   self.ax6, self.ax7, self.ax8, self.ax9, self.ax10]
            for ax in axes:
                if ax is not None:
                    ax.clear()
            
            # Plot 1: Original frame or depth visualization
            self.ax1.imshow(frame_rgb)
            if self.use_arducam:
                self.ax1.set_title("Depth Map Visualization")
            else:
                self.ax1.set_title("Original Frame")
            self.ax1.axis("off")
            
            # Plot 2: Processing channel (hue or depth)
            self.ax2.imshow(processing_channel, cmap="gray")
            if self.use_arducam:
                self.ax2.set_title("Depth Channel (Normalized)")
            else:
                self.ax2.set_title("Hue Channel")
            self.ax2.axis("off")
            
            # Process each block size
            for idx, num_blocks in enumerate(self.num_blocks_list):
                result = results[num_blocks]
                new_values = result['new_values']
                servo_matrix = result['servo_matrix']
                
                if idx == 0:
                    # First block size - use ax3, ax5, ax7, ax9
                    # Heatmap - adjust text display based on grid size
                    im = self.ax3.imshow(servo_matrix, cmap='viridis', vmin=0, vmax=180)
                    self.ax3.set_title(f"Servo Angles Heatmap ({num_blocks}x{num_blocks})")
                    # Only show text if grid is small enough (<= 5x5)
                    if num_blocks <= 5:
                        font_size = max(8, 12 - num_blocks)
                        for i in range(servo_matrix.shape[0]):
                            for j in range(servo_matrix.shape[1]):
                                self.ax3.text(j, i, f'{int(servo_matrix[i,j])}°', 
                                            ha='center', va='center', color='white', 
                                            fontweight='bold', fontsize=font_size)
                    
                    # 3D Bar plot
                    xpos, ypos = np.meshgrid(np.arange(new_values.shape[1]), 
                                            np.arange(new_values.shape[0]))
                    xpos = xpos.flatten()
                    ypos = ypos.flatten()
                    zpos = np.zeros_like(xpos)
                    dx = dy = 0.8
                    dz = new_values.flatten()
                    colors = cm.bone(dz / dz.max() if dz.max() > 0 else 1)
                    self.ax5.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
                    self.ax5.set_title(f"3D Bar Plot ({num_blocks}x{num_blocks})")
                    self.ax5.view_init(elev=30, azim=135)
                    
                    # Surface plot
                    x, y = np.meshgrid(np.arange(new_values.shape[1]), 
                                      np.arange(new_values.shape[0]))
                    self.ax7.plot_surface(x, y, new_values, cmap="bone", edgecolor='k')
                    self.ax7.set_title(f"Surface Plot ({num_blocks}x{num_blocks})")
                    self.ax7.view_init(elev=45, azim=135)
                    
                    # Text grid - adjust font size based on grid size
                    if num_blocks <= 5:
                        servo_text = "\n".join([" | ".join([f"{int(angle):3d}°" for angle in row]) 
                                               for row in servo_matrix])
                        font_size = max(6, 12 - num_blocks * 1.5)
                    else:
                        # For larger grids, show summary or first few rows
                        servo_text = "\n".join([" | ".join([f"{int(angle):3d}°" for angle in row[:5]]) 
                                               for row in servo_matrix[:5]])
                        servo_text += f"\n... ({num_blocks}x{num_blocks} grid - showing first 5x5)"
                        font_size = 8
                    self.ax9.text(0.1, 0.5, f"Servo Angles ({num_blocks}x{num_blocks}):\n{servo_text}", 
                                 transform=self.ax9.transAxes, fontsize=font_size, 
                                 verticalalignment='center', fontfamily='monospace')
                    self.ax9.set_title(f"Servo Angles Grid ({num_blocks}x{num_blocks})")
                    self.ax9.axis("off")
                    
                elif idx == 1:
                    # Second block size - use ax4, ax6, ax8, ax10
                    # Heatmap - adjust text display based on grid size
                    im = self.ax4.imshow(servo_matrix, cmap='viridis', vmin=0, vmax=180)
                    self.ax4.set_title(f"Servo Angles Heatmap ({num_blocks}x{num_blocks})")
                    # Only show text if grid is small enough (<= 5x5)
                    if num_blocks <= 5:
                        font_size = max(8, 12 - num_blocks)
                        for i in range(servo_matrix.shape[0]):
                            for j in range(servo_matrix.shape[1]):
                                self.ax4.text(j, i, f'{int(servo_matrix[i,j])}°', 
                                            ha='center', va='center', color='white', 
                                            fontweight='bold', fontsize=font_size)
                    
                    # 3D Bar plot
                    xpos, ypos = np.meshgrid(np.arange(new_values.shape[1]), 
                                            np.arange(new_values.shape[0]))
                    xpos = xpos.flatten()
                    ypos = ypos.flatten()
                    zpos = np.zeros_like(xpos)
                    dx = dy = 0.8
                    dz = new_values.flatten()
                    colors = cm.bone(dz / dz.max() if dz.max() > 0 else 1)
                    self.ax6.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
                    self.ax6.set_title(f"3D Bar Plot ({num_blocks}x{num_blocks})")
                    self.ax6.view_init(elev=30, azim=135)
                    
                    # Surface plot
                    x, y = np.meshgrid(np.arange(new_values.shape[1]), 
                                      np.arange(new_values.shape[0]))
                    self.ax8.plot_surface(x, y, new_values, cmap="bone", edgecolor='k')
                    self.ax8.set_title(f"Surface Plot ({num_blocks}x{num_blocks})")
                    self.ax8.view_init(elev=45, azim=135)
                    
                    # Text grid - adjust font size based on grid size
                    if num_blocks <= 5:
                        servo_text = "\n".join([" | ".join([f"{int(angle):3d}°" for angle in row]) 
                                               for row in servo_matrix])
                        font_size = max(6, 12 - num_blocks * 1.5)
                    else:
                        # For larger grids, show summary or first few rows
                        servo_text = "\n".join([" | ".join([f"{int(angle):3d}°" for angle in row[:5]]) 
                                               for row in servo_matrix[:5]])
                        servo_text += f"\n... ({num_blocks}x{num_blocks} grid - showing first 5x5)"
                        font_size = 8
                    self.ax10.text(0.1, 0.5, f"Servo Angles ({num_blocks}x{num_blocks}):\n{servo_text}", 
                                  transform=self.ax10.transAxes, fontsize=font_size, 
                                  verticalalignment='center', fontfamily='monospace')
                    self.ax10.set_title(f"Servo Angles Grid ({num_blocks}x{num_blocks})")
                    self.ax10.axis("off")
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to update display

    # --- Replace start_video_processing in VideoProcessor with this version ---
    def start_video_processing(self):
        """Start video processing loop and send servo angles via serial like the webcam script"""
        # Open serial once (change SERIAL_PORT to '/dev/ttyUSB0' on Raspberry Pi if needed)
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(1)  # let ESP32 reset if it does on open
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print(f"Serial opened on {SERIAL_PORT} at {BAUD_RATE}")
        except Exception as e:
            ser = None
            print(f"Warning: could not open serial {SERIAL_PORT}: {e}")
            print("Will continue without sending serial data.")

        if self.use_arducam:
            # Initialize ArduCAM TOF camera
            try:
                self.cam = ArducamCamera()
                ret = self.cam.open(ArducamDepthCamera.Connection.CSI, 0)
                if ret != 0:
                    print(f"Error: Could not open ArduCAM TOF camera. Error code: {ret}")
                    return

                ret = self.cam.start(ArducamDepthCamera.FrameType.DEPTH)
                if ret != 0:
                    print(f"Error: Could not start ArduCAM camera. Error code: {ret}")
                    self.cam.close()
                    return

                MAX_DISTANCE = 4000
                self.cam.setControl(ArducamDepthCamera.Control.RANGE, MAX_DISTANCE)
                self.max_distance = self.cam.getControl(ArducamDepthCamera.Control.RANGE)

                info = self.cam.getCameraInfo()
                print("ArduCAM TOF Camera initialized successfully")
                print(f"Camera resolution: {info.width}x{info.height}")
                print(f"Max distance: {self.max_distance}mm")
            except Exception as e:
                print(f"Error initializing ArduCAM camera: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to OpenCV VideoCapture...")
                self.use_arducam = False
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    print(f"Error: Could not open video source {self.video_source}")
                    return
        else:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                print(f"Error: Could not open video source {self.video_source}")
                return

        print("Starting video processing. Press 'q' to quit.")
        print("Press 's' to save current servo angles to file.")

        frame_count = 0

        try:
            while True:
                if self.use_arducam:
                    # Get frame from ArduCAM TOF camera
                    try:
                        frame = self.cam.requestFrame(2000)  # 2000ms timeout
                        if frame is None:
                            print("Failed to capture frame from ArduCAM")
                            time.sleep(0.1)
                            continue

                        if not isinstance(frame, ArducamDepthCamera.DepthData):
                            print("Frame is not DepthData type")
                            self.cam.releaseFrame(frame)
                            time.sleep(0.1)
                            continue

                        depth_data = frame.depth_data
                        confidence_data = frame.confidence_data

                        confidence_value = 30
                        depth_data = depth_data.copy()
                        depth_data[confidence_data < confidence_value] = 0

                        frame_rgb, processing_channel, results = \
                            self.process_frame(None, depth_data=depth_data)

                        self.cam.releaseFrame(frame)

                    except Exception as e:
                        print(f"Error capturing frame: {e}")
                        import traceback
                        traceback.print_exc()
                        try:
                            if 'frame' in locals():
                                self.cam.releaseFrame(frame)
                        except:
                            pass
                        time.sleep(0.1)
                        continue

                else:
                    # Get frame from OpenCV
                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video or failed to capture frame")
                        break

                    frame_rgb, processing_channel, results = self.process_frame(frame)

                # Update plots every few frames
                if frame_count % 2 == 0:
                    self.update_plots(frame_rgb, processing_channel, results)

                # Send servo angles (for the first block size) every frame (or change frequency)
                try:
                    if len(self.num_blocks_list) > 0:
                        first_block_size = self.num_blocks_list[0]
                        result = results[first_block_size]
                        servo_matrix = result['servo_matrix']
                        # Flatten and convert to integers like the working code
                        flattened = servo_matrix.flatten()
                        servo_angles = [int(a) for a in flattened]
                        data_to_send = ",".join(map(str, servo_angles)) + "\n"

                        if ser is not None and ser.is_open:
                            ser.write(data_to_send.encode())
                            # optional small delay
                            time.sleep(0.01)
                            # read any response
                            resp = []
                            while ser.in_waiting:
                                response = ser.readline().decode(errors='ignore').strip()
                                if response:
                                    resp.append(response)
                            if resp:
                                for r in resp:
                                    print(f"ESP32 Response: {r}")
                            # Print what was sent (console)
                            print(f"Sent to ESP32: {data_to_send.strip()}")
                        else:
                            # If serial not open, still print angles
                            print(f"(No serial) Servo Angles Sent: {','.join(map(str,servo_angles))}")
                except Exception as e:
                    print(f"Serial send error in loop: {e}")

                # Print servo angles to console less frequently
                if frame_count % 10 == 0:
                    if len(self.num_blocks_list) > 0:
                        first_block_size = self.num_blocks_list[0]
                        result = results[first_block_size]
                        servo_matrix = result['servo_matrix']
                        print(f"\nFrame {frame_count} - Servo Angles ({first_block_size}x{first_block_size}):")
                        print(servo_matrix.astype(int))

                frame_count += 1

                # Check for quit or save
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('s'):
                        for num_blocks in self.num_blocks_list:
                            result = results[num_blocks]
                            servo_matrix = result['servo_matrix']
                            self.save_servo_angles(servo_matrix, frame_count, num_blocks)
                except:
                    pass

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")

        finally:
            # cleanup serial
            try:
                if ser is not None and ser.is_open:
                    ser.close()
                    print("Serial closed")
            except Exception as e:
                print(f"Error closing serial: {e}")
            self.cleanup()


# --- Replace SimpleVideoProcessor.process_video_simple with this version ---
    def process_video_simple(self, video_source=0):
        """Simple video processing that only outputs servo angles and sends them via serial"""
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        # Try opening serial once
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(1)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print(f"Serial opened on {SERIAL_PORT}")
        except Exception as e:
            ser = None
            print(f"Warning: could not open serial {SERIAL_PORT}: {e}")

        print("Simple video processing started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)

            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue = hsv[:, :, 0] / 179.0

            h, w = hue.shape
            sub_row_size = h // self.num_blocks
            sub_col_size = w // self.num_blocks

            servo_angles_matrix = []
            for i in range(self.num_blocks):
                row_angles = []
                for j in range(self.num_blocks):
                    row_start = i * sub_row_size
                    col_start = j * sub_col_size
                    row_end = (i + 1) * sub_row_size if i != self.num_blocks - 1 else h
                    col_end = (j + 1) * sub_col_size if j != self.num_blocks - 1 else w

                    block = hue[row_start:row_end, col_start:col_end]
                    block_sum = np.sum(block)
                    block_size = block.size
                    value = abs(block_sum - block_size)

                    angle = np.interp(value, [0, block_size], [0, 180])
                    row_angles.append(int(angle))
                servo_angles_matrix.append(row_angles)

            # Flatten and send like your webcam script
            flat_angles = [int(a) for row in servo_angles_matrix for a in row]
            data_to_send = ",".join(map(str, flat_angles)) + "\n"

            try:
                if ser is not None and ser.is_open:
                    ser.write(data_to_send.encode())
                    time.sleep(0.01)
                    print(f"Sent to ESP32: {data_to_send.strip()}")
                    # optional read
                    while ser.in_waiting:
                        response = ser.readline().decode(errors='ignore').strip()
                        if response:
                            print(f"ESP32 Response: {response}")
                else:
                    print(f"(No serial) Servo Angles: {data_to_send.strip()}")
            except Exception as e:
                print(f"Serial send error: {e}")

            # Show frame (optional)
            cv2.imshow('Video Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        try:
            if ser is not None and ser.is_open:
                ser.close()
        except:
            pass

    def save_servo_angles(self, servo_matrix, frame_num, num_blocks=None):
        """Save current servo angles to file"""
        if num_blocks is not None:
            filename = f"servo_angles_{num_blocks}x{num_blocks}_frame_{frame_num}.txt"
        else:
            filename = f"servo_angles_frame_{frame_num}.txt"
        np.savetxt(filename, servo_matrix.astype(int), fmt='%d', delimiter=',')
        print(f"Servo angles saved to {filename}")

    def cleanup(self):
        """Clean up resources"""
        if self.use_arducam and self.cam:
            try:
                self.cam.stop()
                self.cam.close()
            except Exception as e:
                print(f"Error during ArduCAM cleanup: {e}")
        elif self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        plt.close('all')
        print("Cleanup completed")

# Usage example
if __name__ == "__main__":
    # For ArduCAM TOF camera (default on Raspberry Pi)
    if ARDUCAM_AVAILABLE:
        print("Using ArduCAM TOF Camera")
        # Process with both 3x3 and 5x5 block sizes simultaneously
        processor = VideoProcessor(num_blocks=[3, 15], video_source=0, use_arducam=True)
    else:
        print("ArduCAM SDK not available, using OpenCV VideoCapture")
        # For webcam (fallback) - can also use multiple block sizes
        processor = VideoProcessor(num_blocks=[3, 15], video_source=0, use_arducam=False)
    
    # For video file, uncomment and modify path:
    # processor = VideoProcessor(num_blocks=[3, 5], video_source=r"path_to_your_video.mp4", use_arducam=False)
    
    # For single block size, use:
    # processor = VideoProcessor(num_blocks=3, video_source=0, use_arducam=True)
    
    processor.start_video_processing()


# Alternative: Simplified version for real-time servo control only
class SimpleVideoProcessor:
    def __init__(self, num_blocks=3):
        self.num_blocks = num_blocks
        
    def process_video_simple(self, video_source=0):
        """Simple video processing that only outputs servo angles"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        print("Simple video processing started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)
            
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue = hsv[:, :, 0] / 179.0
            
            # Calculate servo angles
            h, w = hue.shape
            sub_row_size = h // self.num_blocks
            sub_col_size = w // self.num_blocks
            
            servo_angles = []
            for i in range(self.num_blocks):
                row_angles = []
                for j in range(self.num_blocks):
                    row_start = i * sub_row_size
                    col_start = j * sub_col_size
                    row_end = (i + 1) * sub_row_size if i != self.num_blocks - 1 else h
                    col_end = (j + 1) * sub_col_size if j != self.num_blocks - 1 else w
                    
                    block = hue[row_start:row_end, col_start:col_end]
                    block_sum = np.sum(block)
                    block_size = block.size
                    value = abs(block_sum - block_size)
                    
                    # Map to servo angle (0-180 degrees)
                    angle = np.interp(value, [0, block_size], [0, 180])
                    row_angles.append(int(angle))
                servo_angles.append(row_angles)
            
            # Print servo angles
            print("Servo Angles:")
            for row in servo_angles:
                print(" | ".join([f"{angle:3d}°" for angle in row]))
            print("-" * 20)
            
            # Show frame (optional)
            cv2.imshow('Video Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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
        
        cap.release()
        cv2.destroyAllWindows()

# To use simple version:
# simple_processor = SimpleVideoProcessor(num_blocks=3)
# simple_processor.process_video_simple(0)  # 0 for webcam