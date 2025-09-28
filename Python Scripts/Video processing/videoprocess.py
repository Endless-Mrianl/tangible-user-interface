import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time

class VideoProcessor:
    def __init__(self, num_blocks=3, video_source=0):
        """
        Initialize video processor
        video_source: 0 for webcam, or path to video file
        """
        self.num_blocks = num_blocks
        self.video_source = video_source
        self.cap = None
        self.min_blocks = 2
        self.max_blocks = 20
        
        # Setup matplotlib for real-time plotting
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(15, 8))
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax3 = self.fig.add_subplot(2, 3, 3, projection="3d")
        self.ax4 = self.fig.add_subplot(2, 3, 4)
        self.ax5 = self.fig.add_subplot(2, 3, 5, projection="3d")
        self.ax6 = self.fig.add_subplot(2, 3, 6)
        
        # Initialize plots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup initial plot configurations"""
        self.ax1.set_title("Original Frame")
        self.ax1.axis("off")
        
        self.ax2.set_title("Hue Channel")
        self.ax2.axis("off")
        
        self.ax3.set_title("3D Bar Plot - Servo Angles")
        
        self.ax4.set_title("Heatmap")
        
        self.ax5.set_title("Surface Plot")
        
        self.ax6.set_title("Servo Angles Grid")
        self.ax6.axis("off")
        
        plt.tight_layout()

    def sub_matrix(self, hue_channel, num_blocks):
        """Extract submatrices and compute sums"""
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

    def process_frame(self, frame):
        """Process a single frame"""
        # Convert color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror horizontally
        
        # Convert to HSV and extract Hue
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[:, :, 0] / 179.0  # Normalize Hue to [0, 1]
        
        # Process submatrices
        sub_matrices, sum_matrix = self.sub_matrix(hue, self.num_blocks)
        block_size = sub_matrices[0][0].size
        new_values = np.abs(sum_matrix - block_size)
        
        # Calculate servo angles
        servo_angles = np.interp(new_values.flatten(), 
                               [new_values.min(), new_values.max()], 
                               [0, 180])
        servo_matrix = servo_angles.reshape(new_values.shape)
        
        return frame_rgb, hue, new_values, servo_matrix, servo_angles

    def update_plots(self, frame_rgb, hue, new_values, servo_matrix, servo_angles):
        """Update all plots with new data"""
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        
        # Plot 1: Original frame
        self.ax1.imshow(frame_rgb)
        self.ax1.set_title("Original Frame")
        self.ax1.axis("off")
        
        # Plot 2: Hue channel
        self.ax2.imshow(hue, cmap="gray")
        self.ax2.set_title("Hue Channel")
        self.ax2.axis("off")
        
        # Plot 3: 3D Bar plot
        if new_values.size > 0:  # Check if we have data
            xpos, ypos = np.meshgrid(np.arange(new_values.shape[1]), 
                                    np.arange(new_values.shape[0]))
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)
            dx = dy = 0.8
            dz = new_values.flatten()
            
            if dz.max() > 0:
                colors = cm.bone(dz / dz.max())
            else:
                colors = 'blue'
            self.ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
        self.ax3.set_title(f"3D Bar Plot - {self.num_blocks}x{self.num_blocks} Blocks")
        self.ax3.view_init(elev=30, azim=135)
        
        # Plot 4: Heatmap
        im = self.ax4.imshow(servo_matrix, cmap='viridis', vmin=0, vmax=180)
        self.ax4.set_title(f"Servo Angles - {self.num_blocks}x{self.num_blocks}")
        
        # Add text annotations (only if blocks are not too small to read)
        if self.num_blocks <= 10:
            for i in range(servo_matrix.shape[0]):
                for j in range(servo_matrix.shape[1]):
                    self.ax4.text(j, i, f'{int(servo_matrix[i,j])}°', 
                                ha='center', va='center', color='white', fontweight='bold')
        
        # Plot 5: Surface plot
        x, y = np.meshgrid(np.arange(new_values.shape[1]), 
                          np.arange(new_values.shape[0]))
        self.ax5.plot_surface(x, y, new_values, cmap="bone", edgecolor='k')
        self.ax5.set_title(f"Surface Plot - {self.num_blocks}x{self.num_blocks}")
        self.ax5.view_init(elev=45, azim=135)
        
        # Plot 6: Servo angles as text grid
        servo_text = "\n".join([" | ".join([f"{int(angle):3d}°" for angle in row]) 
                               for row in servo_matrix])
        self.ax6.text(0.1, 0.5, f"Servo Angles:\n{servo_text}", 
                     transform=self.ax6.transAxes, fontsize=12, 
                     verticalalignment='center', fontfamily='monospace')
        self.ax6.set_title("Servo Angles Grid")
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to update display

    def start_video_processing(self):
        """Start video processing loop"""
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return
        
        print("Starting video processing. Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current servo angles to file")
        print("- Press '+' to increase block size")
        print("- Press '-' to decrease block size") 
        print("- Press '1'-'9' to set specific block size")
        print("- Press 'r' to reset to original block size")
        print(f"- Current block size: {self.num_blocks}x{self.num_blocks}")
        
        original_blocks = self.num_blocks  # Store original for reset
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video or failed to capture frame")
                    break
                
                # Process frame
                frame_rgb, hue, new_values, servo_matrix, servo_angles = self.process_frame(frame)
                
                # Update plots every few frames to maintain performance
                if frame_count % 2 == 0:  # Update every 2nd frame
                    self.update_plots(frame_rgb, hue, new_values, servo_matrix, servo_angles)
                
                # Print servo angles to console
                if frame_count % 10 == 0:  # Print every 10th frame
                    print(f"\nFrame {frame_count} - Servo Angles:")
                    print(servo_matrix.astype(int))
                
                frame_count += 1
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_servo_angles(servo_matrix, frame_count)
                elif key == ord('+') or key == ord('='):
                    if self.num_blocks < self.max_blocks:
                        self.num_blocks += 1
                        print(f"Block size increased to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('-') or key == ord('_'):
                    if self.num_blocks > self.min_blocks:
                        self.num_blocks -= 1
                        print(f"Block size decreased to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('r'):
                    self.num_blocks = original_blocks  # Reset to original
                    print(f"Block size reset to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('1'):
                    self.num_blocks = 1
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('2'):
                    self.num_blocks = 2
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('3'):
                    self.num_blocks = 3
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('4'):
                    self.num_blocks = 4
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('5'):
                    self.num_blocks = 5
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('6'):
                    self.num_blocks = 6
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('7'):
                    self.num_blocks = 7
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('8'):
                    self.num_blocks = 8
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                elif key == ord('9'):
                    self.num_blocks = 9
                    print(f"Block size set to {self.num_blocks}x{self.num_blocks}")
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            self.cleanup()

    def save_servo_angles(self, servo_matrix, frame_num):
        """Save current servo angles to file"""
        filename = f"servo_angles_frame_{frame_num}.txt"
        np.savetxt(filename, servo_matrix.astype(int), fmt='%d', delimiter=',')
        print(f"Servo angles saved to {filename}")

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        print("Cleanup completed")

# Usage example
if __name__ == "__main__":
    # Option 1: Use your video file
    video_path = r"./Python Scripts/Video processing/shapes_loop.mp4"
    processor = VideoProcessor(num_blocks=3, video_source=video_path)
    
    # Option 2: For webcam (uncomment to use)
    # processor = VideoProcessor(num_blocks=3, video_source=0)
    
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
        
        cap.release()
        cv2.destroyAllWindows()