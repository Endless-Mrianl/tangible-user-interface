#************************it is the final one for nxm matric no hardware**************************** 
#************************it is the final one for nxm matric no hardware****************************
#************************it is the final one for nxm matric no hardware****************************
#************************it is the final one for nxm matric no hardware****************************
import pygame
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (800, 600)
BLOCK_SIZE = 60
PADDING = 20
NUM_BLOCKS = 3  # You can change this value

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Servo Angle Controller")

# Initialize servo angles array
servo_angles = np.zeros((NUM_BLOCKS, NUM_BLOCKS))

def draw_grid():
    screen.fill(WHITE)
    
    # Calculate grid position to center it
    grid_width = NUM_BLOCKS * (BLOCK_SIZE + PADDING)
    grid_height = NUM_BLOCKS * (BLOCK_SIZE + PADDING)
    start_x = (WINDOW_SIZE[0] - grid_width) // 2
    start_y = (WINDOW_SIZE[1] - grid_height) // 2
    
    # Draw blocks
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            x = start_x + j * (BLOCK_SIZE + PADDING)
            y = start_y + i * (BLOCK_SIZE + PADDING)
            
            # Draw block
            pygame.draw.rect(screen, GRAY, (x, y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, BLOCK_SIZE, BLOCK_SIZE), 2)
            
            # Draw angle text
            font = pygame.font.Font(None, 24)
            angle_text = font.render(f"{int(servo_angles[i][j])}°", True, BLACK)
            text_rect = angle_text.get_rect(center=(x + BLOCK_SIZE//2, y + BLOCK_SIZE//2))
            screen.blit(angle_text, text_rect)


