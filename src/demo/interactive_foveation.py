"""
pygame renderer for different foveation techniques
"""

import sys

import numpy as np
import pygame
from foveation import *


def main():
    # Initialize Pygame
    pygame.init()

    # Set up display
    screen_height = 600
    screen_width = 600*2
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Foveated Vision Visualization")

    # Load original_image 
    original_image = pygame.image.load("abbeyroad.jpg")
    original_image = pygame.transform.scale(original_image, (screen_width/2,screen_height))

    clock = pygame.time.Clock()

    # Event Loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Track the cursor
        cursor_x, cursor_y = pygame.mouse.get_pos()

        # Create the Foveated Image
        nx = 70
        ny = 70
        fovea_width = 100
        fovea_height = 100
        #points_x, points_y = sample_geometric_sequence(cursor_x, cursor_y, nx, ny, fovea_width, fovea_height)
        points_x, points_y = sample_trapezoidal(cursor_x, cursor_y, nx, ny, fovea_width, fovea_height)
        all_points_x = points_x
        all_points_y = points_y

        # base_points_x, base_points_y = sample_uniform(screen_width/4, screen_height/2, nx, ny, screen_width/2, screen_height)
        # all_points_x = np.sort(np.concatenate((base_points_x, points_x)))
        # all_points_y = np.sort(np.concatenate((base_points_y, points_y)))
        # for px in base_points_x:
        #     for py in base_points_y:
        #         pygame.draw.circle(screen, (0,0,255), (px,py), 3)

        img_array = pygame.surfarray.array3d(original_image)
        img_array = direct_foveation(all_points_x, all_points_y, img_array)
        foveated_image = pygame.surfarray.make_surface(img_array)
        
        # Draw original_image onto the left, and foveated_image onto the right
        screen.blit(original_image, (0, 0))
        screen.blit(foveated_image, (screen_width/2, 0))

        # Draw the sampled foveation points
        for px in points_x:
            for py in points_y:
                pygame.draw.circle(screen, (255,0,0), (px,py), 1)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
