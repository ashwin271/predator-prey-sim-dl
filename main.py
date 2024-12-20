import pygame as pg
import random

# Initialize Pygame
pg.init()

# Screen and Clock Setup
screen = pg.display.set_mode((1280, 720))
clock = pg.time.Clock()
running = True

# Constants
GRID_SIZE = 40  # Size of each grid cell in pixels
GRID_COLS = 20  # Number of columns
GRID_ROWS = 15  # Number of rows
PLAYER_RADIUS = 20  # Radius of the player
PLAYER_COLOR = "red"
PREY_COLOR = "green"
GRID_COLOR = "gray"
BOUNDARY_COLOR = "white"
BG_COLOR = "purple"
SPEED = 300  # Player speed (in pixels per second)
dt = 0.1  # Time delta for movement updates
MOVE_DELAY = 15  # Controls how often the prey moves (in frames)
frame_count = 0  # To track frames for prey movement

# Calculate grid boundaries
grid_width = GRID_COLS * GRID_SIZE
grid_height = GRID_ROWS * GRID_SIZE
boundary_rect = pg.Rect(
    (screen.get_width() - grid_width) // 2,
    (screen.get_height() - grid_height) // 2,
    grid_width,
    grid_height
)

# Initialize player at the center of the grid
player_pos = pg.Vector2(
    boundary_rect.left + (GRID_COLS // 2) * GRID_SIZE + GRID_SIZE // 2,
    boundary_rect.top + (GRID_ROWS // 2) * GRID_SIZE + GRID_SIZE // 2
)

# Initialize prey at a random grid position (corrected)
prey_pos = pg.Vector2(
    boundary_rect.left + GRID_SIZE * random.randint(0, GRID_COLS - 1) + GRID_SIZE // 2,
    boundary_rect.top + GRID_SIZE * random.randint(0, GRID_ROWS - 1) + GRID_SIZE // 2
)

# Main Loop
while running:
    # Event handling
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    # Clear the screen
    screen.fill(BG_COLOR)

    # Draw grid
    for x in range(GRID_COLS + 1):
        pg.draw.line(
            screen, GRID_COLOR,
            (boundary_rect.left + x * GRID_SIZE, boundary_rect.top),
            (boundary_rect.left + x * GRID_SIZE, boundary_rect.bottom)
        )
    for y in range(GRID_ROWS + 1):
        pg.draw.line(
            screen, GRID_COLOR,
            (boundary_rect.left, boundary_rect.top + y * GRID_SIZE),
            (boundary_rect.right, boundary_rect.top + y * GRID_SIZE)
        )

    # Draw boundary
    pg.draw.rect(screen, BOUNDARY_COLOR, boundary_rect, 2)

    # Move prey randomly every MOVE_DELAY frames
    frame_count += 1
    if frame_count >= MOVE_DELAY:
        frame_count = 0
        possible_moves = []
        
        # Check all possible moves (up, down, left, right)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_x = prey_pos.x + dx * GRID_SIZE
            new_y = prey_pos.y + dy * GRID_SIZE
            
            # Only add moves that stay within boundary
            if (boundary_rect.left + GRID_SIZE // 2 <= new_x <= boundary_rect.right - GRID_SIZE // 2 and
                boundary_rect.top + GRID_SIZE // 2 <= new_y <= boundary_rect.bottom - GRID_SIZE // 2):
                possible_moves.append((new_x, new_y))
        
        # Add current position to possible moves (prey might not move)
        possible_moves.append((prey_pos.x, prey_pos.y))
        
        # Choose and apply random move
        new_pos = random.choice(possible_moves)
        prey_pos.x, prey_pos.y = new_pos

    # Draw prey
    pg.draw.circle(screen, PREY_COLOR, prey_pos, GRID_SIZE // 2)

    # Handle player movement
    keys = pg.key.get_pressed()
    new_pos = pg.Vector2(player_pos)
    if keys[pg.K_w]:
        new_pos.y -= GRID_SIZE
    if keys[pg.K_s]:
        new_pos.y += GRID_SIZE
    if keys[pg.K_a]:
        new_pos.x -= GRID_SIZE
    if keys[pg.K_d]:
        new_pos.x += GRID_SIZE

    # Check boundaries
    if boundary_rect.collidepoint(new_pos.x, new_pos.y):
        player_pos = new_pos

    # Draw player
    pg.draw.circle(screen, PLAYER_COLOR, player_pos, PLAYER_RADIUS)

    # Check if player caught prey
    if player_pos == prey_pos:
        # Spawn new prey at random position (centered in grid cell)
        prey_pos = pg.Vector2(
            boundary_rect.left + GRID_SIZE * random.randint(0, GRID_COLS - 1) + GRID_SIZE // 2,
            boundary_rect.top + GRID_SIZE * random.randint(0, GRID_ROWS - 1) + GRID_SIZE // 2
        )

    # Update display
    pg.display.flip()
    clock.tick(20)

# Quit Pygame
pg.quit()
