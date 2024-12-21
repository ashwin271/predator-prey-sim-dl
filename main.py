import pygame as pg
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
GRID_COLOR = (50, 50, 50)  # Darker gray for grid
GRID_LINE_WIDTH = 1  # Thinner grid lines
BOUNDARY_COLOR = "white"
BG_COLOR = (20, 20, 20)  # Dark background
SPEED = 300  # Player speed (in pixels per second)
dt = 0.1  # Time delta for movement updates
MOVE_DELAY = 10  # Controls how often the prey moves (in frames)
frame_count = 0  # To track frames for prey movement
score = 0  # Initialize score
elapsed_time = 0  # Initialize elapsed time
FPS = 20

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
while True:
    prey_pos = pg.Vector2(
        boundary_rect.left + GRID_SIZE * random.randint(0, GRID_COLS - 1) + GRID_SIZE // 2,
        boundary_rect.top + GRID_SIZE * random.randint(0, GRID_ROWS - 1) + GRID_SIZE // 2
    )
    if prey_pos != player_pos:  # Avoid overlap
        break

# Add Neural Network class
class PredatorNet(nn.Module):
    def __init__(self):
        super(PredatorNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),  # Input: relative x, y, distance, angle
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)   # Output: up, down, left, right
        )
    
    def forward(self, x):
        return self.network(x)

# Add these after the pygame initialization
predator_net = PredatorNet()
optimizer = optim.Adam(predator_net.parameters(), lr=0.001)
memory = []  # For experience replay
GAMMA = 0.99  # Discount factor
EPSILON = 0.3  # For exploration

# Replace the player movement code with this
def get_state():
    # Calculate relative position and other features
    rel_x = (prey_pos.x - player_pos.x) / grid_width
    rel_y = (prey_pos.y - player_pos.y) / grid_height
    distance = player_pos.distance_to(prey_pos) / np.sqrt(grid_width**2 + grid_height**2)
    angle = np.arctan2(prey_pos.y - player_pos.y, prey_pos.x - player_pos.x) / np.pi
    return torch.FloatTensor([rel_x, rel_y, distance, angle])

# Use a better font
font = pg.font.Font('freesansbold.ttf', 32)  # Slightly smaller font

# Main Loop
while running:
    # Event handling
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    # Clear screen with background color
    screen.fill(BG_COLOR)

    # Draw grid
    for x in range(GRID_COLS + 1):
        pg.draw.line(
            screen, GRID_COLOR,
            (boundary_rect.left + x * GRID_SIZE, boundary_rect.top),
            (boundary_rect.left + x * GRID_SIZE, boundary_rect.bottom),
            GRID_LINE_WIDTH
        )
    for y in range(GRID_ROWS + 1):
        pg.draw.line(
            screen, GRID_COLOR,
            (boundary_rect.left, boundary_rect.top + y * GRID_SIZE),
            (boundary_rect.right, boundary_rect.top + y * GRID_SIZE),
            GRID_LINE_WIDTH
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
        
        # Enhance prey movement to move away from player if close
        if player_pos.distance_to(prey_pos) <= GRID_SIZE * 3:  # Within 3 cells
            possible_moves = sorted(
                possible_moves,
                key=lambda move: -pg.Vector2(move).distance_to(player_pos)
            )
        
        # Choose and apply random move
        new_pos = random.choice(possible_moves)
        prey_pos.x, prey_pos.y = new_pos

    # Draw prey with a border
    pg.draw.circle(screen, "black", prey_pos, GRID_SIZE // 2 + 2)  # Border
    pg.draw.circle(screen, PREY_COLOR, prey_pos, GRID_SIZE // 2)

    # Neural network movement
    state = get_state()
    
    # Epsilon-greedy action selection
    if random.random() < EPSILON:
        action = random.randint(0, 3)  # Random action
    else:
        with torch.no_grad():
            q_values = predator_net(state)
            action = torch.argmax(q_values).item()
    
    # Convert action to movement
    new_pos = pg.Vector2(player_pos)
    if action == 0:  # up
        new_pos.y -= GRID_SIZE
    elif action == 1:  # down
        new_pos.y += GRID_SIZE
    elif action == 2:  # left
        new_pos.x -= GRID_SIZE
    elif action == 3:  # right
        new_pos.x += GRID_SIZE
    
    # Calculate reward
    old_distance = player_pos.distance_to(prey_pos)
    new_distance = new_pos.distance_to(prey_pos)
    reward = (old_distance - new_distance) / GRID_SIZE  # Positive if getting closer
    
    if player_pos == prey_pos:
        reward = 10.0  # Big reward for catching prey
        score += 1  # Increment score when prey is caught
        
        # Reset prey position
        while True:
            prey_pos = pg.Vector2(
                boundary_rect.left + GRID_SIZE * random.randint(0, GRID_COLS - 1) + GRID_SIZE // 2,
                boundary_rect.top + GRID_SIZE * random.randint(0, GRID_ROWS - 1) + GRID_SIZE // 2
            )
            if prey_pos != player_pos:  # Avoid overlap
                break
    
    # Store experience
    if boundary_rect.collidepoint(new_pos.x, new_pos.y):
        next_state = get_state()
        memory.append((state, action, reward, next_state))
        player_pos = new_pos
    else:
        reward = -1.0  # Penalty for hitting boundary
        next_state = state
        memory.append((state, action, reward, next_state))

    # Training step (after collecting enough experience)
    if len(memory) >= 32:  # Batch size
        batch = random.sample(memory, 32)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Compute Q values
        current_q_values = predator_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = predator_net(next_states).max(1)[0].detach()
        target_q_values = rewards + GAMMA * next_q_values
        
        # Compute loss and update network
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Draw player with a border
    pg.draw.circle(screen, "black", player_pos, PLAYER_RADIUS + 2)  # Border
    pg.draw.circle(screen, PLAYER_COLOR, player_pos, PLAYER_RADIUS)

    # Create UI panel background
    ui_panel = pg.Surface((200, 120))
    ui_panel.fill((40, 40, 40))
    screen.blit(ui_panel, (10, 10))

    # Display score with improved styling
    score_text = font.render(f"Score: {score}", True, "white")
    screen.blit(score_text, (20, 20))

    # Display elapsed time with improved styling
    elapsed_time += dt
    time_text = font.render(f"Time: {int(elapsed_time)}s", True, "white")
    screen.blit(time_text, (20, 70))

    # Update display
    pg.display.flip()
    clock.tick(FPS)

# Quit Pygame
pg.quit()
