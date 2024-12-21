import pygame as pg
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Initialize Pygame
pg.init()

# Screen and Clock Setup
SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 900  # Increased resolution
screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Predator-Prey Neural Network Simulation")
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
DEFAULT_SPEED = 300  # Player speed (in pixels per second)
DEFAULT_EPSILON = 0.3  # For exploration
dt = 0.1  # Time delta for movement updates
MOVE_DELAY = 10  # Controls how often the prey moves (in frames)
frame_count = 0  # To track frames for prey movement
score = 0  # Initialize score
elapsed_time = 0  # Initialize elapsed time
FPS = 30  # Increased FPS for smoother simulation

# Adjustable parameters
SPEED = DEFAULT_SPEED
EPSILON = DEFAULT_EPSILON

# Calculate grid boundaries
grid_width = GRID_COLS * GRID_SIZE
grid_height = GRID_ROWS * GRID_SIZE
boundary_rect = pg.Rect(
    (SCREEN_WIDTH - grid_width) // 2,
    (SCREEN_HEIGHT - grid_height) // 2,
    grid_width,
    grid_height
)

# Initialize player at the center of the grid
player_pos = pg.Vector2(
    boundary_rect.left + (GRID_COLS // 2) * GRID_SIZE + GRID_SIZE // 2,
    boundary_rect.top + (GRID_ROWS // 2) * GRID_SIZE + GRID_SIZE // 2
)

# Initialize prey at a random grid position (corrected)
def get_random_prey_pos():
    while True:
        pos = pg.Vector2(
            boundary_rect.left + GRID_SIZE * random.randint(0, GRID_COLS - 1) + GRID_SIZE // 2,
            boundary_rect.top + GRID_SIZE * random.randint(0, GRID_ROWS - 1) + GRID_SIZE // 2
        )
        if pos != player_pos:
            return pos

prey_pos = get_random_prey_pos()

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

# Initialize neural network components
predator_net = PredatorNet()
optimizer = optim.Adam(predator_net.parameters(), lr=0.001)
memory = []  # For experience replay
GAMMA = 0.99  # Discount factor
MEMORY_SIZE = 10000  # Increased memory size
BATCH_SIZE = 64  # Increased batch size for better learning

# Replace the player movement code with this
def get_state():
    # Calculate relative position and other features
    rel_x = (prey_pos.x - player_pos.x) / grid_width
    rel_y = (prey_pos.y - player_pos.y) / grid_height
    distance = player_pos.distance_to(prey_pos) / np.sqrt(grid_width**2 + grid_height**2)
    angle = np.arctan2(prey_pos.y - player_pos.y, prey_pos.x - player_pos.x) / np.pi
    return torch.FloatTensor([rel_x, rel_y, distance, angle])

# Use a better font
font_large = pg.font.Font('freesansbold.ttf', 28)
font_medium = pg.font.Font('freesansbold.ttf', 22)
font_small = pg.font.Font('freesansbold.ttf', 18)

# Button class for UI
class Button:
    def __init__(self, rect, color, text, font, text_color="white"):
        self.rect = pg.Rect(rect)
        self.color = color
        self.text = text
        self.font = font
        self.text_color = text_color
        self.hovered = False

    def draw(self, surface):
        pg.draw.rect(surface, self.color, self.rect)
        if self.hovered:
            pg.draw.rect(surface, "yellow", self.rect, 3)
        else:
            pg.draw.rect(surface, "black", self.rect, 2)

        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_hovered(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

# Slider class for UI
class Slider:
    def __init__(self, x, y, width, min_val, max_val, step, initial, label, font):
        self.rect = pg.Rect(x, y, width, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = initial
        self.handle_radius = 10
        self.handle_pos = self.get_handle_pos()
        self.dragging = False
        self.label = label
        self.font = font

    def get_handle_pos(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + ratio * self.rect.width), self.rect.centery

    def draw(self, surface):
        # Draw line
        pg.draw.line(surface, "white", (self.rect.x, self.rect.centery),
                     (self.rect.x + self.rect.width, self.rect.centery), 2)
        # Draw handle
        self.handle_pos = self.get_handle_pos()
        pg.draw.circle(surface, "red", self.handle_pos, self.handle_radius)
        # Draw label
        label_surf = self.font.render(f"{self.label}: {self.value}", True, "white")
        surface.blit(label_surf, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        mouse_pos = pg.mouse.get_pos()
        if event.type == pg.MOUSEBUTTONDOWN:
            if pg.Vector2(mouse_pos).distance_to(self.handle_pos) <= self.handle_radius:
                self.dragging = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pg.MOUSEMOTION:
            if self.dragging:
                # Update handle position
                x = max(self.rect.x, min(mouse_pos[0], self.rect.x + self.rect.width))
                ratio = (x - self.rect.x) / self.rect.width
                raw_val = self.min_val + ratio * (self.max_val - self.min_val)
                # Snap to step
                self.value = round(raw_val / self.step) * self.step
                self.value = max(self.min_val, min(self.value, self.max_val))

# Initialize Buttons and Sliders
buttons = {}
sliders = {}

# Buttons (if needed later for discrete actions)
# Example:
# buttons['reset'] = Button((20, 500, 160, 40), (70, 130, 180), "Reset Simulation", font_medium)

# Sliders
sliders['speed'] = Slider(
    x=40,
    y=150,
    width=200,
    min_val=100,
    max_val=600,
    step=10,
    initial=DEFAULT_SPEED,
    label="Speed",
    font=font_small
)

sliders['epsilon'] = Slider(
    x=40,
    y=250,
    width=200,
    min_val=0.0,
    max_val=1.0,
    step=0.05,
    initial=DEFAULT_EPSILON,
    label="Epsilon",
    font=font_small
)

# Add UI elements for user control
def draw_ui_controls():
    # Create a panel for controls
    control_panel = pg.Surface((300, 300))
    control_panel.fill((60, 60, 60))
    control_panel.set_alpha(200)
    screen.blit(control_panel, (20, SCREEN_HEIGHT - 320))
    
    # Draw each slider
    for slider in sliders.values():
        slider.draw(control_panel)
    
    # Instructions
    instructions = [
        "Controls:",
        "Adjust Speed and Epsilon using sliders.",
        "Speed affects the player's movement rate.",
        "Epsilon controls exploration vs. exploitation.",
    ]
    for i, line in enumerate(instructions):
        instr_surf = font_small.render(line, True, "white")
        control_panel.blit(instr_surf, (20, 20 + i * 20))
    
    # Return sliders to handle events
    return sliders

# Display the current action taken by the predator with directional arrows
def display_predator_action(action):
    actions = ["Up", "Down", "Left", "Right"]
    action_text = font_medium.render(f"Action: {actions[action]}", True, "white")
    screen.blit(action_text, (20, SCREEN_HEIGHT - 350))
    
    # Draw directional indicators
    directions = {
        0: (player_pos.x, player_pos.y - PLAYER_RADIUS - 15),
        1: (player_pos.x, player_pos.y + PLAYER_RADIUS + 15),
        2: (player_pos.x - PLAYER_RADIUS - 15, player_pos.y),
        3: (player_pos.x + PLAYER_RADIUS + 15, player_pos.y)
    }
    for idx, pos in directions.items():
        color = "yellow" if idx == action else "grey"
        pg.draw.polygon(screen, color, [
            (pos[0], pos[1]),
            (pos[0] - 10, pos[1] - 10) if idx in [0,2] else (pos[0] -10, pos[1] +10),
            (pos[0] +10, pos[1] -10) if idx in [0,3] else (pos[0] +10, pos[1] +10)
        ])

# Display model training feedback
def display_training_feedback(loss, avg_reward):
    loss_text = font_small.render(f"Loss: {loss:.4f}" if loss is not None else "Loss: N/A", True, "white")
    screen.blit(loss_text, (20, SCREEN_HEIGHT - 300))
    
    reward_text = font_small.render(f"Avg Reward: {avg_reward:.2f}", True, "white")
    screen.blit(reward_text, (20, SCREEN_HEIGHT - 275))

# Main Loop
while running:
    # Event handling
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        # Handle slider events
        for slider in sliders.values():
            slider.handle_event(event)
    
    # Update SPEED and EPSILON from sliders
    SPEED = sliders['speed'].value
    EPSILON = sliders['epsilon'].value

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
    pg.draw.circle(screen, "black", (int(prey_pos.x), int(prey_pos.y)), GRID_SIZE // 2 + 3)  # Border
    pg.draw.circle(screen, PREY_COLOR, (int(prey_pos.x), int(prey_pos.y)), GRID_SIZE // 2)

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
    
    # Ensure new position is within boundaries
    if not boundary_rect.collidepoint(new_pos):
        # Invalid move, assign negative reward
        reward = -1.0
        next_state = state
    else:
        # Calculate reward
        old_distance = player_pos.distance_to(prey_pos)
        new_distance = new_pos.distance_to(prey_pos)
        reward = (old_distance - new_distance) / GRID_SIZE  # Positive if getting closer
        player_pos = new_pos  # Update position only if move is valid
        
        if player_pos.distance_to(prey_pos) < PLAYER_RADIUS:
            reward += 10.0  # Big reward for catching prey
            score += 1  # Increment score when prey is caught
            prey_pos = get_random_prey_pos()
        
        next_state = get_state()
    
    # Store experience
    memory.append((state, action, reward, next_state))
    if len(memory) > MEMORY_SIZE:
        memory.pop(0)  # Maintain memory size

    # Training step (after collecting enough experience)
    loss = None
    if len(memory) >= BATCH_SIZE:
        batch = random.sample(memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Compute Q values
        current_q_values = predator_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = predator_net(next_states).max(1)[0].detach()
        target_q_values = rewards + GAMMA * next_q_values
        
        # Compute loss and update network
        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Draw player with a border
    pg.draw.circle(screen, "black", (int(player_pos.x), int(player_pos.y)), PLAYER_RADIUS + 3)  # Border
    pg.draw.circle(screen, PLAYER_COLOR, (int(player_pos.x), int(player_pos.y)), PLAYER_RADIUS)
    
    # Draw directional indicators
    draw_directions = True  # Toggle if needed
    if draw_directions:
        directions = {
            0: (player_pos.x, player_pos.y - PLAYER_RADIUS - 20),
            1: (player_pos.x, player_pos.y + PLAYER_RADIUS + 20),
            2: (player_pos.x - PLAYER_RADIUS - 20, player_pos.y),
            3: (player_pos.x + PLAYER_RADIUS + 20, player_pos.y)
        }
        for idx, pos in directions.items():
            color = "yellow" if idx == action else "grey"
            if idx == 0:
                arrow = [(pos[0], pos[1]-10), (pos[0]-10, pos[1]+10), (pos[0]+10, pos[1]+10)]
            elif idx == 1:
                arrow = [(pos[0], pos[1]+10), (pos[0]-10, pos[1]-10), (pos[0]+10, pos[1]-10)]
            elif idx == 2:
                arrow = [(pos[0]-10, pos[1]), (pos[0]+10, pos[1]-10), (pos[0]+10, pos[1]+10)]
            elif idx == 3:
                arrow = [(pos[0]+10, pos[1]), (pos[0]-10, pos[1]-10), (pos[0]-10, pos[1]+10)]
            pg.draw.polygon(screen, color, arrow)
    
    # Create UI panel background and draw controls
    draw_ui_controls()
    
    # Display predator action
    display_predator_action(action)
    
    # Calculate average reward (for demonstration, using score over time)
    avg_reward = score / max(elapsed_time, 1)
    
    # Display model training feedback
    display_training_feedback(loss, avg_reward)
    
    # Display score with improved styling
    score_text = font_large.render(f"Score: {score}", True, "white")
    screen.blit(score_text, (20, SCREEN_HEIGHT - 430))
    
    # Display elapsed time with improved styling
    elapsed_time += 1 / FPS
    time_text = font_large.render(f"Time: {int(elapsed_time)}s", True, "white")
    screen.blit(time_text, (20, SCREEN_HEIGHT - 480))
    
    # Update display
    pg.display.flip()
    clock.tick(FPS)

# Quit Pygame
pg.quit()
sys.exit()
