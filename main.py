# main.py

import pygame as pg
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import multiprocessing as mp
from queue import Empty

# Ensure that the 'spawn' start method is used for compatibility (especially on Windows)
mp.set_start_method('spawn', force=True)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network Class
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

# Function to initialize random prey position
def get_random_prey_pos(boundary_rect, player_pos):
    while True:
        pos = pg.Vector2(
            boundary_rect.left + GRID_SIZE * random.randint(0, GRID_COLS - 1) + GRID_SIZE // 2,
            boundary_rect.top + GRID_SIZE * random.randint(0, GRID_ROWS - 1) + GRID_SIZE // 2
        )
        if pos != player_pos:
            return pos

# Training Loop Function to run in a separate process
def training_loop(experience_queue, model_queue, device, MEMORY_SIZE, TRAINING_BATCH_SIZE, GAMMA):
    # Initialize training model
    training_model = PredatorNet().to(device)
    training_optimizer = optim.Adam(training_model.parameters(), lr=0.001)
    local_memory = []
    
    while True:
        try:
            # Collect experiences from the queue
            experiences = experience_queue.get(timeout=1)  # Wait for experiences
            if experiences == 'END':
                break  # Exit signal
            local_memory.extend(experiences)
            
            # Trim memory if too large
            if len(local_memory) > MEMORY_SIZE:
                local_memory = local_memory[-MEMORY_SIZE:]
            
            # Perform batch training if enough samples
            while len(local_memory) >= TRAINING_BATCH_SIZE:
                batch = random.sample(local_memory, TRAINING_BATCH_SIZE)
                states, actions, rewards, next_states = zip(*batch)
                
                # Convert to tensors
                states = torch.stack(states).to(device)
                next_states = torch.stack(next_states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                
                # Compute Q values
                current_q_values = training_model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = training_model(next_states).max(1)[0].detach()
                target_q_values = rewards + GAMMA * next_q_values
                
                # Compute loss and update network
                loss_fn = nn.MSELoss()
                loss = loss_fn(current_q_values, target_q_values)
                training_optimizer.zero_grad()
                loss.backward()
                training_optimizer.step()
                
                # Send updated model to display process
                model_queue.put((training_model.state_dict(), loss.item()))
                
                # Remove sampled experiences
                local_memory = local_memory[TRAINING_BATCH_SIZE:]
                
        except Empty:
            continue
        except Exception as e:
            print(f"Training process error: {e}")
            break

if __name__ == "__main__":
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
    prey_pos = get_random_prey_pos(boundary_rect, player_pos)
    
    # State Representation
    def get_state():
        # Calculate relative position and other features
        rel_x = (prey_pos.x - player_pos.x) / grid_width
        rel_y = (prey_pos.y - player_pos.y) / grid_height
        distance = player_pos.distance_to(prey_pos) / np.sqrt(grid_width**2 + grid_height**2)
        angle = np.arctan2(prey_pos.y - player_pos.y, prey_pos.x - player_pos.x) / np.pi
        return torch.FloatTensor([rel_x, rel_y, distance, angle]).to(device)
    
    # Fonts
    font_large = pg.font.Font('freesansbold.ttf', 28)
    font_medium = pg.font.Font('freesansbold.ttf', 22)
    font_small = pg.font.Font('freesansbold.ttf', 18)
    
    # Button Class for UI
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
    
    # Slider Class for UI
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
    
    # UI Panels Setup
    LEFT_PANEL_WIDTH = 300
    RIGHT_PANEL_WIDTH = 300
    MAIN_AREA_WIDTH = SCREEN_WIDTH - LEFT_PANEL_WIDTH - RIGHT_PANEL_WIDTH
    MAIN_AREA_HEIGHT = SCREEN_HEIGHT
    
    left_panel = pg.Rect(0, 0, LEFT_PANEL_WIDTH, SCREEN_HEIGHT)
    right_panel = pg.Rect(LEFT_PANEL_WIDTH + MAIN_AREA_WIDTH, 0, RIGHT_PANEL_WIDTH, SCREEN_HEIGHT)
    
    # Add UI elements for user control
    def draw_left_panel(loss, avg_reward):
        # Create a semi-transparent left panel
        panel_surface = pg.Surface((LEFT_PANEL_WIDTH, SCREEN_HEIGHT), pg.SRCALPHA)
        panel_surface.fill((50, 50, 50, 220))  # Dark gray with some transparency
        screen.blit(panel_surface, (0, 0))
        
        # Draw Sliders
        for slider in sliders.values():
            slider.draw(screen)
        
        # Display Score
        score_surf = font_large.render(f"Score: {score}", True, "white")
        screen.blit(score_surf, (20, 350))
        
        # Display Elapsed Time
        time_surf = font_large.render(f"Time: {int(elapsed_time)}s", True, "white")
        screen.blit(time_surf, (20, 400))
        
        # Display Training Feedback
        if loss is not None:
            loss_surf = font_medium.render(f"Loss: {loss:.4f}", True, "white")
            screen.blit(loss_surf, (20, 450))
        else:
            loss_surf = font_medium.render(f"Loss: N/A", True, "white")
            screen.blit(loss_surf, (20, 450))
        
        reward_surf = font_medium.render(f"Avg Reward: {avg_reward:.2f}", True, "white")
        screen.blit(reward_surf, (20, 480))
    
    def draw_right_panel(current_action):
        # Create semi-transparent right panel
        panel_surface = pg.Surface((RIGHT_PANEL_WIDTH, SCREEN_HEIGHT), pg.SRCALPHA)
        panel_surface.fill((50, 50, 50, 220))
        screen.blit(panel_surface, (LEFT_PANEL_WIDTH + MAIN_AREA_WIDTH, 0))
        
        # Display Predator Action title
        action_label = font_large.render("Predator Action", True, "white")
        screen.blit(action_label, (LEFT_PANEL_WIDTH + MAIN_AREA_WIDTH + 50, 50))
        
        # Center point for the controller layout
        center_x = LEFT_PANEL_WIDTH + MAIN_AREA_WIDTH + RIGHT_PANEL_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        spacing = 80  # Space between buttons
        
        # Draw directional buttons in a controller layout
        directions = {
            "Up": (center_x, center_y - spacing),     # Top
            "Right": (center_x + spacing, center_y),  # Right
            "Down": (center_x, center_y + spacing),   # Bottom
            "Left": (center_x - spacing, center_y)    # Left
        }
        
        for idx, (direction, (x, y)) in enumerate(directions.items()):
            color = "yellow" if idx == current_action else "grey"
            
            # Define arrow points based on direction
            if direction == "Up":
                arrow = [(x, y - 20), (x - 15, y + 10), (x + 15, y + 10)]
            elif direction == "Down":
                arrow = [(x, y + 20), (x - 15, y - 10), (x + 15, y - 10)]
            elif direction == "Left":
                arrow = [(x - 20, y), (x + 10, y - 15), (x + 10, y + 15)]
            elif direction == "Right":
                arrow = [(x + 20, y), (x - 10, y - 15), (x - 10, y + 15)]
                
            # Draw arrow and label
            pg.draw.polygon(screen, color, arrow)
            label_surf = font_medium.render(direction, True, "white")
            
            # Position labels around the arrows
            if direction == "Up":
                label_pos = (x, y - 40)
            elif direction == "Down":
                label_pos = (x, y + 40)
            elif direction == "Left":
                label_pos = (x - 40, y)
            else:  # Right  
                label_pos = (x + 40, y)
                
            label_rect = label_surf.get_rect(center=label_pos)
            screen.blit(label_surf, label_rect)
    
    # Initialize neural network components
    predator_net = PredatorNet().to(device)
    # Initially, we won't set up optimizer or memory in main process
    # Training is handled in a separate process
    # Optimizer and memory variables are removed from main
    
    # Initialize multiprocessing queues
    experience_queue = mp.Queue()
    model_queue = mp.Queue()
    
    # Training constants
    PARALLEL_ENVIRONMENTS = 8  # Number of parallel training environments
    TRAINING_BATCH_SIZE = 512  # Larger batch size for parallel training
    SYNC_INTERVAL = 30  # How often to sync the display model with training model (in frames)
    MEMORY_SIZE = 10000  # Increased memory size
    BATCH_SIZE = 64  # Original batch size for experience collection
    GAMMA = 0.99  # Discount factor
    
    # Start training process
    training_process = mp.Process(target=training_loop, args=(
        experience_queue, model_queue, device, MEMORY_SIZE, TRAINING_BATCH_SIZE, GAMMA))
    training_process.start()
    
    # Initialize local experience buffer
    local_experiences = []
    frame_since_sync = 0
    
    # Initialize training feedback variables
    loss = None
    avg_reward = 0.0
    reward_history = []
    
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
    
        # Draw Grid in the Main Area
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
    
        # Draw Boundary
        pg.draw.rect(screen, BOUNDARY_COLOR, boundary_rect, 2)
    
        # Move Prey randomly every MOVE_DELAY frames
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
            new_pos = pg.Vector2(random.choice(possible_moves))
            prey_pos.x, prey_pos.y = new_pos
    
        # Draw Prey with a border
        pg.draw.circle(screen, "black", (int(prey_pos.x), int(prey_pos.y)), GRID_SIZE // 2 + 3)  # Border
        pg.draw.circle(screen, PREY_COLOR, (int(prey_pos.x), int(prey_pos.y)), GRID_SIZE // 2)
    
        # Neural network movement
        state = get_state()
        
        # Epsilon-greedy action selection (unchanged)
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
                prey_pos = get_random_prey_pos(boundary_rect, player_pos)
            
            next_state = get_state()
        
        # Store experience locally
        local_experiences.append((state.cpu(), action, reward, next_state.cpu()))
        
        # Send experiences to training process periodically
        if len(local_experiences) >= BATCH_SIZE:
            experience_queue.put(local_experiences.copy())
            local_experiences = []
        
        # Update Reward History for Average Reward Calculation
        reward_history.append(reward)
        if len(reward_history) > 100:
            reward_history.pop(0)
        avg_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0
    
        # Update display model periodically
        frame_since_sync += 1
        if frame_since_sync >= SYNC_INTERVAL:
            frame_since_sync = 0
            # Retrieve the latest model state if available
            while not model_queue.empty():
                try:
                    new_state_dict, new_loss = model_queue.get_nowait()
                    predator_net.load_state_dict(new_state_dict)
                    loss = new_loss
                except Empty:
                    break
    
        # Draw Player with a border
        pg.draw.circle(screen, "black", (int(player_pos.x), int(player_pos.y)), PLAYER_RADIUS + 3)  # Border
        pg.draw.circle(screen, PLAYER_COLOR, (int(player_pos.x), int(player_pos.y)), PLAYER_RADIUS)
    
        # Draw UI Panels
        draw_left_panel(loss, avg_reward)
        draw_right_panel(action)
    
        # Update elapsed time
        elapsed_time += 1 / FPS
    
        # Update display
        pg.display.flip()
        clock.tick(FPS)
    
    # Cleanup
    experience_queue.put('END')  # Signal the training process to terminate
    training_process.join()
    pg.quit()
    sys.exit()
