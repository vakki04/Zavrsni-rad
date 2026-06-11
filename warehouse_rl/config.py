"""
Configuration constants for Warehouse RL simulation.
Centralizes all tunable parameters for easy adjustment of game rules, visuals, and learning.
"""

# ─── GRID & DISPLAY ──────────────────────────────────────────────────────────
GRID_W, GRID_H = 16, 12          # Warehouse grid dimensions: 16 columns, 12 rows
TILE = 48                         # Size of each grid cell in pixels (affects visual scale)
WIN_W, WIN_H = GRID_W * TILE, GRID_H * TILE + 80  # Window size: grid + 80px HUD bar at bottom

# ─── FPS MODES ───────────────────────────────────────────────────────────────
FPS_SLOW = 8                   # Mode 0: slow visualization (8 fps) - see each step clearly
FPS_MED = 30                   # Mode 1: medium speed (30 fps) - balance visibility & speed
FPS_ULTRA = 120                # Mode 2: ultra speed (120 fps) - smooth multi-step animation
FPS_FAST = 0                   # Mode 3: max speed (unlimited) - training only, no rendering
ULTRA_STEPS_PER_FRAME = 5      # In ultra mode, execute 5 agent steps per rendered frame

# ─── Q-LEARNING HYPERPARAMETERS ──────────────────────────────────────────────
EPISODES = 1200                # Total training episodes (iterations)
MAX_STEPS = 300                # Maximum steps per episode (prevents infinite loops)
ALPHA   = 0.3                # Learning rate: how much to update Q-values (0.0-1.0)
GAMMA   = 0.95                 # Discount factor: importance of future rewards (0.0-1.0)
EPS_START = 1.0                # Initial epsilon: 100% random exploration
EPS_END   = 0.05               # Final epsilon: 5% exploration, 95% exploitation
EPS_DECAY = 0.997              # Decay rate: multiply epsilon each episode

# ─── REWARDS ─────────────────────────────────────────────────────────────────
# Reward values shape agent behavior: guides learning toward desired outcomes
R_GOAL     =  100.0            # +100: reached goal with item (major success)
R_OBSTACLE = -20.0             # -20: hit wall OR reached goal without item (penalty)
R_STEP     =  -0.5             # -0.5: per step (encourages efficient paths)
R_REVISIT  =  -1.0             # -1: penalty for revisiting same state (encourages exploration)
R_PICKUP   =  50.0             # +50: picked up the item (major milestone)

# ─── COLORS (pixel art RGB palette) ───────────────────────────────────────────
# Background and environment
C_BG         = (18,  18,  28)   # Dark background
C_FLOOR      = (28,  32,  48)   # Floor tile 1
C_FLOOR2     = (32,  36,  54)   # Floor tile 2 (checkerboard)
C_GRID_LINE  = (28,  32,  48)   # Subtle grid overlay

# Obstacles (walls)
C_WALL       = (55,  45,  80)   # Wall body
C_WALL_TOP   = (75,  62, 105)   # Wall top edge (highlight)
C_WALL_SHADE = (38,  30,  58)   # Wall shadow (3D effect)

# Shelves/storage
C_SHELF      = (80,  55,  30)   # Shelf base
C_SHELF_TOP  = (110, 78,  42)   # Shelf top edge
C_SHELF_BOX  = (180, 130,  60)  # Boxes on shelf

# Goal position
C_GOAL       = (50, 200,  80)   # Goal marker (green)
C_GOAL_GLOW  = (80, 255, 120)   # Goal glow (bright green)

# Agent sprite
C_AGENT      = (80, 180, 255)   # Agent body (blue)
C_AGENT_EYE  = (220, 240, 255)  # Agent eyes (bright)
C_AGENT_BODY = (50, 140, 220)   # Agent treads (dark blue)

# Path visualization
C_PATH       = (60, 100, 160)   # Current path (bright)
C_PATH_OLD   = (35,  60, 100)   # Path history (faded)

# HUD elements
C_HUD_BG     = (12,  12,  20)   # HUD background
C_HUD_TEXT   = (180, 200, 255)  # Normal text
C_HUD_ACCENT = (80, 180, 255)   # Highlighted text
C_HUD_WARN   = (255, 160,  60)  # Warning/caution text
C_HUD_GOOD   = (80, 220, 120)   # Success/positive text

# ─── GRAPH VISUALIZATION ─────────────────────────────────────────────────────
GRAPH_UPDATE_INTERVAL = 2000   # Matplotlib graph refresh interval: 2000ms (2 seconds)

def apply_map_config(map_config):
    global GRID_W, GRID_H, WIN_W, WIN_H
    GRID_W = map_config["grid_w"]
    GRID_H = map_config["grid_h"]
    WIN_W = GRID_W * TILE
    WIN_H = GRID_H * TILE + 80
