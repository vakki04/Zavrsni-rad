"""
Graphics rendering functions for warehouse simulation.
"""

import pygame
import math
from config import (
    TILE, GRID_W, GRID_H, WIN_W, WIN_H,
    C_BG, C_FLOOR, C_FLOOR2, C_WALL, C_WALL_TOP, C_WALL_SHADE,
    C_SHELF, C_SHELF_TOP, C_SHELF_BOX, C_GOAL, C_GOAL_GLOW,
    C_AGENT, C_AGENT_EYE, C_AGENT_BODY, C_PATH, C_PATH_OLD,
    C_HUD_BG, C_HUD_TEXT, C_HUD_ACCENT, C_HUD_WARN, C_HUD_GOOD,
    C_GRID_LINE, EPISODES
)
import numpy as np


def lerp_color(c1, c2, t):
    """Linear interpolation between two RGB colors for smooth transitions.
    
    Args:
        c1, c2: RGB tuples (r, g, b)
        t: interpolation parameter 0.0 to 1.0 (0=c1, 1=c2)
    Returns:
        Interpolated RGB color tuple
    """
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))


def draw_tile_floor(surf, col, row, tick):
    """Draw a floor tile with checkerboard pattern and subtle grid lines.
    
    Creates a pleasant base surface with alternating colors and grid overlay.
    """
    x, y = col*TILE, row*TILE
    # Checkerboard pattern: alternate between two floor colors
    color = C_FLOOR if (col+row) % 2 == 0 else C_FLOOR2
    pygame.draw.rect(surf, color, (x, y, TILE, TILE))
    # Grid lines for subtle structure
    pygame.draw.rect(surf, C_GRID_LINE, (x, y, TILE, TILE), 1)


def draw_tile_wall(surf, col, row):
    """Draw a wall/obstacle tile with 3D shading and brick texture.
    
    Creates visual depth with highlights on top edge and shadows on right side.
    """
    x, y = col*TILE, row*TILE
    # Main wall body
    pygame.draw.rect(surf, C_WALL, (x, y, TILE, TILE))
    # Top edge highlight (simulates light source)
    pygame.draw.rect(surf, C_WALL_TOP, (x, y, TILE, 6))
    # Right edge shadow (simulates 3D depth)
    pygame.draw.rect(surf, C_WALL_SHADE, (x+TILE-5, y+6, 5, TILE-6))
    # Brick pattern texture: vertical and horizontal lines
    for bx in range(0, TILE, 12):  # Horizontal brick divisions
        for by in range(8, TILE, 8):  # Vertical patterns
            pygame.draw.line(surf, C_WALL_SHADE, (x+bx, y+by), (x+min(bx+11, TILE-1), y+by), 1)


def draw_tile_shelf(surf, col, row, tick):
    """Draw a shelf/storage unit tile with stacked boxes.
    
    Visually distinguishes shelves from walls to show warehouse structure.
    """
    x, y = col*TILE, row*TILE
    # Shelf base (main platform)
    pygame.draw.rect(surf, C_SHELF, (x+2, y+6, TILE-4, TILE-8))
    # Shelf top (edge highlight)
    pygame.draw.rect(surf, C_SHELF_TOP, (x+2, y+4, TILE-4, 6))
    # Draw stacked boxes on the shelf
    box_w = (TILE-8) // 3  # Divide space for 3 boxes
    for i in range(3):  # Draw 3 boxes per shelf
        bx = x + 4 + i * (box_w + 1)
        by = y + 14
        # Vary box color slightly for visual interest
        col_box = lerp_color(C_SHELF_BOX, (200, 100, 50), (i*0.25))
        pygame.draw.rect(surf, col_box, (bx, by, box_w, TILE-22))  # Box body
        pygame.draw.rect(surf, (220, 160, 80), (bx, by, box_w, 3))  # Box top edge


def draw_goal(surf, col, row, tick):
    """Draw the goal/delivery destination tile with animated pulsing glow.
    
    Shows the target location where agent must deliver the picked-up item.
    Animation pulses to draw attention and provide visual feedback.
    """
    x, y = col*TILE, row*TILE
    # Start with floor tile base
    draw_tile_floor(surf, col, row, tick)
    # Pulsing animation: sin wave creates smooth expansion/contraction
    pulse = 0.5 + 0.5 * math.sin(tick * 0.08)  # Oscillates between 0.0 and 1.0
    glow_r = int(8 + pulse * 8)  # Glow radius expands with pulse
    # Interpolate glow color based on pulse animation
    glow_col = lerp_color(C_GOAL, C_GOAL_GLOW, pulse)
    # Center position
    cx, cy = x + TILE//2, y + TILE//2
    # Draw pulsing glow circle
    pygame.draw.circle(surf, glow_col, (cx, cy), glow_r)
    # Draw upward-pointing arrow marker to indicate delivery destination
    pts = [(cx, cy-14), (cx-8, cy+6), (cx-3, cy+6), (cx-3, cy+14),
           (cx+3, cy+14), (cx+3, cy+6), (cx+8, cy+6)]  # Arrow polygon vertices
    pygame.draw.polygon(surf, C_GOAL_GLOW, pts)  # Filled arrow
    pygame.draw.rect(surf, glow_col, (cx-3, cy-14, 6, 28), 1)  # Arrow outline


def draw_path(surf, path, tick):
    """Draw the agent's path history as fading dots that fade with age.
    
    Newest dots are bright, older dots fade out. Skip start/goal positions.
    """
    for i, (pr, pc) in enumerate(path):
        # Skip drawing at start and goal positions (they have their own markers)
        if (pr, pc) == (1, 1) or (pr, pc) == (GRID_H - 2, GRID_W - 2):
            continue
        x, y = pc*TILE + TILE//2, pr*TILE + TILE//2
        # Calculate age-based fade: older steps fade more (distance from path end)
        age = len(path) - i
        alpha = max(0.1, 1.0 - age * 0.06)  # Fade faster with older dots
        # Interpolate color from old (dark) to new (bright)
        col = lerp_color(C_PATH_OLD, C_PATH, alpha)
        # Radius shrinks with age for visual depth
        r = max(2, int(5 * alpha))
        pygame.draw.circle(surf, col, (x, y), r)


def draw_agent(surf, row, col, tick):
    """Draw agent sprite (small robot) with bobbing animation.
    
    Agent has body, head, eyes, antenna, and treads. Bobs up/down for life.
    """
    x, y = col*TILE + TILE//2, row*TILE + TILE//2
    # Vertical bobbing motion: creates illusion of movement
    body_bob = int(1.5 * math.sin(tick * 0.15))  # Oscillates every ~40 frames
    # Shadow underneath agent
    pygame.draw.ellipse(surf, (20,20,35), (x-12, y+10+body_bob, 24, 8))
    # Left and right treads (wheels)
    pygame.draw.rect(surf, C_AGENT_BODY, (x-14, y+4+body_bob, 8, 10))  # Left tread
    pygame.draw.rect(surf, C_AGENT_BODY, (x+6,  y+4+body_bob, 8, 10))  # Right tread
    # Main body
    pygame.draw.rect(surf, C_AGENT, (x-10, y-10+body_bob, 20, 16))
    # Head
    pygame.draw.rect(surf, C_AGENT, (x-8, y-18+body_bob, 16, 12))
    # Eyes (facing forward)
    pygame.draw.rect(surf, C_AGENT_EYE, (x-6, y-16+body_bob, 5, 5))  # Left eye
    pygame.draw.rect(surf, C_AGENT_EYE, (x+1,  y-16+body_bob, 5, 5))  # Right eye
    # Antenna on top
    pygame.draw.line(surf, C_AGENT_EYE, (x, y-18+body_bob), (x, y-24+body_bob), 2)  # Shaft
    pygame.draw.circle(surf, C_HUD_ACCENT, (x, y-25+body_bob), 3)  # Tip


def draw_start(surf, row, col):
    """Draw the starting position marker with 'S' label.
    
    Indicates where the agent begins each episode.
    """
    x, y = col*TILE, row*TILE
    # Draw blue square to mark starting area
    pygame.draw.rect(surf, (40, 60, 100), (x+4, y+4, TILE-8, TILE-8))
    # Add white 'S' text in center
    font_s = pygame.font.SysFont("monospace", 20, bold=True)
    t = font_s.render("S", True, (100, 160, 255))
    surf.blit(t, (x + TILE//2 - t.get_width()//2, y + TILE//2 - t.get_height()//2))


def draw_pickup(surf, row, col, tick, collected=False):
    """Draw item/package to be picked up."""
    if collected:
        return
    x, y = col*TILE, row*TILE
    draw_tile_floor(surf, col, row, tick)
    pulse = 0.5 + 0.5 * math.sin(tick * 0.11 + 1.2)
    cx, cy = x + TILE//2, y + TILE//2
    # Box / package
    bw = int(18 + pulse * 3)
    bh = int(14 + pulse * 2)
    bx, by = cx - bw//2, cy - bh//2 + 2
    # Shadow
    pygame.draw.ellipse(surf, (15, 15, 25), (cx-12, cy+8, 24, 7))
    # Package body
    box_col  = lerp_color((200, 150, 60), (255, 200, 80), pulse)
    dark_col = lerp_color((140, 100, 30), (180, 130, 40), pulse)
    pygame.draw.rect(surf, box_col,  (bx,    by,    bw,    bh))
    pygame.draw.rect(surf, dark_col, (bx,    by+bh-4, bw, 4))      # bottom
    pygame.draw.rect(surf, dark_col, (bx+bw-4, by,  4,  bh))      # right side
    # Ribbon
    pygame.draw.rect(surf, (220, 80, 80), (cx-1, by, 2, bh))
    pygame.draw.rect(surf, (220, 80, 80), (bx, cy-1, bw, 2))
    # Bow
    pygame.draw.polygon(surf, (255, 100, 100),
        [(cx-5, cy-3), (cx, cy), (cx-5, cy+3)])
    pygame.draw.polygon(surf, (255, 100, 100),
        [(cx+5, cy-3), (cx, cy), (cx+5, cy+3)])


def draw_agent_with_item(surf, row, col, tick, has_item):
    """Draw agent; if carrying item, show it on top."""
    draw_agent(surf, row, col, tick)
    if has_item:
        x, y = col*TILE + TILE//2, row*TILE + TILE//2
        body_bob = int(1.5 * math.sin(tick * 0.15))
        # Small package on agent
        pygame.draw.rect(surf, (220, 170, 60), (x-7, y-28+body_bob, 14, 10))
        pygame.draw.rect(surf, (180, 120, 30), (x-7, y-20+body_bob, 14, 2))
        pygame.draw.rect(surf, (220, 80,  80), (x-1, y-28+body_bob, 2,  10))


def draw_hud(surf, agent, episode, speed_mode, paused, font_sm, font_md):
    """Draw heads-up display (stats, progress bars, controls)."""
    hud_y = GRID_H * TILE
    pygame.draw.rect(surf, C_HUD_BG, (0, hud_y, WIN_W, 80))
    pygame.draw.line(surf, C_HUD_ACCENT, (0, hud_y), (WIN_W, hud_y), 2)

    eps_pct = int(agent.epsilon * 100)
    total_ep = len(agent.rewards_history)
    avg = np.mean(agent.recent) if agent.recent else 0
    suc = sum(agent.success_history[-100:]) if agent.success_history else 0
    speed_names = {0:"SLOW", 1:"MEDIUM", 2:"ULTRA", 3:"MAX (hidden)"}
    sp_name = speed_names.get(speed_mode, "?")

    # Left column
    col1 = [
        (f"Epizoda:  {total_ep:>4}/{EPISODES}", C_HUD_TEXT),
        (f"Nagrada:  {avg:>7.1f} (avg100)", C_HUD_ACCENT if avg > 0 else C_HUD_WARN),
    ]
    # Middle column
    col2 = [
        (f"Epsilon:  {eps_pct:>3}%", C_HUD_TEXT),
        (f"Uspjeh:   {suc:>3}/100", C_HUD_GOOD if suc > 50 else C_HUD_WARN),
    ]
    # Right column
    col3 = [
        (f"Brzina:  [{sp_name}]", C_HUD_ACCENT),
        (f"{'[PAUZA]' if paused else '[G]graf [R]reset [ESC]izlaz'}", C_HUD_WARN if paused else C_HUD_TEXT),
    ]
    
    for i, (text, color) in enumerate(col1):
        surf.blit(font_sm.render(text, True, color), (12, hud_y + 10 + i*28))
    for i, (text, color) in enumerate(col2):
        surf.blit(font_sm.render(text, True, color), (230, hud_y + 10 + i*28))
    for i, (text, color) in enumerate(col3):
        surf.blit(font_sm.render(text, True, color), (430, hud_y + 10 + i*28))

    # Progress bar
    bar_x, bar_y = 650, hud_y + 14
    bar_w, bar_h = WIN_W - 660, 12
    pygame.draw.rect(surf, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h))
    prog = total_ep / EPISODES
    pygame.draw.rect(surf, C_HUD_ACCENT, (bar_x, bar_y, int(bar_w * prog), bar_h))
    pct_t = font_sm.render(f"{int(prog*100)}%", True, C_HUD_TEXT)
    surf.blit(pct_t, (bar_x + bar_w//2 - pct_t.get_width()//2, bar_y - 2))

    # Epsilon bar
    bar_y2 = hud_y + 48
    pygame.draw.rect(surf, (40, 40, 60), (bar_x, bar_y2, bar_w, 12))
    pygame.draw.rect(surf, C_HUD_WARN, (bar_x, bar_y2, int(bar_w * agent.epsilon), 12))
    e_t = font_sm.render(f"ε={agent.epsilon:.3f}", True, C_HUD_TEXT)
    surf.blit(e_t, (bar_x + bar_w//2 - e_t.get_width()//2, bar_y2 - 2))


def draw_grid(screen, grid, tick):
    """Draw the warehouse grid (walls, shelves, floors)."""
    for row in range(GRID_H):
        for col in range(GRID_W):
            if grid[row, col] == 1:
                # Distinguish outer walls from shelves
                if row == 0 or row == GRID_H-1 or col == 0 or col == GRID_W-1:
                    draw_tile_wall(screen, col, row)
                else:
                    draw_tile_shelf(screen, col, row, tick)
            else:
                draw_tile_floor(screen, col, row, tick)


def draw_q_overlay(screen, agent, grid):
    """Draw semi-transparent Q-value heatmap overlay."""
    if agent.episode > 50:
        q_all = np.max(agent.Q, axis=1)
        q_pos = np.maximum(q_all[0::2], q_all[1::2])
        q_max = q_pos.reshape(GRID_H, GRID_W)
        qmin = q_max[grid == 0].min() if (grid == 0).any() else 0
        qmax_v = q_max[grid == 0].max() if (grid == 0).any() else 1
        rng = max(1.0, qmax_v - qmin)
        for row in range(GRID_H):
            for col in range(GRID_W):
                if grid[row, col] == 0:
                    v = (q_max[row, col] - qmin) / rng
                    alpha_v = int(v * 50)
                    s = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                    s.fill((0, int(v*180), int(v*80), alpha_v))
                    screen.blit(s, (col*TILE, row*TILE))


def draw_fast_mode_overlay(screen, agent, font_md, font_sm):
    """Draw overlay for hidden fast training mode."""
    overlay = pygame.Surface((WIN_W, GRID_H*TILE), pygame.SRCALPHA)
    overlay.fill((10, 10, 20, 180))
    screen.blit(overlay, (0, 0))
    msg1 = font_md.render("MAKSIMALNI MOD — simulacija u tijeku...", True, C_HUD_ACCENT)
    msg2 = font_sm.render(f"Epizoda {agent.episode}/{EPISODES}  |  Pritisnite [1], [2] ili [3] za vizualni prikaz", True, C_HUD_TEXT)
    screen.blit(msg1, (WIN_W//2 - msg1.get_width()//2, GRID_H*TILE//2 - 30))
    screen.blit(msg2, (WIN_W//2 - msg2.get_width()//2, GRID_H*TILE//2 + 10))
