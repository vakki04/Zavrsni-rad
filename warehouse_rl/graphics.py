"""
Graphics rendering functions for warehouse simulation.
"""

import pygame
import math
from config import (
    TILE,
    WIN_W,
    WIN_H,
    C_BG,
    C_FLOOR,
    C_FLOOR2,
    C_WALL,
    C_WALL_TOP,
    C_WALL_SHADE,
    C_SHELF,
    C_SHELF_TOP,
    C_SHELF_BOX,
    C_GOAL,
    C_GOAL_GLOW,
    C_AGENT,
    C_AGENT_EYE,
    C_AGENT_BODY,
    C_PATH,
    C_PATH_OLD,
    C_HUD_BG,
    C_HUD_TEXT,
    C_HUD_ACCENT,
    C_HUD_WARN,
    C_HUD_GOOD,
    C_GRID_LINE,
    EPISODES,
)
import numpy as np

# Asset variables
floor_img: pygame.Surface
wall_img: pygame.Surface
shelf_img: pygame.Surface
goal_img: pygame.Surface
robot_img: pygame.Surface
robot_item_img: pygame.Surface
pickup_img: pygame.Surface
start_img: pygame.Surface


def load_assets():
    global floor_img, wall_img, shelf_img
    global goal_img, robot_img, robot_item_img
    global pickup_img, start_img

    floor_img = pygame.transform.scale(
        pygame.image.load("assets/floor_tile.png").convert_alpha(), (TILE, TILE)
    )

    wall_img = pygame.transform.scale(
        pygame.image.load("assets/wall_tile.png").convert_alpha(), (TILE, TILE)
    )

    shelf_img = pygame.transform.scale(
        pygame.image.load("assets/shelf.png").convert_alpha(), (TILE, TILE)
    )

    goal_img = pygame.transform.scale(
        pygame.image.load("assets/goal.png").convert_alpha(), (TILE, TILE)
    )

    robot_img = pygame.transform.scale(
        pygame.image.load("assets/agent.png").convert_alpha(), (TILE, TILE)
    )

    robot_item_img = pygame.transform.scale(
        pygame.image.load("assets/agent_item.png").convert_alpha(), (TILE, TILE)
    )

    pickup_img = pygame.transform.scale(
        pygame.image.load("assets/pickup.png").convert_alpha(),
        (TILE, TILE),
    )

    start_img = pygame.transform.scale(
        pygame.image.load("assets/start.png").convert_alpha(), (TILE, TILE)
    )


def lerp_color(c1, c2, t):
    """Linear interpolation between two RGB colors for smooth transitions.

    Args:
        c1, c2: RGB tuples (r, g, b)
        t: interpolation parameter 0.0 to 1.0 (0=c1, 1=c2)
    Returns:
        Interpolated RGB color tuple
    """
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_tile_floor(surf, col, row, tick):
    """Draw a floor tile with checkerboard pattern and subtle grid lines.

    Creates a pleasant base surface with alternating colors and grid overlay.
    """
    x, y = col * TILE, row * TILE
    surf.blit(floor_img, (x, y))


def draw_tile_wall(surf, col, row):
    """Draw a wall/obstacle tile with 3D shading and brick texture.

    Creates visual depth with highlights on top edge and shadows on right side.
    """
    x, y = col * TILE, row * TILE
    surf.blit(wall_img, (x, y))


def draw_tile_shelf(surf, col, row, tick):
    x, y = col * TILE, row * TILE

    # floor background
    surf.blit(floor_img, (x, y))

    # shelf on top
    surf.blit(shelf_img, (x, y))


def draw_goal(surf, col, row, tick):
    x, y = col * TILE, row * TILE

    offset = (goal_img.get_width() - TILE) // 2

    # floor background
    surf.blit(floor_img, (x, y))

    surf.blit(goal_img, (x - offset, y - offset))


def draw_path(surf, path, tick):
    """Draw the agent's path history as fading dots."""

    if not path:
        return

    rows = max(p[0] for p in path) + 1
    cols = max(p[1] for p in path) + 1

    for i, (pr, pc) in enumerate(path):

        # skip outer border corners
        if (pr, pc) == (1, 1) or (pr, pc) == (rows - 2, cols - 2):
            continue

        x = pc * TILE + TILE // 2
        y = pr * TILE + TILE // 2

        age = len(path) - i
        alpha = max(0.1, 1.0 - age * 0.06)

        col = lerp_color(C_PATH_OLD, C_PATH, alpha)

        r = max(2, int(5 * alpha))

        pygame.draw.circle(surf, col, (x, y), r)


def draw_agent(surf, row, col, tick):
    """Draw agent sprite (small robot) with bobbing animation.

    Agent has body, head, eyes, antenna, and treads. Bobs up/down for life.
    """
    x, y = col * TILE, row * TILE
    surf.blit(robot_img, (x, y))


def draw_start(surf, row, col):
    """Draw the starting position marker with 'S' label.

    Indicates where the agent begins each episode.
    """
    x, y = col * TILE, row * TILE
    # floor background
    surf.blit(floor_img, (x, y))
    surf.blit(start_img, (x, y))


def draw_pickup(surf, row, col, tick, collected=False):
    if collected:
        return

    x = col * TILE + (TILE - pickup_img.get_width()) // 2
    y = row * TILE + (TILE - pickup_img.get_height()) // 2

    surf.blit(pickup_img, (x, y))


def draw_agent_with_item(surf, row, col, tick, has_item):
    x, y = col * TILE, row * TILE

    if has_item:
        surf.blit(robot_item_img, (x, y))
    else:
        surf.blit(robot_img, (x, y))


def draw_hud(surf, agent, episode, speed_mode, paused, font_sm, font_md):
    """Draw heads-up display."""

    hud_y = surf.get_height() - 80

    pygame.draw.rect(surf, C_HUD_BG, (0, hud_y, WIN_W, 80))
    pygame.draw.line(surf, C_HUD_ACCENT, (0, hud_y), (WIN_W, hud_y), 2)

    eps_pct = int(agent.epsilon * 100)
    total_ep = len(agent.rewards_history)

    avg = np.mean(agent.recent) if agent.recent else 0
    suc = sum(agent.success_history[-100:]) if agent.success_history else 0

    speed_names = {0: "SLOW", 1: "MEDIUM", 2: "ULTRA", 3: "MAX (hidden)"}

    sp_name = speed_names.get(speed_mode, "?")

    col1 = [
        (f"Epizoda:  {total_ep:>4}/{EPISODES}", C_HUD_TEXT),
        (f"Nagrada:  {avg:>7.1f} (avg100)", C_HUD_ACCENT if avg > 0 else C_HUD_WARN),
    ]

    col2 = [
        (f"Epsilon:  {eps_pct:>3}%", C_HUD_TEXT),
        (f"Uspjeh:   {suc:>3}/100", C_HUD_GOOD if suc > 50 else C_HUD_WARN),
    ]

    col3 = [
        (f"Brzina:  [{sp_name}]", C_HUD_ACCENT),
        (
            f"{'[PAUZA]' if paused else '[G]graf [R]reset [ESC]izlaz'}",
            C_HUD_WARN if paused else C_HUD_TEXT,
        ),
    ]

    for i, (text, color) in enumerate(col1):
        surf.blit(font_sm.render(text, True, color), (12, hud_y + 10 + i * 28))

    for i, (text, color) in enumerate(col2):
        surf.blit(font_sm.render(text, True, color), (230, hud_y + 10 + i * 28))

    for i, (text, color) in enumerate(col3):
        surf.blit(font_sm.render(text, True, color), (430, hud_y + 10 + i * 28))

    bar_x, bar_y = 650, hud_y + 14
    bar_w, bar_h = WIN_W - 660, 12

    pygame.draw.rect(surf, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h))

    prog = total_ep / EPISODES

    pygame.draw.rect(surf, C_HUD_ACCENT, (bar_x, bar_y, int(bar_w * prog), bar_h))

    pct_t = font_sm.render(f"{int(prog*100)}%", True, C_HUD_TEXT)

    surf.blit(pct_t, (bar_x + bar_w // 2 - pct_t.get_width() // 2, bar_y - 2))

    bar_y2 = hud_y + 48

    pygame.draw.rect(surf, (40, 40, 60), (bar_x, bar_y2, bar_w, 12))

    pygame.draw.rect(surf, C_HUD_WARN, (bar_x, bar_y2, int(bar_w * agent.epsilon), 12))

    e_t = font_sm.render(f"ε={agent.epsilon:.3f}", True, C_HUD_TEXT)

    surf.blit(e_t, (bar_x + bar_w // 2 - e_t.get_width() // 2, bar_y2 - 2))


def draw_grid(screen, grid, tick):
    """Draw warehouse grid."""

    rows, cols = grid.shape

    for row in range(rows):
        for col in range(cols):

            if grid[row, col] == 1:

                # outer border walls
                if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                    draw_tile_wall(screen, col, row)
                else:
                    draw_tile_shelf(screen, col, row, tick)

            else:
                draw_tile_floor(screen, col, row, tick)


def draw_q_overlay(screen, agent, grid):
    """Draw Q-value heatmap overlay."""

    if agent.episode <= 50:
        return

    rows, cols = grid.shape

    q_all = np.max(agent.Q, axis=1)

    q_pos = np.maximum(q_all[0::2], q_all[1::2])

    q_max = q_pos.reshape(rows, cols)

    qmin = q_max[grid == 0].min() if (grid == 0).any() else 0
    qmax_v = q_max[grid == 0].max() if (grid == 0).any() else 1

    rng = max(1.0, qmax_v - qmin)

    for row in range(rows):
        for col in range(cols):

            if grid[row, col] == 0:

                v = (q_max[row, col] - qmin) / rng

                alpha_v = int(v * 50)

                s = pygame.Surface((TILE, TILE), pygame.SRCALPHA)

                s.fill((0, int(v * 180), int(v * 80), alpha_v))

                screen.blit(s, (col * TILE, row * TILE))


def draw_fast_mode_overlay(screen, agent, font_md, font_sm):
    """Draw overlay for hidden fast training mode."""

    overlay_h = screen.get_height() - 80

    overlay = pygame.Surface((WIN_W, overlay_h), pygame.SRCALPHA)

    overlay.fill((10, 10, 20, 180))

    screen.blit(overlay, (0, 0))

    msg1 = font_md.render("MAKSIMALNI MOD — simulacija u tijeku...", True, C_HUD_ACCENT)

    msg2 = font_sm.render(
        f"Epizoda {agent.episode}/{EPISODES}  |  Pritisnite [1], [2] ili [3] za vizualni prikaz",
        True,
        C_HUD_TEXT,
    )

    center_y = overlay_h // 2

    screen.blit(msg1, (WIN_W // 2 - msg1.get_width() // 2, center_y - 30))

    screen.blit(msg2, (WIN_W // 2 - msg2.get_width() // 2, center_y + 10))
