"""
Primjena Q-learning algoritma za upravljanje autonomnim agentom
u 2D simuliranom skladišnom okruženju

Upravljanje tipkovnicom:
  SPACE     - Pauza / Nastavak simulacije
  G         - Otvori/zatvori live grafove (Matplotlib)
  +/=       - Povećaj brzinu animacije
  -         - Smanji brzinu animacije
  1         - Spora brzina (vidljiva animacija)
  2         - Srednja brzina
  3         - Ultra brzina (vidljiva animacija, više koraka/frame)
  4         - Maksimalna brzina (skrivena simulacija, uči u pozadini)
  R         - Resetiraj i počni iznova
  Q / ESC   - Izlaz
"""

import pygame
import numpy as np
import random
import sys

from config import (
    GRID_W, GRID_H, TILE, WIN_W, WIN_H,
    FPS_SLOW, FPS_MED, FPS_ULTRA,
    EPISODES, MAX_STEPS,
    R_PICKUP, R_GOAL, R_OBSTACLE, R_STEP, R_REVISIT,
    C_BG, ULTRA_STEPS_PER_FRAME
)
from agent import QLearningAgent
from graphics import (
    draw_grid, draw_q_overlay, draw_path, draw_pickup, draw_goal,
    draw_start, draw_agent_with_item, draw_hud, draw_fast_mode_overlay
)
from graphs import toggle_graphs, push_graph_update, close_graphs


# ─── WAREHOUSE LAYOUT ─────────────────────────────────────────────────────────

def build_warehouse():
    """Create 2D warehouse map: 0=walkable space, 1=wall/shelf (obstacle)."""
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    # Create outer boundary walls (perimeter)
    grid[0, :] = 1      # Top wall
    grid[-1, :] = 1     # Bottom wall
    grid[:, 0] = 1      # Left wall
    grid[:, -1] = 1     # Right wall

    # Create shelves (rows of obstacles)
    shelf_rows = [2, 4, 7, 9]  # Which rows have shelves
    shelf_cols = list(range(3, 7)) + list(range(9, 13))  # Columns with shelves
    for r in shelf_rows:
        for c in shelf_cols:
            grid[r, c] = 1  # Mark as wall/shelf

    # Add extra obstacles (support columns, crates)
    extras = [(2,8),(4,8),(7,2),(9,2),(5,14),(6,14),(3,14),(9,14)]
    for (r, c) in extras:
        grid[r, c] = 1  # Mark as obstacle

    return grid


GRID = build_warehouse()

# Početna i ciljna pozicija
START = (1, 1)
GOAL  = (GRID_H - 2, GRID_W - 2)

# Akcije: gore, dolje, lijevo, desno
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
N_ACTIONS = 4
# State = pozicija * 2 faze (0=nema predmet, 1=ima predmet)
N_STATES  = GRID_H * GRID_W * 2


def state_id(r, c, has_item=False):
    """Convert (position, item_status) into a unique state ID for Q-table lookup.
    
    Formula: (r * GRID_W + c) * 2 + has_item
    - First part encodes position as linear index
    - Multiply by 2 to reserve space for item flag
    - Add 0 if no item, 1 if carrying item
    - This gives each unique situation a unique ID.
    """
    return (r * GRID_W + c) * 2 + int(has_item)


def valid(r, c):
    """Check if position (r,c) is walkable: must be in bounds and not a wall/shelf."""
    in_bounds = 0 <= r < GRID_H and 0 <= c < GRID_W  # Within grid
    is_walkable = GRID[r, c] == 0  # 0 = floor, 1 = obstacle
    return in_bounds and is_walkable


def random_pickup_pos():
    """Select a random walkable position for the item to pick up.
    
    Constraints: must be floor (GRID[r,c]==0), not START, not GOAL
    This ensures agent has a true task: navigate to item, then to goal.
    """
    free = [(r, c) for r in range(GRID_H) for c in range(GRID_W)
            if GRID[r, c] == 0 and (r, c) != START and (r, c) != GOAL]  # All valid positions
    return random.choice(free)  # Random selection for variety


# ─── GLAVNA PETLJA ────────────────────────────────────────────────────────────

def run_simulation():
    """Main simulation loop with visual rendering and training."""
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Warehouse RL — Q-Learning Pixel Art Simulacija")

    try:
        font_sm = pygame.font.SysFont("monospace", 14, bold=True)
        font_md = pygame.font.SysFont("monospace", 18, bold=True)
    except Exception:
        font_sm = pygame.font.Font(None, 16)
        font_md = pygame.font.Font(None, 20)

    clock = pygame.time.Clock()
    agent = QLearningAgent(N_STATES, N_ACTIONS)

    # Random item position - regenerated each time simulation starts
    pickup_pos = random_pickup_pos()

    # Reward structure: feedback values for different agent actions/outcomes
    rewards = {
        'goal': R_GOAL,          # +100 for reaching goal with item
        'obstacle': R_OBSTACLE,  # -20 for hitting wall or reaching goal without item
        'step': R_STEP,          # -0.5 per step (encourages efficiency)
        'revisit': R_REVISIT,    # -1 for revisiting same position (encourages exploration)
        'pickup': R_PICKUP,      # +50 for collecting the item
    }

    # ── Main simulation state ──
    speed_mode  = 0        # 0=slow (visualize each step), 1=medium, 2=ultra(multi-step visible), 3=max(background training)
    paused      = False    # User can pause/resume with SPACE key
    tick        = 0        # Frame counter for animation effects
    best_path   = []       # Greedy path from trained Q-values (displayed after training)

    # ── Visual episode tracking: shows what agent is currently doing ──
    vis_r, vis_c = START   # Current position of visualized agent
    vis_path  = [START]    # Path taken by agent in current visualization
    vis_steps = 0          # Step counter for visualization
    vis_done  = False      # Has agent reached goal in current visualization?
    vis_has_item = False   # Does agent carry item in visualization?

    # ── Detailed visualization episode state ──
    vis_state     = state_id(*START, False)  # Encode current state for Q-table lookup
    vis_visited   = set([vis_state])         # Track visited states to penalize loops
    vis_ep_reward = 0.0                      # Accumulate reward during visualization

    def start_new_vis_episode():
        nonlocal vis_r, vis_c, vis_path, vis_steps, vis_done, vis_state, vis_visited, vis_ep_reward, vis_has_item
        vis_r, vis_c = START
        vis_has_item = False
        vis_state = state_id(*START, False)
        vis_visited = set([vis_state])
        vis_path = [START]
        vis_steps = 0
        vis_done = False
        vis_ep_reward = 0.0

    start_new_vis_episode()

    running = True
    while running:
        # ── Process keyboard/quit events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # User closes window
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):  # Q or ESC: quit
                    running = False
                elif k == pygame.K_SPACE:  # SPACE: pause/resume
                    paused = not paused
                elif k == pygame.K_g:  # G: toggle live graph visualization
                    toggle_graphs(agent)
                elif k == pygame.K_r:  # R: reset and start fresh episode
                    # Close graphs on reset
                    close_graphs()
                    agent = QLearningAgent(N_STATES, N_ACTIONS)  # Fresh agent
                    pickup_pos = random_pickup_pos()  # New item location
                    best_path = []
                    start_new_vis_episode()
                    tick = 0
                elif k in (pygame.K_1,):  # 1: switch to slow mode
                    speed_mode = 0
                elif k in (pygame.K_2,):  # 2: switch to medium mode
                    speed_mode = 1
                elif k in (pygame.K_3,):  # 3: switch to ultra mode
                    speed_mode = 2
                elif k in (pygame.K_4,):  # 4: switch to max mode (background training)
                    speed_mode = 3
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):  # +: increase speed
                    speed_mode = min(3, speed_mode + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):  # -: decrease speed
                    speed_mode = max(0, speed_mode - 1)

        if paused:
            # While paused, show the display but don't advance training
            if speed_mode < 2:
                pass  # rendering happens below
            clock.tick(15)  # Slow refresh rate to reduce CPU load
        else:
            # Training is active: run episodes based on speed mode
            if not paused and agent.episode < EPISODES:
                if speed_mode == 3:
                    # MODE 3 - MAX SPEED: Run multiple full episodes in background with no rendering
                    # This is for rapid training without visual overhead
                    batch = 20  # Number of episodes to train per frame
                    for _ in range(batch):
                        if agent.episode < EPISODES:
                            agent.run_episode(pickup_pos, GRID, ACTIONS, state_id, valid, rewards)
                    
                    # Update best path and graphs every 200 episodes
                    if agent.episode % 200 == 0:
                        best_path = agent.get_best_path(pickup_pos, GRID, ACTIONS, state_id, valid)
                        push_graph_update(agent)  # Send to matplotlib graph window
                        
                elif speed_mode == 2:
                    # MODE 2 - ULTRA: Multiple steps per frame with visible animation
                    # Shows agent movement but at accelerated speed (5 steps per frame)
                    for _ in range(ULTRA_STEPS_PER_FRAME):
                        if vis_done or agent.episode >= EPISODES:
                            break
                        
                        # Choose action using epsilon-greedy
                        a = agent.choose_action(vis_state)
                        dr, dc = ACTIONS[a]
                        nr, nc = vis_r + dr, vis_c + dc

                        # Evaluate move validity and determine reward
                        if not valid(nr, nc):
                            reward = R_OBSTACLE  # Hit wall: penalize and don't move
                            ns = vis_state
                            nr, nc = vis_r, vis_c
                        elif (nr, nc) == pickup_pos and not vis_has_item:
                            # Reached item: pick it up
                            vis_has_item = True
                            reward = R_PICKUP
                            ns = state_id(nr, nc, vis_has_item)
                            if ns in vis_visited: 
                                reward += R_REVISIT  # Penalize revisits
                            vis_visited.add(ns)
                        elif (nr, nc) == GOAL and vis_has_item:
                            # Reached goal WITH item: SUCCESS! Episode complete
                            reward = R_GOAL
                            ns = state_id(nr, nc, vis_has_item)
                            vis_done = True
                        elif (nr, nc) == GOAL and not vis_has_item:
                            # Reached goal but no item: penalize (wrong sequence)
                            reward = R_OBSTACLE
                            ns = vis_state
                            nr, nc = vis_r, vis_c
                        else:
                            # Normal step: move to new floor tile
                            reward = R_STEP
                            ns = state_id(nr, nc, vis_has_item)
                            if ns in vis_visited: 
                                reward += R_REVISIT  # Penalize revisits
                            vis_visited.add(ns)

                        # Learn from this transition
                        agent.update(vis_state, a, reward, ns, vis_done)
                        vis_ep_reward += reward
                        vis_r, vis_c = nr, nc
                        vis_state = ns
                        vis_path.append((vis_r, vis_c))  # Track path for visualization
                        vis_steps += 1

                        # Check if episode is complete
                        if vis_done or vis_steps >= MAX_STEPS:
                            # Episode ended: finalize stats and start new one
                            agent.decay_epsilon()
                            agent.episode += 1
                            agent.rewards_history.append(vis_ep_reward)
                            agent.steps_history.append(vis_steps)
                            agent.epsilon_history.append(agent.epsilon)
                            agent.success_history.append(1 if vis_done else 0)
                            agent.recent.append(vis_ep_reward)
                            agent.avg100.append(np.mean(agent.recent))
                            best_path = agent.get_best_path(pickup_pos, GRID, ACTIONS, state_id, valid)
                            push_graph_update(agent)  # Update matplotlib graphs
                            start_new_vis_episode()  # Reset for next episode
                            
                else:
                    # MODE 0-1 - VISUAL: One agent action per frame (slowest, most detailed)
                    # Shows each step of agent movement clearly
                    if not vis_done and agent.episode < EPISODES:
                        a = agent.choose_action(vis_state)
                        dr, dc = ACTIONS[a]
                        nr, nc = vis_r + dr, vis_c + dc

                        if not valid(nr, nc):
                            # Hit obstacle: penalize but stay in place
                            reward = R_OBSTACLE
                            ns = vis_state
                            nr, nc = vis_r, vis_c
                        elif (nr, nc) == pickup_pos and not vis_has_item:
                            # Pick up item
                            vis_has_item = True
                            reward = R_PICKUP
                            ns = state_id(nr, nc, vis_has_item)
                            if ns in vis_visited: 
                                reward += R_REVISIT
                            vis_visited.add(ns)
                        elif (nr, nc) == GOAL and vis_has_item:
                            # Reached goal WITH item: SUCCESS!
                            reward = R_GOAL
                            ns = state_id(nr, nc, vis_has_item)
                            vis_done = True
                        elif (nr, nc) == GOAL and not vis_has_item:
                            # Reached goal WITHOUT item: wrong (penalize)
                            reward = R_OBSTACLE
                            ns = vis_state
                            nr, nc = vis_r, vis_c
                        else:
                            # Normal step: move to new tile
                            reward = R_STEP
                            ns = state_id(nr, nc, vis_has_item)
                            if ns in vis_visited: 
                                reward += R_REVISIT
                            vis_visited.add(ns)

                        # Learn from this transition
                        agent.update(vis_state, a, reward, ns, vis_done)
                        vis_ep_reward += reward
                        vis_r, vis_c = nr, nc
                        vis_state = ns
                        vis_path.append((vis_r, vis_c))  # Record path for visualization
                        vis_steps += 1

                        if vis_done or vis_steps >= MAX_STEPS:
                            # Episode complete: finalize and reset
                            agent.decay_epsilon()
                            agent.episode += 1
                            agent.rewards_history.append(vis_ep_reward)
                            agent.steps_history.append(vis_steps)
                            agent.epsilon_history.append(agent.epsilon)
                            agent.success_history.append(1 if vis_done else 0)
                            agent.recent.append(vis_ep_reward)
                            agent.avg100.append(np.mean(agent.recent))
                            best_path = agent.get_best_path(pickup_pos, GRID, ACTIONS, state_id, valid)
                            push_graph_update(agent)  # Update matplotlib graphs
                            start_new_vis_episode()  # Reset for next episode

        tick += 1  # Increment frame counter for animations

        # Calculate best path at training completion
        if agent.episode >= EPISODES:
            best_path = agent.get_best_path(pickup_pos, GRID, ACTIONS, state_id, valid)

        # ── Rendering: decide what to draw based on speed mode ──
        # Draw if: slow/medium/ultra mode, OR paused, OR training complete
        # (skip rendering in max speed mode during training to save performance)
        if speed_mode < 3 or paused or agent.episode >= EPISODES:
            screen.fill(C_BG)  # Clear screen with background color

            # Draw the warehouse layout (walls, shelves, floor tiles)
            draw_grid(screen, GRID, tick)

            # Draw semi-transparent heatmap of Q-values (shows learned value distribution)
            draw_q_overlay(screen, agent, GRID)

            # Draw the optimal path discovered by the agent
            draw_path(screen, best_path, tick)

            # Draw the item that needs to be picked up (pulses if not collected)
            draw_pickup(screen, pickup_pos[0], pickup_pos[1], tick,
                        collected=vis_has_item and speed_mode < 3)

            # Draw the goal/destination position (marked with star + glow)
            draw_goal(screen, GOAL[1], GOAL[0], tick)
            
            # Draw starting position marker
            draw_start(screen, START[0], START[1])

            # Draw current episode path and agent (only in visualization modes)
            if speed_mode < 3:
                draw_path(screen, vis_path, tick)  # Show current path taken
                draw_agent_with_item(screen, vis_r, vis_c, tick, vis_has_item)  # Show agent sprite

            # Draw HUD: episode count, rewards, epsilon value, speed mode, etc.
            draw_hud(screen, agent, agent.episode, speed_mode, paused, font_sm, font_md)

            # Show overlay message if in max-speed hidden training mode
            if speed_mode == 3 and agent.episode < EPISODES:
                draw_fast_mode_overlay(screen, agent, font_md, font_sm)

            pygame.display.flip()  # Update display with all drawn elements

        # ── Frame rate control ──
        # Adjust FPS based on speed mode to balance visual smoothness vs training speed
        if speed_mode == 0:  # SLOW
            clock.tick(FPS_SLOW)  # 8 FPS
        elif speed_mode == 1:  # MEDIUM
            clock.tick(FPS_MED)  # 30 FPS
        elif speed_mode == 2:  # ULTRA
            clock.tick(FPS_ULTRA)  # 120 FPS
        else:  # MAX (mode 3)
            clock.tick(0)  # No frame rate limit: run as fast as possible
            pygame.event.pump()  # Keep event queue responsive

    # Cleanup on exit
    pygame.quit()
    close_graphs()  # Close any open matplotlib windows
    sys.exit(0)


if __name__ == "__main__":
    run_simulation()
