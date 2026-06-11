import pygame
import numpy as np
import random
import sys

from config import TILE, EPISODES, R_PICKUP, R_GOAL, R_OBSTACLE, R_STEP, R_REVISIT
import config
from agent import QLearningAgent
from graphics import draw_grid, draw_q_overlay, draw_path, draw_pickup, draw_goal, draw_start, draw_agent_with_item, draw_hud
from graphs import show_graphs, close_graphs
from experiment_logger import save_experiment

# napravi grid sa zadanim dimenzijama, na rub stavi granicne zidove
def build_warehouse(map_config):
    grid = np.zeros((config.GRID_H, config.GRID_W), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    # dodaj prepreke prema maps.json
    for r, c in map_config["obstacles"]:
        if 0 <= r < config.GRID_H and 0 <= c < config.GRID_W:
            grid[r, c] = 1
    return grid

GRID: np.ndarray
START: tuple[int, int]
GOAL: tuple[int, int]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 4

# funkcija za preslikavanje stanja u jedinstven id
def state_id(r, c, has_item=False):
    return (r * config.GRID_W + c) * 2 + int(has_item)

# provjeri je li polje r, c zauzeto ili ne
def valid(r, c):
    return 0 <= r < config.GRID_H and 0 <= c < config.GRID_W and GRID[r, c] == 0

# r, c random pozicija gdje ce se nalaziti paket, samo na slobodnim poljima
def random_pickup_pos():
    pos = [(r, c) for r in range(config.GRID_H) for c in range(config.GRID_W) if GRID[r, c] == 0 and (r, c) != START and (r, c) != GOAL]
    return random.choice(pos)

def run_simulation(map_config):
    global GRID, START, GOAL

    config.apply_map_config(map_config)
    close_graphs()

    GRID = build_warehouse(map_config)
    START = tuple(map_config["start"])
    GOAL = tuple(map_config["goal"])
    N_STATES = config.GRID_H * config.GRID_W * 2

    pygame.init()
    screen = pygame.display.set_mode((config.GRID_W * TILE, config.GRID_H * TILE + 80))
    pygame.display.set_caption("Warehouse Q-Learning Simulation")
    font_sm = pygame.font.SysFont("monospace", 14, bold=True)
    font_md = pygame.font.SysFont("monospace", 18, bold=True)
    clock = pygame.time.Clock()

    agent = QLearningAgent(N_STATES, N_ACTIONS)
    # pickup_pos = (10, 16)
    pickup_pos = random_pickup_pos()

    rewards = {
        "goal": R_GOAL,
        "obstacle": R_OBSTACLE,
        "step": R_STEP,
        "revisit": R_REVISIT,
        "pickup": R_PICKUP,
    }

    speed_mode = 0
    paused = False
    tick = 0
    best_path = []
    results_saved = False

    FPS_SLOW = 8                   
    FPS_MED = 30                   
    FPS_ULTRA = 120             

    # varijable za vizualizaciju odradenog puta 
    current_path = []
    step_index = 0

    # petlja simulacije
    running = True
    while running:
        # event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif k == pygame.K_SPACE: paused = not paused
                elif k == pygame.K_g: show_graphs(agent, GRID)
                elif k == pygame.K_r:
                    close_graphs()
                    agent = QLearningAgent(N_STATES, N_ACTIONS)
                    pickup_pos = random_pickup_pos()
                    best_path, current_path, step_index, tick = [], [], 0, 0
                    results_saved = False
                elif k == pygame.K_1: speed_mode = 0
                elif k == pygame.K_2: speed_mode = 1
                elif k == pygame.K_3: speed_mode = 2
                elif k == pygame.K_4: speed_mode = 3

        # ako nije pauzirano i ima jos epizoda za odraditi
        if not paused and agent.episode < EPISODES:
            # ako je pozadinski mod
            if speed_mode == 3:
                # napravi 20 epizoda po iteraciji petlje (da je sve odjednom nebi mogli mijenjati brzinu dok se dovrsi taj veci broj epizoda)
                for _ in range(20):
                    if agent.episode < EPISODES:
                        agent.run_episode(START, GOAL, pickup_pos, ACTIONS, state_id, valid, rewards)
            else:
                if step_index >= len(current_path):
                    _, _, _, current_path = agent.run_episode(START, GOAL, pickup_pos, ACTIONS, state_id, valid, rewards)
                    step_index = 0

                if step_index < len(current_path):
                    step_index += 1

        tick += 1

        # ako su prosle sve epizode
        if agent.episode >= EPISODES and not results_saved:
            # dohvati najbolji put
            best_path = agent.get_best_path(START, GOAL, pickup_pos, ACTIONS, state_id, valid)
            # spremanje za analizu
            # save_experiment(f"alpha_{config.ALPHA}", agent.rewards_history, agent.success_history, agent.steps_history, agent.epsilon_history, agent.avg100)
            # posalji podatke za crtanje grafova 
            results_saved = True

       # render asseta za sve modove osim pozadinskog
        if speed_mode < 3 or paused or agent.episode >= EPISODES:
            screen.fill("#12121c")
            draw_grid(screen, GRID, tick)
            draw_q_overlay(screen, agent, GRID)
            draw_path(screen, best_path, tick)
            
            # ako postoji put za crtanje i nismo dosli do kraja
            if current_path and step_index > 0 and step_index <= len(current_path):
                # dohvati pozicijiu i status paketa za trenutni korak
                vr, vc, vhi = current_path[step_index - 1]
                # nacrtaj paket (sakrij ako je pokupljen)
                draw_pickup(screen, pickup_pos[0], pickup_pos[1], tick, collected=vhi)
                # nacrtaj agenta (ako ima paket onda sa paketom)
                draw_agent_with_item(screen, vr, vc, tick, vhi)
            else:
                # nacrtaj samo paket na mapi
                draw_pickup(screen, pickup_pos[0], pickup_pos[1], tick, collected=False)

            draw_goal(screen, GOAL[1], GOAL[0], tick)
            draw_start(screen, START[0], START[1])
            draw_hud(screen, agent, agent.episode, speed_mode, paused, font_sm, font_md)
            # osvjezi ekran
            pygame.display.flip()
        else:
            # osvjezi ekran svki 60. tick
            if tick % 60 == 0:
                screen.fill("#12121c")
                draw_hud(screen, agent, agent.episode, speed_mode, paused, font_sm, font_md)
                pygame.display.flip()

        # prilagodi brzinu
        if paused: clock.tick(15)
        elif speed_mode == 0: clock.tick(FPS_SLOW) 
        elif speed_mode == 1: clock.tick(FPS_MED)  
        elif speed_mode == 2: clock.tick(FPS_ULTRA)
        elif speed_mode == 3: clock.tick(0)     

    pygame.quit()
    close_graphs()
    sys.exit(0)