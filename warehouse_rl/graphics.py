import pygame
from config import (
    TILE,
    WIN_W,
    C_HUD_BG,
    C_HUD_TEXT,
    C_HUD_ACCENT,
    C_HUD_WARN,
    C_HUD_GOOD,
    EPISODES,
    TILE
)
import numpy as np

floor_img: pygame.Surface
wall_img: pygame.Surface
shelf_img: pygame.Surface
goal_img: pygame.Surface
robot_img: pygame.Surface
robot_item_img: pygame.Surface
pickup_img: pygame.Surface
start_img: pygame.Surface

# ucitaj slike iz assets foldera
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

# pod
def draw_tile_floor(surf, col, row, tick):
    x, y = col * TILE, row * TILE
    surf.blit(floor_img, (x, y))

# rub mape
def draw_tile_wall(surf, col, row):
    x, y = col * TILE, row * TILE
    surf.blit(wall_img, (x, y))

# police
def draw_tile_shelf(surf, col, row, tick):
    x, y = col * TILE, row * TILE
    surf.blit(floor_img, (x, y))
    surf.blit(shelf_img, (x, y))

# krajnja tocka
def draw_goal(surf, col, row, tick):
    x, y = col * TILE, row * TILE
    offset = (goal_img.get_width() - TILE) // 2
    surf.blit(floor_img, (x, y))
    surf.blit(goal_img, (x - offset, y - offset))

# nacrtaj tocke na putanji
def draw_path(surf, path, tick):
    if not path:
        return

    rows = max(p[0] for p in path) + 1
    cols = max(p[1] for p in path) + 1

    for i, (pr, pc) in enumerate(path):

        # preskoci vanjske zidove
        if (pr, pc) == (1, 1) or (pr, pc) == (rows - 2, cols - 2):
            continue

        # sredina tilea
        x = pc * TILE + TILE // 2
        y = pr * TILE + TILE // 2
        col = "#1642a3"
        r = 3
        pygame.draw.circle(surf, col, (x, y), r)

# agent bez paketa
def draw_agent(surf, row, col, tick):
    x, y = col * TILE, row * TILE
    surf.blit(robot_img, (x, y))

# pocetna pozicija
def draw_start(surf, row, col):
    x, y = col * TILE, row * TILE
    # floor background
    surf.blit(floor_img, (x, y))
    surf.blit(start_img, (x, y))

# ako nije pokupljen nacrtaj ga na koordinatama zadanim
def draw_pickup(surf, row, col, tick, collected=False):
    if collected:
        return

    x = col * TILE + (TILE - pickup_img.get_width()) // 2
    y = row * TILE + (TILE - pickup_img.get_height()) // 2

    surf.blit(pickup_img, (x, y))

# nacrtaj agenta ako ima paket ili nema
def draw_agent_with_item(surf, row, col, tick, has_item):
    x, y = col * TILE, row * TILE

    if has_item:
        surf.blit(robot_item_img, (x, y))
    else:
        surf.blit(robot_img, (x, y))



def draw_grid(screen, grid, tick):
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            if grid[row, col] == 1:
                # vanjski zidovi
                if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                    draw_tile_wall(screen, col, row)
                else: # police
                    draw_tile_shelf(screen, col, row, tick)
            else: # pod
                draw_tile_floor(screen, col, row, tick)


def draw_q_overlay(screen, agent, grid):
    if agent.episode <= 50:
        return
    # dimenzije mape
    rows, cols = grid.shape
    # nadi najvecu q vrijednost akcije za svako stanje
    q_all = np.max(agent.Q, axis=1)
    # spoji stanja sa i bez paketa (uzima se onaj koji ima vecu vrijednost)
    q_pos = np.maximum(q_all[0::2], q_all[1::2])
    # pretvori nazad u oblik 2d mape
    q_max = q_pos.reshape(rows, cols)
    # nadi najmanju i najvecu q vrijednost na slobodnim poljima
    qmin = q_max[grid == 0].min() if (grid == 0).any() else 0
    qmax_v = q_max[grid == 0].max() if (grid == 0).any() else 1
    rng = max(1.0, qmax_v - qmin)

    for row in range(rows):
        for col in range(cols):
            # na slobodnim poljima
            if grid[row, col] == 0:
                # omjer udaljenosti trenutne q vrijednosti od minimuma u odnosu na globalni raspon
                v = (q_max[row, col] - qmin) / rng
                # prozirnost boje
                alpha_v = int(v * 100)
                s = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                s.fill((0, int(v * 180), int(v * 80), alpha_v))
                screen.blit(s, (col * TILE, row * TILE))


def draw_hud(surf, agent, episode, speed_mode, paused, font_sm, font_md):
    # osnovne dimenzije hud-a
    hud_h = 80
    hud_y = surf.get_height() - hud_h
    hud_w = surf.get_width()

    # crtanje pozadine i gornje linije
    pygame.draw.rect(surf, C_HUD_BG, (0, hud_y, hud_w, hud_h))
    pygame.draw.line(surf, C_HUD_ACCENT, (0, hud_y), (hud_w, hud_y), 2)

    # prikupljanje statistike agenta
    avg = np.mean(agent.recent) if agent.recent else 0
    suc = sum(agent.success_history[-100:]) if agent.success_history else 0
    eps_pct = int(agent.epsilon * 100)

    # naziv trenutne brzine
    speed_names = {0: "SLOW", 1: "MEDIUM", 2: "ULTRA", 3: "MAX (hidden)"}
    sp_name = speed_names.get(speed_mode)

    # stupci za hud
    columns = [
        [
            (f"Epizoda:  {episode:>4}/{EPISODES}", C_HUD_TEXT),
            (f"Nagrada:  {avg:>7.1f} (avg100)", C_HUD_ACCENT if avg > 0 else C_HUD_WARN),
        ],
        [
            (f"Epsilon:  {eps_pct:>3}%", C_HUD_TEXT),
            (f"Uspjeh:   {suc:>3}/100", C_HUD_GOOD if suc > 50 else C_HUD_WARN),
        ],
        [
            (f"Brzina:  [{sp_name}]", C_HUD_ACCENT),
            (f"{'[PAUZA]' if paused else '[G]graf [R]reset [ESC]izlaz'}", C_HUD_WARN if paused else C_HUD_TEXT),
        ]
    ]

    # petlja za crtanje svih stupaca
    x_positions = [12, 230, 430]
    for x, col_data in zip(x_positions, columns):
        for i, (text, color) in enumerate(col_data):
            surf.blit(font_sm.render(text, True, color), (x, hud_y + 10 + i * 28))

    # pomocna funkcija za crtanje trake (progress bar)
    def draw_bar(y, progress, color, text):
        bar_x = 650
        bar_w = hud_w - bar_x - 10 # sirina trake do ruba ekrana
        
        # siva pozadina trake
        pygame.draw.rect(surf, (40, 40, 60), (bar_x, y, bar_w, 12))
        # popunjeni obojani dio
        pygame.draw.rect(surf, color, (bar_x, y, int(bar_w * progress), 12))
        
        # tekst centriran na traci
        txt_surf = font_sm.render(text, True, C_HUD_TEXT)
        txt_x = bar_x + bar_w // 2 - txt_surf.get_width() // 2
        surf.blit(txt_surf, (txt_x, y - 2))

    # traka napretka epizoda
    prog = min(1.0, episode / EPISODES)
    draw_bar(hud_y + 14, prog, C_HUD_ACCENT, f"{int(prog*100)}%")

    # traka epsilon pada (istrazivanja)
    draw_bar(hud_y + 48, agent.epsilon, C_HUD_WARN, f"ε={agent.epsilon:.3f}")