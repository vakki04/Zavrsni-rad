import pygame
import json
import sys

from ui import Button, MapCard
# from simulation import run_simulation
import simulation

from graphics import load_assets

# --- CONFIG ---
WIDTH, HEIGHT = 1000, 700
BACKGROUND = "#0f172a"
TEXT_PRIMARY = "#e2e8f0"


def load_maps():
    with open("maps.json", "r") as f:
        return json.load(f)


def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q learning simulation - Main Menu")
    load_assets()

    clock = pygame.time.Clock()

    # fonts
    title_font = pygame.font.SysFont("Arial", 36, bold=True)

    # --- LOAD MAPS ---
    maps = load_maps()

    # --- BUTTONS (INIT ONCE!) ---
    btn_start = Button("Pokreni", WIDTH // 2 - 100, HEIGHT - 120, 200, 50)

    # --- MAP CARDS ---
    cards = []
    start_x = 20
    gap = 330

    for i, (key, config) in enumerate(maps.items()):
        card = MapCard(key, config, start_x + i * gap, 200, 300, 150)
        cards.append(card)

    selected_map = None

    running = True
    while running:
        clock.tick(60)
        win.fill(BACKGROUND)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- MAP SELECT ---
            for card in cards:
                if card.is_clicked(event):
                    selected_map = card.map_key
                    for c in cards:
                        c.is_selected = False
                    card.is_selected = True

            # --- BUTTONS ---
            if btn_start.is_clicked(event):
                if selected_map is not None:
                    simulation.run_simulation()

        # --- DRAW TITLE ---
        title = title_font.render("Q Learning Simulation", True, TEXT_PRIMARY)
        win.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))

        # --- DRAW MAP CARDS ---
        for card in cards:
            card.draw(win)

        # --- DRAW BUTTONS ---
        btn_start.draw(win)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
