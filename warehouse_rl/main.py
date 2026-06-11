import pygame
import json
import sys

from ui import Button, MapCard
import simulation

from graphics import load_assets

# --- CONFIG ---
WIDTH, HEIGHT = 1000, 700
BACKGROUND = "#0f172a"
TEXT_PRIMARY = "#e2e8f0"

# ucitaaj podatke za odabir mapa
def load_maps():
    with open("maps.json", "r") as f:
        return json.load(f)


def main():
    # inicijalizacija pygame
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q learning simulation - Main Menu")
    load_assets()
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("Arial", 36, bold=True)

    maps = load_maps()
    # boutton za pokretanje simulacije
    btn_start = Button("Pokreni", WIDTH // 2 - 100, HEIGHT - 120, 200, 50)

    # kreiranje kartica za odabir mapa (koristi info od maps.json)
    cards = []
    start_x = 20
    gap = 330
    for i, (key, config) in enumerate(maps.items()):
        card = MapCard(config["name"], config, start_x + i * gap, 200, 300, 150)
        card.map_key = key
        cards.append(card)
    # bool koja je mapa odabrana
    selected_map = None
    # petlja za prikaz izbornika mapa
    running = True
    while running:
        # fps limit je 60, prvo napuni pozadinu, na koju se onda crtaju elementi
        clock.tick(60)
        win.fill(BACKGROUND)

        # odrada eventova (klikovi, zatvaranje prozora, itd.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ako je kartica kliknuta, posatvi je kao odabranu
            for card in cards:
                if card.is_clicked(event):
                    selected_map = card.map_key
                    for c in cards:
                        c.is_selected = False
                    card.is_selected = True

            # -ako je odabrana mapa i klikunut start, pokreni simulaciju s odabranom mapom
            if btn_start.is_clicked(event):
                if selected_map is not None:
                    simulation.run_simulation(maps[selected_map])

        # crtanje titlova i ostalih elemenata na ekran
        title = title_font.render("Q Learning Simulation", True, TEXT_PRIMARY)
        win.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))

        for card in cards:
            card.draw(win)

        btn_start.draw(win)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()