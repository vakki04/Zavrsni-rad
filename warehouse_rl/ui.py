import pygame

BACKGROUND = (15, 23, 42)  # Dark blue-gray
SURFACE = (30, 41, 59)     # Slightly lighter blue-gray
ACCENT = (59, 130, 246)    # Blue accent
ACCENT_HOVER = (96, 165, 250)  # Lighter blue for hover
TEXT_PRIMARY = (226, 232, 240)  # Light gray text
TEXT_SECONDARY = (148, 163, 184)  # Medium gray text
SUCCESS = (34, 197, 94)    # Green
WARNING = (251, 191, 36)   # Yellow
BORDER = (51, 65, 85)      # Border color

class Button:
    def __init__(self, text, x, y, w, h, font=None, color="#3b82f6", hover_color="#60a5fa"):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font or pygame.font.SysFont("segoeui", 24, bold=True)
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, win):
        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = self.rect.collidepoint(mouse_pos)

        # Button background with rounded corners effect
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(win, color, self.rect, border_radius=8)

        # Border
        pygame.draw.rect(win, "#334155", self.rect, 2, border_radius=8)

        # Text
        text_surf = self.font.render(self.text, True, "#e2e8f0")
        text_rect = text_surf.get_rect(center=self.rect.center)
        win.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False
    

class MapCard:
    def __init__(self, map_key, config, x, y, w, h):
        self.map_key = map_key
        self.config = config
        self.rect = pygame.Rect(x, y, w, h)
        self.is_selected = False
        self.font_title = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_info = pygame.font.SysFont("segoeui", 16)

    def draw(self, win):
        # Card background
        color = ACCENT if self.is_selected else SURFACE
        pygame.draw.rect(win, color, self.rect, border_radius=12)

        # Border
        border_color = ACCENT_HOVER if self.is_selected else BORDER
        pygame.draw.rect(win, border_color, self.rect, 3, border_radius=12)

        # Title
        title_surf = self.font_title.render(self.config["name"], True, TEXT_PRIMARY)
        title_rect = title_surf.get_rect(centerx=self.rect.centerx, top=self.rect.top + 20)
        win.blit(title_surf, title_rect)

        # Size info
        size_text = f"{self.config['grid_h']}×{self.config['grid_w']}"
        size_surf = self.font_info.render(size_text, True, TEXT_SECONDARY)
        size_rect = size_surf.get_rect(centerx=self.rect.centerx, top=title_rect.bottom + 10)
        win.blit(size_surf, size_rect)

        # Difficulty indicator
        if self.map_key == "small":
            diff_text = "Lagana"
            diff_color = SUCCESS
        elif self.map_key == "medium":
            diff_text = "Srednja"
            diff_color = WARNING
        else:
            diff_text = "Teška"
            diff_color = (239, 68, 68)  # Red

        diff_surf = self.font_info.render(diff_text, True, diff_color)
        diff_rect = diff_surf.get_rect(centerx=self.rect.centerx, top=size_rect.bottom + 5)
        win.blit(diff_surf, diff_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False