import pygame
import sys

from .constants import *

COLORS = {
    "wall_base": (45, 55, 65), 
    "wall_highlight": (70, 85, 95), 
    "wall_shadow": (20, 25, 30), 
    "floor": (211, 223, 223),
    "floor_grid_lines": (180, 200, 200),
    "ui_bg": (20, 24, 30),       
    "panel_bg": (35, 40, 50),     
    "panel_border": (60, 70, 85),
    "text": (210, 220, 230),      
    "accent": (40, 150, 255),     
    "danger": (255, 80, 80),
    "vulnerable": (255, 170, 50),
    "high_risk": (255, 50, 50),
    "burning": (255, 100, 0),
    "flammable": (255, 200, 0),
    "burnt": (80, 80, 80),
}

class Checkbox:
    def __init__(self, x, y, label, default=True):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.label = label
        self.active = default
        self.font = pygame.font.SysFont("Arial", 16)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        if self.active:
            pygame.draw.rect(screen, COLORS["accent"], self.rect.inflate(-6, -6))
        
        label_surf = self.font.render(self.label, True, COLORS["text"])
        screen.blit(label_surf, (self.rect.right + 10, self.rect.y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = not self.active
            return True
        return False

class Visualizer:
    def __init__(self, history, max_grid_w=800, max_grid_h=800):
        pygame.init()
        pygame.display.set_caption("Search and Rescue Simulation Visualizer")

        #self.history = history
        self.all_histories = history
        self.ground_truth = history[0]
        self.selected_history_index = 0

        first_grid = self.all_histories[0][0]
        self.rows, self.cols = first_grid.height, first_grid.width

        # Calculate cell size based on max grid dimensions
        self.cell_size = min(max_grid_w // self.cols, max_grid_h // self.rows)
        self.current_step = 0

        # Fonts and dimensions
        self.font = pygame.font.SysFont("Arial", 20)
        self.small_font = pygame.font.SysFont("Arial", 14)
        first_grid = self.all_histories[0][0]
        self.rows, self.cols = first_grid.height, first_grid.width
        self.grid_w, self.grid_h = self.cols * self.cell_size, self.rows * self.cell_size
        self.sidebar_w, self.bottom_h = 200, 80
        
        self.screen = pygame.display.set_mode((self.grid_w + self.sidebar_w, self.grid_h + self.bottom_h))
        
        # pre-rendered wall layer for performance
        self.wall_layer = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
        self.pre_draw_walls()

        # Adaptive ui layout math
        self.sidebar_x = self.grid_w + 10
        padding = 15
        header_h = 45
        row_h = 30

        # Perspective panel (Top)
        self.selector_y = 10
        btn_h = 25
        btn_margin = 5
        header_h = 45
        padding = 15

        num_items = len(self.all_histories)
        num_rows = ((num_items - 1) // 3) + 1

        self.selector_h = header_h + (num_rows * (btn_h + btn_margin)) + padding

        self.layers_y = self.selector_y + self.selector_h + 15

        # Initialize checkboxes
        self.checkboxes = [
            Checkbox(self.grid_w + 20, 50, "Traversability", True),
            Checkbox(self.grid_w + 20, 80, "Victims", True),
            Checkbox(self.grid_w + 20, 110, "Agents", True),
            Checkbox(self.grid_w + 20, 140, "Vulnerability", True),
            Checkbox(self.grid_w + 20, 170, "Fire", True),
            Checkbox(self.grid_w + 20, 200, "Confidence", False),
        ]
        self.layers_panel_h = (len(self.checkboxes) * row_h) + header_h

        # Position the checkboxes and calculate the Layers panel height
        for i, cb in enumerate(self.checkboxes):
           cb.rect.y = self.layers_y + header_h + (i * row_h)

        # Mission stats pannel
        self.stats_y = self.layers_y + self.layers_panel_h + padding
        self.stats_panel_h = 130
        
        self.layers_panel_h = (len(self.checkboxes) * row_h) + header_h

        # Interactive elements
        self.play_button_rect = pygame.Rect(self.sidebar_x + 60, self.grid_h + 25, 70, 35)
        self.slider_rect = pygame.Rect(50, self.grid_h + 45, self.grid_w - 100, 8)
        self.is_dragging_slider = False
        self.is_playing = False
        self.play_speed = 100 
        self.last_update_time = pygame.time.get_ticks()

        # Assets and pre-rendering
        self.load_assets()
        self.generate_static_background()

    def load_assets(self):
        self.use_sprites = True
        try:
            self.agent_sprite = pygame.image.load("assets/robot.png").convert_alpha()
            self.agent_sprite = pygame.transform.scale(self.agent_sprite, (self.cell_size, self.cell_size))
            
            self.victim_sprite = pygame.image.load("assets/victim_bloody.png").convert_alpha()
            self.victim_sprite = pygame.transform.scale(self.victim_sprite, (self.cell_size, self.cell_size))
        except:
            print("Sprites not found in assets folder, falling back to shapes.")
            self.use_sprites = False
        
        self.fire_burning_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        self.fire_burning_surf.fill(COLORS["burning"] + (180,))

        self.fire_flammable_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        self.fire_flammable_surf.fill(COLORS["flammable"] + (180,))

        self.fire_burnt_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        self.fire_burnt_surf.fill(COLORS["burnt"] + (180,))

        self.conf_surfaces = []
        for i in range(11): 
            surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            alpha = int((i / 10.0) * 170)
            surf.fill((250, 180, 0, alpha))
            self.conf_surfaces.append(surf)

        self.vuln_surfaces = {
            VulnerabilityLevel.VULNERABLE: pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA),
            VulnerabilityLevel.HIGH_RISK: pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA),
        }
        self.vuln_surfaces[VulnerabilityLevel.VULNERABLE].fill(COLORS["vulnerable"] + (80,))
        self.vuln_surfaces[VulnerabilityLevel.HIGH_RISK].fill(COLORS["high_risk"] + (80,))

    def generate_static_background(self):
        """ makes grid, sidebar, and footer into one static image"""
        self.full_bg = pygame.Surface((self.grid_w + self.sidebar_w, self.grid_h + self.bottom_h))
        
        self.full_bg.fill(COLORS["floor"])
        for x in range(0, self.grid_w + 1, self.cell_size):
            pygame.draw.line(self.full_bg, COLORS["floor_grid_lines"], (x, 0), (x, self.grid_h))
        for y in range(0, self.grid_h + 1, self.cell_size):
            pygame.draw.line(self.full_bg, COLORS["floor_grid_lines"], (0, y), (self.grid_w, y))
            
        pygame.draw.rect(self.full_bg, COLORS["ui_bg"], (self.grid_w, 0, self.sidebar_w, self.grid_h))
        pygame.draw.rect(self.full_bg, (15, 18, 22), (0, self.grid_h, self.grid_w + self.sidebar_w, self.bottom_h))
        pygame.draw.line(self.full_bg, COLORS["panel_border"], (self.grid_w, 0), (self.grid_w, self.grid_h + self.bottom_h), 1)

    def draw_panel_frame(self, x, y, w, h, title):
        # Draw the main box
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, COLORS["panel_bg"], rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS["panel_border"], rect, 2, border_radius=8)
      
        header_surf = self.small_font.render(title.upper(), True, COLORS["accent"])
        self.screen.blit(header_surf, (x + 10, y + 10))
        pygame.draw.line(self.screen, COLORS["panel_border"], (x + 10, y + 28), (x + w - 10, y + 28))
   
    def pre_draw_walls(self):
            self.wall_layer.fill((0, 0, 0, 0))
            initial_state = self.all_histories[0][0] 
            for y in range(self.rows):
                for x in range(self.cols):
                    if initial_state.traversability.matrix[y][x] == TraversabilityLevel.UNTRAVERSIBLE:
                        self.draw_beveled_wall(self.wall_layer, x, y)

    def draw_beveled_wall(self, surface, x, y):
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(surface, COLORS["wall_base"], rect)
        pygame.draw.line(surface, COLORS["wall_highlight"], rect.topleft, rect.topright, 2)
        pygame.draw.line(surface, COLORS["wall_highlight"], rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(surface, COLORS["wall_shadow"], rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(surface, COLORS["wall_shadow"], rect.topright, rect.bottomright, 2)

    def draw_perspective_selector(self):
        """Draws buttons to switch between Ground Truth and Agents"""
        x, y = self.sidebar_x, self.selector_y
        self.draw_panel_frame(x, y, self.sidebar_w - 20, self.selector_h, "Perspective")
        
        num_perspectives = len(self.all_histories)
        btn_w, btn_h = 40, 25
        
        for i in range(num_perspectives):
            bx = x + 15 + (i % 3) * (btn_w + 5)
            by = y + 40 + (i // 3) * (btn_h + 5)
            btn_rect = pygame.Rect(bx, by, btn_w, btn_h)
            
            color = COLORS["accent"] if self.selected_history_index == i else COLORS["panel_border"]
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=4)
            
            label = "GT" if i == 0 else f"A{i}"
            txt = self.small_font.render(label, True, COLORS["text"])
            self.screen.blit(txt, (btn_rect.centerx - txt.get_width()//2, btn_rect.centery - txt.get_height()//2))

    def handle_selector_click(self, mouse_pos):
        """Checks if the user clicked a perspective button"""
        x, y = self.sidebar_x, self.selector_y
        for i in range(len(self.all_histories)):
            bx = x + 15 + (i % 3) * (45)
            by = y + 40 + (i // 3) * (30)
            if pygame.Rect(bx, by, 40, 25).collidepoint(mouse_pos):
                self.selected_history_index = i

    def render(self):
        #1. base layers
        self.screen.blit(self.full_bg, (0, 0))
        state = self.all_histories[self.selected_history_index][self.current_step]

        show_trav = self.checkboxes[0].active
        show_vict = self.checkboxes[1].active
        show_agnt = self.checkboxes[2].active
        show_vuln = self.checkboxes[3].active
        show_fire = self.checkboxes[4].active
        show_conf = self.checkboxes[5].active

        # show traversability as a separate layer for performance
        if show_trav and self.selected_history_index == 0:
            self.screen.blit(self.wall_layer, (0, 0))
        
        # Single pass rendering loop
        for y in range(self.rows):
            for x in range(self.cols):
                # Calculate coordinates once per cell
                px = x * self.cell_size
                py = y * self.cell_size

                if show_trav and self.selected_history_index != 0:
                    if state.traversability.matrix[y][x] == TraversabilityLevel.UNTRAVERSIBLE:
                        self.draw_beveled_wall(self.screen, x, y)
                
                # Fire
                if show_fire:
                    fire_level = state.fire.matrix[y][x]
                    if fire_level == FireLevel.BURNING:
                        self.screen.blit(self.fire_burning_surf, (px, py))
                    elif fire_level == FireLevel.FLAMMABLE:
                        self.screen.blit(self.fire_flammable_surf, (px, py))
                    elif fire_level == FireLevel.BURNT:
                        self.screen.blit(self.fire_burnt_surf, (px, py))

                # Confidence
                if show_conf:
                    conf_val = state.confidence.matrix[y][x]
                    if conf_val > 0:
                        conf_idx = min(10, max(1, int(conf_val * 10)))  # Map confidence to 0-10
                        self.screen.blit(self.conf_surfaces[conf_idx], (px, py))

                # Victims
                if show_vict and state.victims.matrix[y][x] == VictimPresence.PRESENT:
                    if self.use_sprites:
                        self.screen.blit(self.victim_sprite, (px, py))
                    else:
                        center_x = int(px + self.cell_size / 2)
                        center_y = int(py + self.cell_size / 2)
                        radius = max(1, int(self.cell_size * 0.35))
                        pygame.draw.circle(self.screen, COLORS["danger"], (center_x, center_y), radius)

                # Agents
                if show_agnt and state.agents.matrix[y][x] == AgentPresence.PRESENT:
                    if self.use_sprites:
                        self.screen.blit(self.agent_sprite, (px, py))
                    else:
                        pad = self.cell_size * 0.15
                        size = max(1, self.cell_size * 0.7)
                        pygame.draw.rect(self.screen, COLORS["accent"], (px + pad, py + pad, size, size))

                # Vulnerability
                if show_vuln:
                    vuln_val = state.vulnerability.matrix[y][x]
                    if vuln_val in self.vuln_surfaces:
                        self.screen.blit(self.vuln_surfaces[vuln_val], (px, py))

        
        # 3. SIDEBAR PANELS
        self.draw_perspective_selector()

        self.draw_panel_frame(self.sidebar_x, self.layers_y, self.sidebar_w - 20, self.layers_panel_h, "Layers")
        for cb in self.checkboxes:
           cb.draw(self.screen)

        self.draw_panel_frame(self.sidebar_x, self.stats_y, self.sidebar_w - 20, self.stats_panel_h, "Mission Stats")
        stat_txt = self.small_font.render(f"Step Count: {self.current_step}", True, COLORS["text"])
        self.screen.blit(stat_txt, (self.sidebar_x + 15, self.stats_y + 45))

        # 4. CONTROLS (Footer)
        btn_color = COLORS["accent"] if not self.is_playing else (200, 50, 50)
        pygame.draw.rect(self.screen, btn_color, self.play_button_rect, border_radius=6)
        lbl = "STOP" if self.is_playing else "PLAY"
        btn_text = self.font.render(lbl, True, COLORS["text"])
        self.screen.blit(btn_text, (self.play_button_rect.centerx - btn_text.get_width()//2, self.play_button_rect.centery - btn_text.get_height()//2))

        # Slider
        pygame.draw.rect(self.screen, (60, 60, 70), self.slider_rect, border_radius=4)
        progress = self.current_step / max(1, (len(self.ground_truth) - 1))
        handle_x = self.slider_rect.x + (progress * self.slider_rect.width)

        if self.is_dragging_slider:
            handle_radius = 14
            handle_color = COLORS["accent"]
        else:
            handle_radius = 10
            handle_color = (200, 200, 200)

        pygame.draw.circle(self.screen, handle_color, (int(handle_x), self.slider_rect.centery), handle_radius)
        
    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                # Checkbox events
                for cb in self.checkboxes:
                    cb.handle_event(event)
               
                # Click events
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_selector_click(event.pos)
                    if self.play_button_rect.collidepoint(event.pos):
                       self.is_playing = not self.is_playing
                    
                    slider_hitbox = self.slider_rect.inflate(30, 30)
                    if slider_hitbox.collidepoint(event.pos):
                        self.is_dragging_slider = True
                        
                        # Instantly snap to click position
                        rel_x = max(0, min(event.pos[0] - self.slider_rect.x, self.slider_rect.width))
                        self.current_step = int((rel_x / self.slider_rect.width) * (len(self.ground_truth) - 1))

                # Unclick events
                if event.type == pygame.MOUSEBUTTONUP:
                    self.is_dragging_slider = False

                # Slider events
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                          self.current_step = min(self.current_step + 1, len(self.ground_truth) - 1)
                    elif event.key == pygame.K_LEFT:
                          self.current_step = max(self.current_step - 1, 0)
                      
                if self.is_dragging_slider and event.type == pygame.MOUSEMOTION:
                    rel_x = max(0, min(event.pos[0] - self.slider_rect.x, self.slider_rect.width))
                    self.current_step = int((rel_x / self.slider_rect.width) * (len(self.ground_truth) - 1))
            
            if self.is_playing:
               now = pygame.time.get_ticks()
               if now - self.last_update_time > self.play_speed:
                  if self.current_step < len(self.ground_truth) - 1:
                        self.current_step += 1
                  else:
                        self.is_playing = False
                  self.last_update_time = now

            self.render()
            pygame.display.flip()
            clock.tick(60)
