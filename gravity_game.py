import pygame
import math
import sys

# Testmodus-Konfiguration
TEST_MODE = True
TEST_WINDOW_SIZE = (1200, 800)
TEST_TIMEOUT_SECONDS = 10

# Konstanten
SPACING = 25
GRAVITY = 0.8
BG_COLOR = (5, 5, 10)
LASER_COLOR = (255, 0, 0)
CUT_RADIUS = 30
STIFFNESS = 5
FRICTION = 0.99


class Point:
    def __init__(self, x, y, pinned=False):
        self.x = x
        self.y = y
        self.old_x = x
        self.old_y = y
        self.pinned = pinned


class Stick:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.length = math.hypot(p2.x - p1.x, p2.y - p1.y)
        self.active = True
        self.base_color = pygame.Color(200, 200, 200)


def main():
    pygame.init()

    if TEST_MODE:
        screen = pygame.display.set_mode(TEST_WINDOW_SIZE)
        WIDTH, HEIGHT = TEST_WINDOW_SIZE
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        WIDTH, HEIGHT = screen.get_size()

    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    # Berechne Gittergröße basierend auf Bildschirmgröße
    CLOTH_WIDTH = WIDTH // SPACING
    CLOTH_HEIGHT = HEIGHT // SPACING

    def create_cloth():
        points = []
        sticks = []
        start_x = (WIDTH - CLOTH_WIDTH * SPACING) // 2
        start_y = 50

        for y in range(CLOTH_HEIGHT):
            for x in range(CLOTH_WIDTH):
                pinned = (y == 0)
                if pinned and x % 5 != 0 and x != 0 and x != CLOTH_WIDTH - 1:
                    pinned = False
                p = Point(start_x + x * SPACING, start_y + y * SPACING, pinned)
                points.append(p)

        for y in range(CLOTH_HEIGHT):
            for x in range(CLOTH_WIDTH):
                i = y * CLOTH_WIDTH + x
                if x < CLOTH_WIDTH - 1:
                    sticks.append(Stick(points[i], points[i + 1]))
                if y < CLOTH_HEIGHT - 1:
                    sticks.append(Stick(points[i], points[i + CLOTH_WIDTH]))

        return points, sticks

    points, sticks = create_cloth()
    running = True
    start_ticks = pygame.time.get_ticks()

    while running:
        screen.fill(BG_COLOR)
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    points, sticks = create_cloth()
                if event.key == pygame.K_r:
                    for p in points:
                        p.pinned = False

        # Timeout für Testmodus
        if TEST_MODE:
            seconds = (pygame.time.get_ticks() - start_ticks) / 1000
            if seconds > TEST_TIMEOUT_SECONDS:
                running = False

        # Physik-Update
        for p in points:
            if not p.pinned:
                vx = (p.x - p.old_x) * FRICTION
                vy = (p.y - p.old_y) * FRICTION
                p.old_x = p.x
                p.old_y = p.y
                p.x += vx
                p.y += vy + GRAVITY

                if p.y > HEIGHT:
                    p.y = HEIGHT
                    p.old_y = p.y + vy * 0.1
                if p.x < 0:
                    p.x = 0
                if p.x > WIDTH:
                    p.x = WIDTH

        # Stick-Constraints
        for _ in range(STIFFNESS):
            for s in sticks:
                if not s.active:
                    continue

                dx = s.p2.x - s.p1.x
                dy = s.p2.y - s.p1.y
                dist = math.hypot(dx, dy)

                if dist == 0:
                    continue

                if dist > s.length * 4:
                    s.active = False
                    continue

                diff = (s.length - dist) / dist * 0.5
                offset_x = dx * diff
                offset_y = dy * diff

                if not s.p1.pinned:
                    s.p1.x -= offset_x
                    s.p1.y -= offset_y
                if not s.p2.pinned:
                    s.p2.x += offset_x
                    s.p2.y += offset_y

        # Schneiden mit Maus
        for s in sticks:
            if not s.active:
                continue
            mid_x = (s.p1.x + s.p2.x) / 2
            mid_y = (s.p1.y + s.p2.y) / 2
            dist = math.hypot(mid_x - mx, mid_y - my)
            if dist < CUT_RADIUS:
                s.active = False

        # Zeichnen
        for s in sticks:
            if s.active:
                if s.p1.x < -50 or s.p1.x > WIDTH + 50 or s.p1.y > HEIGHT + 50:
                    continue

                curr_dist = math.hypot(s.p1.x - s.p2.x, s.p1.y - s.p2.y)
                tension = min(1.0, max(0.0, (curr_dist - s.length) / 5.0))
                r = s.base_color.r + (255 - s.base_color.r) * tension
                g = s.base_color.g + (255 - s.base_color.g) * tension
                b = s.base_color.b + (255 - s.base_color.b) * tension

                start_pos = (int(s.p1.x), int(s.p1.y))
                end_pos = (int(s.p2.x), int(s.p2.y))
                pygame.draw.line(screen, (r, g, b), start_pos, end_pos, 2)

        # Cursor zeichnen
        pygame.draw.circle(screen, LASER_COLOR, (mx, my), CUT_RADIUS, 2)
        pygame.draw.circle(screen, (255, 255, 255), (mx, my), CUT_RADIUS - 5, 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
