import pygame
import math
import sys
import numpy as np
from hand_tracker import HandTracker

# Test mode configuration
TEST_MODE = True
TEST_WINDOW_SIZE = (1200, 800)
TEST_TIMEOUT_SECONDS = 60

# Hand tracking settings
USE_HAND_TRACKING = True
SHOW_HAND_OVERLAY = True
HAND_OVERLAY_ALPHA = 0.25

# Constants
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

    # Initialize hand tracker
    hand_tracker = None
    if USE_HAND_TRACKING:
        hand_tracker = HandTracker()
        if not hand_tracker.start_camera():
            print("Warning: Camera could not be started. Falling back to mouse.")
            hand_tracker = None

    # Cursor position and gesture
    cursor_x, cursor_y = WIDTH // 2, HEIGHT // 2
    current_gesture = 'unknown'

    # Calculate grid size based on screen size
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

        # Hand tracking or mouse
        if hand_tracker:
            result = hand_tracker.get_hand_position()
            if result:
                hx, hy, gesture = result
                cursor_x = int(hx * WIDTH)
                cursor_y = int(hy * HEIGHT)
                current_gesture = gesture
        else:
            cursor_x, cursor_y = pygame.mouse.get_pos()
            current_gesture = 'point'  # Mouse always cuts

        mx, my = cursor_x, cursor_y

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

        # Timeout for test mode
        if TEST_MODE:
            seconds = (pygame.time.get_ticks() - start_ticks) / 1000
            if seconds > TEST_TIMEOUT_SECONDS:
                running = False

        # Physics update
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

        # Cut only with 'point' gesture (index finger)
        if current_gesture == 'point':
            for s in sticks:
                if not s.active:
                    continue
                mid_x = (s.p1.x + s.p2.x) / 2
                mid_y = (s.p1.y + s.p2.y) / 2
                dist = math.hypot(mid_x - mx, mid_y - my)
                if dist < CUT_RADIUS:
                    s.active = False

        # Draw cloth
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

        # Draw hand overlay (schematic hand as lines)
        if hand_tracker and SHOW_HAND_OVERLAY:
            overlay_result = hand_tracker.get_hand_overlay(WIDTH, HEIGHT, HAND_OVERLAY_ALPHA)
            if overlay_result:
                hand_lines, hand_points = overlay_result
                
                # Draw lines
                for start_pt, end_pt in hand_lines:
                    pygame.draw.line(screen, (100, 200, 255), start_pt, end_pt, 2)
                
                # Draw joint points
                for pt in hand_points:
                    pygame.draw.circle(screen, (150, 230, 255), pt, 4)

        # Draw cursor (color based on gesture)
        if current_gesture == 'point':
            cursor_color = (255, 0, 0)  # Red = Cutting
        elif current_gesture == 'fist':
            cursor_color = (0, 100, 255)  # Blue = Pushing
        else:
            cursor_color = (255, 255, 0)  # Yellow = Neutral

        pygame.draw.circle(screen, cursor_color, (mx, my), CUT_RADIUS, 2)
        pygame.draw.circle(screen, (255, 255, 255), (mx, my), CUT_RADIUS - 5, 1)

        # Gesture display
        font = pygame.font.Font(None, 36)
        gesture_text = font.render(f"Gesture: {current_gesture}", True, cursor_color)
        screen.blit(gesture_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    if hand_tracker:
        hand_tracker.stop_camera()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
