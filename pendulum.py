import math
import numpy as np
import pygame as pg
from pygame import gfxdraw

def F(S, params):
    theta1, omega1, theta2, omega2 = S
    m1, m2, l1, l2, mu, g = params
    
    delta = theta1 - theta2
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    denominator = 2 * m1 + m2 - m2 * np.cos(2 * delta)
    
    dtheta1 = omega1
    
    numerator1 = (-g * (2 * m1 + m2) * np.sin(theta1) - 
                  m2 * g * np.sin(theta1 - 2 * theta2) - 
                  2 * sin_delta * m2 * (omega2**2 * l2 + omega1**2 * l1 * cos_delta))
    domega1 = numerator1 / (l1 * denominator)
    
    dtheta2 = omega2
    
    numerator2 = (2 * sin_delta * (omega1**2 * l2 * (m1 + m2) + 
                  g * (m1 + m2) * np.cos(theta1) + 
                  omega2**2 * l2 * m2 * cos_delta))
    domega2 = numerator2 / (l2 * denominator)
    
    domega1 -= mu * omega1
    domega2 -= mu * omega2
    
    return np.array([dtheta1, domega1, dtheta2, domega2])

def rk4_double_pendulum(theta1, w1, theta2, w2, l1, l2, m1, m2, mu, g, dt):
    S = np.array([theta1, w1, theta2, w2])
    params = np.array([m1, m2, l1, l2, mu, g])
    K1 = F(S, params)
    K2 = F(S + dt * K1 / 2, params)
    K3 = F(S + dt * K2 / 2, params)
    K4 = F(S + dt * K3, params)
    
    S_next = S + (dt / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
    return S_next[0], S_next[1], S_next[2], S_next[3]
    
WIDTH, HEIGHT = 1400, 800

SLIDER_WIDTH = WIDTH // 2
SLIDER_HEIGHT = 8
SLIDER_HANDLE_RADIUS = 12
SLIDER_MARGIN = SLIDER_HANDLE_RADIUS * 2 + 30

ORIGIN = (WIDTH * 3 // 4, HEIGHT // 3 - SLIDER_MARGIN)
L_SCALE = HEIGHT // 6
PENDULUM_RADIUS = 30
PRECISION = 666

G_MIN, G_MAX = 0.0, 100.0
MU_MIN, MU_MAX = 0.0, 1.0

G = 9.81
MU = 0.01
SIGN = 1

MASS_MIN, MASS_MAX = 0.000000001, 5.0

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREY = (140, 140, 140)
LIGHT_BLUE = (100, 100, 255)

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class doublyLinkedList:
    def __init__(self, max_length):
        self.head = None
        self.tail = None
        self.length = 0
        self.max_length = max_length

    def push_front(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.length += 1
        if self.length > self.max_length:
            self.tail = self.tail.prev
            self.tail.next = None
            self.length -= 1
    
    def clear(self):
        self.head = None
        self.tail = None
        self.length = 0
    
    def __len__(self):
        return self.max_length
            
SCROLL_STEP = 2
records = doublyLinkedList(WIDTH // (2 * SCROLL_STEP))
records_small_pendulum = doublyLinkedList(WIDTH)

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()
font = pg.font.SysFont(None, 28)

dragging = False
slider_drag = None  # None, "g", or "mu"

def get_pendulum_pos(theta, l):
    x = ORIGIN[0] + math.sin(theta) * l * L_SCALE
    y = ORIGIN[1] + math.cos(theta) * l * L_SCALE
    return int(x), int(y)

def display_text(text, pos, color=BLACK):
    """Draw text at the given position."""
    surf = font.render(str(text), True, color)
    screen.blit(surf, pos)

def draw_slider(x, y, value, min_val, max_val, label):
    # Draw slider track
    pg.draw.rect(screen, GREY, (x, y, SLIDER_WIDTH, SLIDER_HEIGHT))
    
    # Calculate handle position
    norm = (value - min_val) / (max_val - min_val)
    handle_x = int(x + norm * SLIDER_WIDTH)
    handle_y = y + SLIDER_HEIGHT // 2
    
    # Draw handle
    gfxdraw.aacircle(screen, handle_x, handle_y, SLIDER_HANDLE_RADIUS, LIGHT_BLUE)
    gfxdraw.filled_circle(screen, handle_x, handle_y, SLIDER_HANDLE_RADIUS, LIGHT_BLUE)
    
    # Draw label
    display_text(f"{label}: {value:.3f}", (x, y - 30))
    return handle_x, handle_y

def draw_graph():
    cur = records.head
    i = 0
    while cur is not None:
        y = int(cur.data)
        if cur.next is not None:
            next_y = int(cur.next.data)
            pg.draw.line(screen, GREY, ((len(records) - i) * SCROLL_STEP, y), ((len(records) - i - 1) * SCROLL_STEP, next_y), 2)
        cur = cur.next
        i += 1
    # x-axis (t)
    pg.draw.line(screen, BLACK, (0, HEIGHT // 2), (len(records) * SCROLL_STEP, HEIGHT // 2), 2)
    
    # y-axis (theta)
    pg.draw.line(screen, BLACK, (50, HEIGHT // 2 + math.pi * L_SCALE * 0.7 + 20), (50, HEIGHT // 2 - math.pi * L_SCALE * 0.7 - 20), 2)
    
    # label
    display_text("θ (rad)", (70, HEIGHT // 2 - math.pi * L_SCALE * 0.7))
    display_text("t (s)", (len(records) * SCROLL_STEP - 50, HEIGHT // 2 + 15))
    display_text("π", (27, HEIGHT // 2 - math.pi * L_SCALE * 0.7 - 10))
    display_text("-π", (20, HEIGHT // 2 + math.pi * L_SCALE * 0.7 - 10))
    pg.draw.line(screen, BLACK, (45, HEIGHT // 2 - math.pi * L_SCALE * 0.7), (55, HEIGHT // 2 - math.pi * L_SCALE * 0.7), 2)
    pg.draw.line(screen, BLACK, (45, HEIGHT // 2 + math.pi * L_SCALE * 0.7), (55, HEIGHT // 2 + math.pi * L_SCALE * 0.7), 2)
    
def draw_path():
    cur = records_small_pendulum.head
    i = 0
    while cur is not None:
        x, y = int(cur.data[0]), int(cur.data[1])
        if cur.next is not None:
            next_x, next_y = int(cur.next.data[0]), int(cur.next.data[1])
            add = (255 - GREY[0]) * i / len(records_small_pendulum) - 1
            pg.draw.line(screen, (GREY[0] + add, GREY[1] + add, GREY[2] + add), (x, y), (next_x, next_y), 2)
        cur = cur.next
        i += 1

def slider_value_from_pos(mouse_x, x, min_val, max_val):
    norm = min(max((mouse_x - x) / SLIDER_WIDTH, 0), 1)
    return min_val + norm * (max_val - min_val)

m1 = 1.0
m2 = 1.0
theta1 = 0.0
theta2 = 0.0
w1 = 0.0
w2 = 0.0
l1 = 1.0
l2 = 1.0
dragging_double = False
drag_double_idx = None  # 0 for first, 1 for second

def get_double_pendulum_positions(theta1, theta2, l1, l2):
    # theta2 is relative to the first arm (swings with the first)
    x1 = ORIGIN[0] + math.sin(theta1) * l1 * L_SCALE
    y1 = ORIGIN[1] + math.cos(theta1) * l1 * L_SCALE
    x2 = x1 + math.sin(theta1 + theta2) * l2 * L_SCALE
    y2 = y1 + math.cos(theta1 + theta2) * l2 * L_SCALE
    return (int(x1), int(y1)), (int(x2), int(y2))

def draw_double_pendulum(theta1, theta2, l1, l2, m1, m2):
    (x1, y1), (x2, y2) = get_double_pendulum_positions(theta1, theta2, l1, l2)
    # Draw arms
    pg.draw.line(screen, BLACK, ORIGIN, (x1, y1), 3)
    pg.draw.line(screen, BLACK, (x1, y1), (x2, y2), 3)
    # Draw masses (radius proportional to mass)
    r1 = int(PENDULUM_RADIUS * (m1 / MASS_MIN) ** 0.3 / (MASS_MAX / MASS_MIN) ** 0.3)
    r2 = int(PENDULUM_RADIUS * (m2 / MASS_MIN) ** 0.3 / (MASS_MAX / MASS_MIN) ** 0.3)
    gfxdraw.aacircle(screen, x1, y1, r1, BLUE)
    gfxdraw.filled_circle(screen, x1, y1, r1, BLUE)
    gfxdraw.aacircle(screen, x2, y2, r2, BLUE)
    gfxdraw.filled_circle(screen, x2, y2, r2, BLUE)
    # Draw origin
    gfxdraw.aacircle(screen, ORIGIN[0], ORIGIN[1], 5, BLACK)
    gfxdraw.filled_circle(screen, ORIGIN[0], ORIGIN[1], 5, BLACK)

running = True
draw_path_enabled = True  # Add a variable to track the checkbox state

def draw_checkbox(x, y, label, checked):
    box_size = 20
    # Draw the checkbox border
    pg.draw.rect(screen, BLACK, (x, y, box_size, box_size), 2)
    # Draw the check mark if checked
    if checked:
        pg.draw.line(screen, BLACK, (x + 4, y + box_size // 2), (x + box_size // 2, y + box_size - 4), 3)
        pg.draw.line(screen, BLACK, (x + box_size // 2, y + box_size - 4), (x + box_size - 4, y + 4), 3)
    # Draw the label next to the checkbox
    display_text(label, (x + box_size + 10, y))

while running:
    dt = clock.tick(60) / 1000
    drag = 0 # 0 for not dragging, 1 for dragging first pendulum, 2 for dragging second pendulum

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            mx, my = pg.mouse.get_pos()
            (x1, y1), (x2, y2) = get_double_pendulum_positions(theta1, theta2, l1, l2)

            if (mx - x2) ** 2 + (my - y2) ** 2 < PENDULUM_RADIUS ** 2:
                dragging_double = True
                drag_double_idx = 1
            elif (mx - x1) ** 2 + (my - y1) ** 2 < PENDULUM_RADIUS ** 2:
                dragging_double = True
                drag_double_idx = 0
            # Check sliders
            gx, gy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 2
            mux, muy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN
            g_handle_x, g_handle_y = draw_slider(gx, gy, G, G_MIN, G_MAX, "g")
            mu_handle_x, mu_handle_y = draw_slider(mux, muy, MU, MU_MIN, MU_MAX, "mu")
            if (mx - g_handle_x) ** 2 + (my - g_handle_y) ** 2 < SLIDER_HANDLE_RADIUS ** 2:
                slider_drag = "g"
            elif (mx - mu_handle_x) ** 2 + (my - mu_handle_y) ** 2 < SLIDER_HANDLE_RADIUS ** 2:
                slider_drag = "mu"
            # Add mass sliders
            mass1x, mass1y = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 4
            mass2x, mass2y = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 3
            mass1_handle_x, mass1_handle_y = draw_slider(mass1x, mass1y, m1, MASS_MIN, MASS_MAX, "mass1")
            mass2_handle_x, mass2_handle_y = draw_slider(mass2x, mass2y, m2, MASS_MIN, MASS_MAX, "mass2")
            if (mx - mass1_handle_x) ** 2 + (my - mass1_handle_y) ** 2 < SLIDER_HANDLE_RADIUS ** 2:
                slider_drag = "mass1"
            elif (mx - mass2_handle_x) ** 2 + (my - mass2_handle_y) ** 2 < SLIDER_HANDLE_RADIUS ** 2:
                slider_drag = "mass2"
            # Add checkbox to toggle the drawing of the path
            checkbox_x, checkbox_y, checkbox_size = WIDTH - 200, HEIGHT - 200, 20
            if checkbox_x <= mx <= checkbox_x + checkbox_size and checkbox_y <= my <= checkbox_y + checkbox_size:
                draw_path_enabled = not draw_path_enabled
        elif event.type == pg.MOUSEBUTTONUP:
            dragging = False
            slider_drag = None
            dragging_double = False
            drag_double_idx = None
        elif event.type == pg.MOUSEMOTION:
            mx, my = event.pos
            if slider_drag:
                mx, my = event.pos
                if slider_drag == "g":
                    gx = WIDTH // 2 - SLIDER_WIDTH // 2
                    G = slider_value_from_pos(mx, gx, G_MIN, G_MAX)
                elif slider_drag == "mu":
                    mux = WIDTH // 2 - SLIDER_WIDTH // 2
                    MU = slider_value_from_pos(mx, mux, MU_MIN, MU_MAX)
                elif slider_drag == "mass1":
                    mass1x = WIDTH // 2 - SLIDER_WIDTH // 2
                    m1 = slider_value_from_pos(mx, mass1x, MASS_MIN, MASS_MAX)
                elif slider_drag == "mass2":
                    mass2x = WIDTH // 2 - SLIDER_WIDTH // 2
                    m2 = slider_value_from_pos(mx, mass2x, MASS_MIN, MASS_MAX)
        elif event.type == pg.KEYDOWN:
            theta1 = 0
            theta2 = 0
            w1 = 0.0
            w2 = 0.0
            l1 = 1.0
            l2 = 1.0
            m1 = 1.0
            m2 = 1.0
            G = 9.81
            MU = 0.01
            records.clear()
            records_small_pendulum.clear()
            dragging = False
            slider_drag = None
            SIGN = 1

    prev_theta1 = theta1
    
    if dragging_double:
        (mx, my) = pg.mouse.get_pos()
        if drag_double_idx == 0:
            dx = mx - ORIGIN[0]
            dy = my - ORIGIN[1]
            l1 = min(math.sqrt(dx ** 2 + dy ** 2) / L_SCALE, 2.0)
            theta1_new = math.atan2(dx, dy)
            w1 = (theta1_new - theta1) / dt
            theta1 = theta1_new
            for i in range(PRECISION):
                _, _, theta2, w2 = rk4_double_pendulum(0 if abs(w1) < 0.00001 else theta1, w1, theta2, w2, l1, l2, m1, m2, MU, G, dt / PRECISION)
        elif drag_double_idx == 1:
            (x1, y1), _ = get_double_pendulum_positions(theta1, theta2, l1, l2)
            dx = mx - x1
            dy = my - y1
            l2 = min(math.sqrt(dx ** 2 + dy ** 2) / L_SCALE, 2.0)
            theta2_new = math.atan2(dx, dy) - theta1
            w2 = (theta2_new - theta2) / dt
            theta2 = theta2_new
    else:
        for i in range(PRECISION):
            theta1, w1, theta2, w2 = rk4_double_pendulum(theta1, w1, theta2, w2, l1, l2, m1, m2, MU, G, dt / PRECISION)
                
    if abs(theta1) > math.pi:
        theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    if abs(theta2) > math.pi:
        theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    if abs(theta1 - prev_theta1) > math.pi:
        SIGN = -SIGN

    screen.fill(WHITE)
    
    # Draw the checkbox
    checkbox_x, checkbox_y = WIDTH - 200, HEIGHT - 200
    draw_checkbox(checkbox_x, checkbox_y, "Draw Path", draw_path_enabled)

    # Conditionally draw the path
    if draw_path_enabled:
        draw_path()
    draw_graph()
    draw_double_pendulum(theta1, theta2 - theta1, l1, l2, m1, m2)

    gx, gy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 2
    mux, muy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN
    mass1x, mass1y = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 4
    mass2x, mass2y = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 3
    draw_slider(gx, gy, G, G_MIN, G_MAX, "g")
    draw_slider(mux, muy, MU, MU_MIN, MU_MAX, "damping")
    draw_slider(mass1x, mass1y, m1, MASS_MIN, MASS_MAX, "mass1")
    draw_slider(mass2x, mass2y, m2, MASS_MIN, MASS_MAX, "mass2")
    display_text(f"length1: {l1:.3f}", (WIDTH - 200, 50))
    display_text(f"length2: {l2:.3f}", (WIDTH - 200, 80))
    display_text(f"mass1: {m1:.2f}", (WIDTH - 200, 110))
    display_text(f"mass2: {m2:.2f}", (WIDTH - 200, 140))
    records.push_front(theta1 * L_SCALE * 0.7 * SIGN + HEIGHT // 2)
    records_small_pendulum.push_front(get_double_pendulum_positions(theta1, theta2 - theta1, l1, l2)[1])
    pg.display.flip()

pg.quit()