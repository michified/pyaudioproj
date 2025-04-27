import math
import pygame as pg
from pygame import gfxdraw

def theta_double_dot(theta, w, l, mu, g):
    return -((g / l) * math.sin(theta) + w * mu)

def damped_pendulum_angle(thetaT, wT, l, mu, g, dt): # euler method
    """
    Parameters:
        thetaT: Angle at time t (rad)
        wT: Angular velocity (rad/s)
        l: Length of the pendulum (m)
        mu: Damping coefficient (kgm^2s^-1)
        g: Acceleration due to gravity (ms^-2)
        dt: Time step (s)

    Outputs: 
        Angle at time t + dt (rad)
        Angular velocity at time t + dt (rad/s)
    """
    theta = thetaT + wT * dt
    w = wT + theta_double_dot(theta, wT, l, mu, g) * dt
    return theta, w

WIDTH, HEIGHT = 1400, 800

SLIDER_WIDTH = WIDTH // 2
SLIDER_HEIGHT = 8
SLIDER_HANDLE_RADIUS = 12
SLIDER_MARGIN = SLIDER_HANDLE_RADIUS * 2 + 30

ORIGIN = (WIDTH * 3 // 4, HEIGHT // 2 - SLIDER_MARGIN)
L_SCALE = HEIGHT // 6
PENDULUM_RADIUS = 30
PRECISION = 10000

G_MIN, G_MAX = 0.0, 100.0
MU_MIN, MU_MAX = 0.0, 1.0

G = 9.81
MU = 0.01
L = 1.0
SIGN = 1

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

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()
font = pg.font.SysFont(None, 28)

theta = 0.0
w = 0.0
dragging = False
slider_drag = None  # None, "g", or "mu"

def get_pendulum_pos(theta, l):
    x = ORIGIN[0] + math.sin(theta) * l * L_SCALE
    y = ORIGIN[1] + math.cos(theta) * l * L_SCALE
    return int(x), int(y)

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
    text = font.render(f"{label}: {value:.3f}", True, BLACK)
    screen.blit(text, (x, y - 30))
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
    text = font.render("θ (rad)", True, BLACK)
    screen.blit(text, (70, HEIGHT // 2 - math.pi * L_SCALE * 0.7))
    text = font.render("t (s)", True, BLACK)
    screen.blit(text, (len(records) * SCROLL_STEP - 50, HEIGHT // 2 + 15))
    text = font.render("π", True, BLACK)
    screen.blit(text, (27, HEIGHT // 2 - math.pi * L_SCALE * 0.7 - 10))
    text = font.render("-π", True, BLACK)
    screen.blit(text, (20, HEIGHT // 2 + math.pi * L_SCALE * 0.7 - 10))
    pg.draw.line(screen, BLACK, (45, HEIGHT // 2 - math.pi * L_SCALE * 0.7), (55, HEIGHT // 2 - math.pi * L_SCALE * 0.7), 2)
    pg.draw.line(screen, BLACK, (45, HEIGHT // 2 + math.pi * L_SCALE * 0.7), (55, HEIGHT // 2 + math.pi * L_SCALE * 0.7), 2)
    
def slider_value_from_pos(mouse_x, x, min_val, max_val):
    norm = min(max((mouse_x - x) / SLIDER_WIDTH, 0), 1)
    return min_val + norm * (max_val - min_val)

running = True
while running:
    dt = clock.tick(60) / 1000
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            mx, my = pg.mouse.get_pos()
            px, py = get_pendulum_pos(theta, L)

            if (mx - px) ** 2 + (my - py) ** 2 < PENDULUM_RADIUS ** 2:
                dragging = True
                
            # Check sliders
            gx, gy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 2
            mux, muy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN
            g_handle_x, g_handle_y = draw_slider(gx, gy, G, G_MIN, G_MAX, "g")
            mu_handle_x, mu_handle_y = draw_slider(mux, muy, MU, MU_MIN, MU_MAX, "mu")
            if (mx - g_handle_x) ** 2 + (my - g_handle_y) ** 2 < SLIDER_HANDLE_RADIUS ** 2:
                slider_drag = "g"
            elif (mx - mu_handle_x) ** 2 + (my - mu_handle_y) ** 2 < SLIDER_HANDLE_RADIUS ** 2:
                slider_drag = "mu"
        elif event.type == pg.MOUSEBUTTONUP:
            dragging = False
            slider_drag = None
        elif event.type == pg.MOUSEMOTION and slider_drag:
            mx, my = event.pos
            if slider_drag == "g":
                gx = WIDTH // 2 - SLIDER_WIDTH // 2
                G = slider_value_from_pos(mx, gx, G_MIN, G_MAX)
            elif slider_drag == "mu":
                mux = WIDTH // 2 - SLIDER_WIDTH // 2
                MU = slider_value_from_pos(mx, mux, MU_MIN, MU_MAX)
        elif event.type == pg.KEYDOWN:
            theta = 0
            w = 0
            L = 1.0
            G = 9.81
            MU = 0.01
            records.clear()
            dragging = False
            slider_drag = None
            SIGN = 1

    prev_theta = theta
    if dragging:
        mx, my = pg.mouse.get_pos()
        dx = mx - ORIGIN[0]
        dy = my - ORIGIN[1]
        L = min(math.sqrt(dx ** 2 + dy ** 2) / L_SCALE, 2.0)
        theta1 = math.atan2(dx, dy)
        w = (theta1 - theta) / dt
        theta = theta1
    else:
        for i in range(PRECISION):
            theta, w = damped_pendulum_angle(theta, w, L, MU, G, dt / PRECISION)
    if abs(theta) > math.pi:
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
    if abs(theta - prev_theta) > math.pi:
        SIGN = -SIGN
    
    screen.fill(WHITE)
    px, py = get_pendulum_pos(theta, L)
    pg.draw.line(screen, BLACK, ORIGIN, (px, py), 3)
    gfxdraw.aacircle(screen, px, py, PENDULUM_RADIUS, BLUE)
    gfxdraw.filled_circle(screen, px, py, PENDULUM_RADIUS, BLUE)
    gfxdraw.aacircle(screen, ORIGIN[0], ORIGIN[1], 5, BLACK)
    gfxdraw.filled_circle(screen, ORIGIN[0], ORIGIN[1], 5, BLACK)

    gx, gy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN * 2
    mux, muy = WIDTH // 2 - SLIDER_WIDTH // 2, HEIGHT - SLIDER_MARGIN
    draw_slider(gx, gy, G, G_MIN, G_MAX, "g")
    draw_slider(mux, muy, MU, MU_MIN, MU_MAX, "damping")
    text = font.render(f"{"length"}: {L:.3f}", True, BLACK)
    screen.blit(text, (WIDTH - 150, 50))
    records.push_front(theta * L_SCALE * 0.7 * SIGN + HEIGHT // 2)
    
    draw_graph()

    pg.display.flip()

pg.quit()