import pyaudio
import pygame as pg
import numpy as np
import random
import math

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WIDTH, HEIGHT = 1200, 600
obstacle_speed = 15
obstacle_width = 75       
min_obstacle_spacing = 600

fft_history = []
obstacles = []
score = 0
lives = 3
invulnerability_timer = 0
game_over = False
game_over_timer = 0
flashing = False
flash_timer = 0
collided_obs = None

def init_audio():
    global p, stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, frames_per_buffer=CHUNK)

def init_pygame():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    global font
    font = pg.font.SysFont("Arial", 48, bold=True)  # increased size and set bold
    return screen, clock

def process_audio():
    global fft_history
    # mic input, fft
    data = stream.read(CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.int16)
    fft_vals = np.abs(np.fft.rfft(samples))
    max_fft = np.max(fft_vals) if np.max(fft_vals) > 0 else 1
    normalized_fft = fft_vals / max_fft
    fft_history.append(normalized_fft)
    if len(fft_history) > 4:
        fft_history.pop(0)
    avg_fft = np.mean(fft_history, axis=0)
    
    # limit freq range, find dominant freq
    max_freq_display = 5000  # Hz
    bin_resolution = RATE / CHUNK
    max_bin = int(max_freq_display / bin_resolution)
    limited_len = min(max_bin, len(avg_fft))
    peak_index = np.argmax(avg_fft[:limited_len])
    if peak_index > 0 and peak_index + 1 < limited_len:
        alpha = avg_fft[peak_index - 1]
        beta = avg_fft[peak_index]
        gamma = avg_fft[peak_index + 1]
        p_interp = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        refined_peak = peak_index + p_interp
    else:
        refined_peak = peak_index
    freq = refined_peak * bin_resolution
    # maps the bird in a cycle
    C_ref = 130.81  # C3, hz
    relative = math.log2(0.000000001 + freq / C_ref) % 1
    return relative, avg_fft

def update_obstacles(bird_x, current_score):
    global obstacles, score
    gap = max(150, 250 - current_score * 2)
    if not obstacles or obstacles[-1]["x"] < WIDTH - min_obstacle_spacing:
        gap_y = random.randint(gap // 2, HEIGHT - gap // 2)
        obstacles.append({"x": WIDTH, "gap_y": gap_y, "passed": False})
    for obs in obstacles:
        obs["x"] -= obstacle_speed
        if obs["x"] + obstacle_width < bird_x and not obs["passed"]:
            score += 1
            obs["passed"] = True
    obstacles[:] = [obs for obs in obstacles if obs["x"] + obstacle_width > 0]
    return gap

def check_collisions(bird_rect, gap):
    global obstacles, lives, invulnerability_timer, game_over, game_over_timer
    global flashing, flash_timer, collided_obs
    for obs in obstacles:
        top_rect = pg.Rect(obs["x"], 0, obstacle_width, obs["gap_y"] - gap // 2)
        bottom_rect = pg.Rect(obs["x"], obs["gap_y"] + gap // 2, obstacle_width, HEIGHT - (obs["gap_y"] + gap // 2))
        if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
            if invulnerability_timer <= 0:
                lives -= 1
                invulnerability_timer = 1000  # 1 second invulnerability
                if lives <= 0:
                    game_over = True
                    game_over_timer = 3000
                    flashing = False
                    collided_obs = obs
                else:
                    flashing = True
                    flash_timer = 1000
                    collided_obs = obs
            break

def draw_fourier_graph(screen, local_avg):
    max_freq_display = 5000  # hz
    bin_resolution = RATE / CHUNK
    max_bin = int(max_freq_display / bin_resolution)
    limited_len = min(max_bin, len(local_avg))
    fourier_color = (0, 0, 255)  # blue
    points = []
    for i in range(limited_len):
        x = i * (WIDTH / limited_len)
        y = HEIGHT - (local_avg[i] * (HEIGHT * 0.5))
        points.append((x, y))
    if len(points) > 1:
        pg.draw.lines(screen, fourier_color, False, points, 2)

def draw_background(screen):
    whole_notes = [("C", 0), ("D", 2), ("E", 4), ("F", 5), 
                   ("G", 7), ("A", 9), ("B", 11), ("C", 0)]
    for note, semitone in whole_notes:
        rel_note = semitone / 12.0
        y_line = HEIGHT - int(rel_note * HEIGHT)
        pg.draw.line(screen, (50, 50, 50), (0, y_line), (WIDTH, y_line))
        label = font.render(note, True, (100, 100, 100))
        screen.blit(label, (10, y_line - label.get_height() // 2))

def draw_obstacles(screen, gap):
    for obs in obstacles:
        top_rect = pg.Rect(obs["x"], 0, obstacle_width, obs["gap_y"] - gap // 2)
        bottom_rect = pg.Rect(obs["x"], obs["gap_y"] + gap // 2, obstacle_width, 
                              HEIGHT - (obs["gap_y"] + gap // 2))
        pg.draw.rect(screen, (0, 255, 0), top_rect)
        pg.draw.rect(screen, (0, 255, 0), bottom_rect)

def draw_bird(screen, bird_x, bird_y):
    if flashing:
        flash_color = (255, 255, 255) if (flash_timer // 100) % 2 == 0 else (255, 0, 0)
        pg.draw.circle(screen, flash_color, (bird_x, bird_y), 30)
    else:
        pg.draw.circle(screen, (255, 0, 0), (bird_x, bird_y), 30)

def draw_ui(screen, score, lives):
    score_surface = font.render(f"Score: {score}", True, (255, 255, 255))
    lives_surface = font.render(f"Lives: {lives}", True, (255, 255, 255))
    screen.blit(score_surface, (WIDTH - score_surface.get_width() - 15, 15))
    screen.blit(lives_surface, (WIDTH - lives_surface.get_width() - 15, 90))
    if game_over:
        over_surface = font.render("GAME OVER", True, (255, 0, 0))
        screen.blit(over_surface, (WIDTH // 2 - over_surface.get_width() // 2, 
                                   HEIGHT // 2 - over_surface.get_height() // 2))

def main():
    global flashing, flash_timer, collided_obs, invulnerability_timer
    global game_over, game_over_timer, lives, score
    running = True
    while running:
        dt = clock.tick(30)
        if invulnerability_timer > 0:
            invulnerability_timer -= dt
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        if not game_over:
            if not flashing:
                relative, local_avg = process_audio()
                bird_x = 75
                bird_y = HEIGHT - int(relative * HEIGHT)
                gap = max(150, 250 - score * 2)
                gap = update_obstacles(bird_x, score)
                bird_rect = pg.Rect(bird_x - 30, bird_y - 30, 60, 60)
                check_collisions(bird_rect, gap)
            else:
                flash_timer -= dt
                if flash_timer <= 0:
                    flashing = False
                    if collided_obs in obstacles:
                        obstacles.remove(collided_obs)
                    collided_obs = None
        else:
            game_over_timer -= dt
            if game_over_timer <= 0:
                running = False

        screen.fill((0, 0, 0))
        if fft_history:
            local_avg_draw = np.mean(fft_history, axis=0)
            draw_fourier_graph(screen, local_avg_draw)
        draw_background(screen)
        current_gap = max(100, 250 - score * 10)
        draw_obstacles(screen, current_gap)
        relative_dummy, _ = process_audio()
        draw_bird(screen, 75, HEIGHT - int(relative_dummy * HEIGHT))
        draw_ui(screen, score, lives)
        pg.display.flip()
    stream.stop_stream()
    stream.close()
    p.terminate()
    pg.quit()

if __name__ == "__main__":
    init_audio()
    screen, clock = init_pygame()
    main()
