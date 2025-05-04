import pygame as pg
import math
import numpy as np
import twophase.solver as sv

WIDTH, HEIGHT = 900, 900
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

anim_move = None
anim_speed = 0.05
solution_moves = []
solution_display = ""
green = False

cube_vertices = np.array([
    [-1.5, -1.5, -1.5],
    [1.5, -1.5, -1.5],
    [1.5, 1.5, -1.5],
    [-1.5, 1.5, -1.5],
    [-1.5, -1.5, 1.5],
    [1.5, -1.5, 1.5],
    [1.5, 1.5, 1.5],
    [-1.5, 1.5, 1.5],
])

face_colors = np.array([
    (255, 255, 255), # U - white
    (255, 255, 0),   # D - yellow
    (0, 255, 0),     # F - green
    (0, 0, 255),     # B - blue
    (255, 140, 0),   # L - orange
    (255, 0, 0)      # R - red
])

cube_faces = np.array([
    (3, 2, 1, 0), # U
    (4, 5, 6, 7), # D
    (0, 1, 5, 4), # F
    (6, 2, 3, 7), # B
    (7, 3, 0, 4), # L
    (1, 2, 6, 5)  # R
])

cube_colors = np.zeros((6, 3, 3), dtype=int)
for i in range(6):
    cube_colors[i, :, :] = i

def rodrigues_rotate(v, k, theta):
    v = np.array(v)
    k = np.array(k)
    k = k / np.linalg.norm(k)
    return (v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + k * np.dot(k, v) * (1 - math.cos(theta))).tolist()

def project(point, width, height, fov, viewer_distance):
    x, y, z = point
    factor = fov / (viewer_distance + z)
    x = x * factor + width / 2
    y = -y * factor + height / 2
    return (int(x), int(y))

def rotate_face_clockwise(face):
    tmp00 = cube_colors[face][0, 0]
    tmp01 = cube_colors[face][0, 1]
    tmp02 = cube_colors[face][0, 2]
    tmp10 = cube_colors[face][1, 0]
    tmp12 = cube_colors[face][1, 2]
    tmp20 = cube_colors[face][2, 0]
    tmp21 = cube_colors[face][2, 1]
    tmp22 = cube_colors[face][2, 2]

    cube_colors[face][0, 0] = tmp20
    cube_colors[face][0, 2] = tmp00
    cube_colors[face][2, 2] = tmp02
    cube_colors[face][2, 0] = tmp22

    cube_colors[face][0, 1] = tmp10
    cube_colors[face][1, 2] = tmp01
    cube_colors[face][2, 1] = tmp12
    cube_colors[face][1, 0] = tmp21
    
def rotate_face_counterclockwise(face):
    for _ in range(3):
        rotate_face_clockwise(face)

def rotate_slice_clockwise(slice): # green is facing you
    if slice >= 0 and slice <= 2: # F-B
        slice_idx = 2 - slice
        tmp0 = cube_colors[0, slice_idx].copy()
        tmp1 = cube_colors[5, :, slice_idx].copy()
        tmp2 = cube_colors[1, slice_idx].copy()
        tmp3 = cube_colors[4, :, slice_idx].copy()

        cube_colors[0, slice_idx] = np.flip(tmp3)
        cube_colors[5, :, slice_idx] = tmp0
        cube_colors[1, slice_idx] = np.flip(tmp1)
        cube_colors[4, :, slice_idx] = tmp2
    elif slice >= 3 and slice <= 5: # U-D
        slice_idx = slice - 3
        tmp0 = cube_colors[2, slice_idx].copy()
        tmp1 = cube_colors[5, slice_idx].copy()
        tmp2 = cube_colors[3, slice_idx].copy()
        tmp3 = cube_colors[4, slice_idx].copy()
        cube_colors[2, slice_idx] = np.flip(tmp1)
        cube_colors[5, slice_idx] = tmp2
        cube_colors[3, slice_idx] = np.flip(tmp3)
        cube_colors[4, slice_idx] = tmp0
    else: # L-R
        slice_idx = slice - 6
        tmp0 = cube_colors[2, :, slice_idx].copy()
        tmp1 = cube_colors[1, :, slice_idx].copy()
        tmp2 = cube_colors[3, :, slice_idx].copy()
        tmp3 = cube_colors[0, :, slice_idx].copy()
        cube_colors[2, :, slice_idx] = tmp3
        cube_colors[1, :, slice_idx] = np.flip(tmp0)
        cube_colors[3, :, slice_idx] = tmp1
        cube_colors[0, :, slice_idx] = np.flip(tmp2)

def rotate_slice_counterclockwise(slice): # green is F
    for i in range(3):
        rotate_slice_clockwise(slice)

def turn_cube_clockwise(turn):
    match turn:
        case 'F':
            rotate_face_clockwise(2)
            rotate_slice_clockwise(0)
        case 'B':
            rotate_face_counterclockwise(3)
            rotate_slice_counterclockwise(2)
        case 'U':
            rotate_face_clockwise(0)
            rotate_slice_clockwise(3)
        case 'D':
            rotate_face_counterclockwise(1)
            rotate_slice_counterclockwise(5)
        case 'L':
            rotate_face_clockwise(4)
            rotate_slice_clockwise(6)
        case 'R':
            rotate_face_counterclockwise(5)
            rotate_slice_counterclockwise(8)

def execute_turn(turn, prime):
    match prime:
        case False:
            turn_cube_clockwise(turn)
        case True:
            for i in range(3):
                turn_cube_clockwise(turn)
            

def draw_lattice_lines_perspective(screen, face_vertices_3d, project_func, width, height, fov, viewer_distance):
    # vertical 
    for i in range(3):
        t = i / 3
        v0 = [face_vertices_3d[0][j] + (face_vertices_3d[1][j] - face_vertices_3d[0][j]) * t for j in range(3)]
        v1 = [face_vertices_3d[3][j] + (face_vertices_3d[2][j] - face_vertices_3d[3][j]) * t for j in range(3)]
        p0 = project_func(v0, width, height, fov, viewer_distance)
        p1 = project_func(v1, width, height, fov, viewer_distance)
        pg.draw.line(screen, BLACK, p0, p1, 10)
    
    # horizontal
    for i in range(3):
        t = i / 3
        v0 = [face_vertices_3d[0][j] + (face_vertices_3d[3][j] - face_vertices_3d[0][j]) * t for j in range(3)]
        v1 = [face_vertices_3d[1][j] + (face_vertices_3d[2][j] - face_vertices_3d[1][j]) * t for j in range(3)]
        p0 = project_func(v0, width, height, fov, viewer_distance)
        p1 = project_func(v1, width, height, fov, viewer_distance)
        pg.draw.line(screen, BLACK, p0, p1, 10)

def rotate_matrix_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotate_matrix_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotate_matrix_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def get_face_normal(face_verts):
    # normal vector of the face defined by three vertices
    v1 = np.array(face_verts[1]) - np.array(face_verts[0])
    v2 = np.array(face_verts[2]) - np.array(face_verts[0])
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def get_visible_faces(rotated_verts):
    visible = []
    for i, face in enumerate(cube_faces):
        verts = [rotated_verts[j] for j in face]
        normal = get_face_normal(verts)
        
        # visible when normal points towards the camera
        if normal[2] < 0:  # camera looks along -z
            visible.append(i)
    return visible

def get_sticker_polygons(face_idx, rotated_verts, width, height, fov, viewer_distance, transform_func=None):
    face = cube_faces[face_idx]
    v = [rotated_verts[j] for j in face]
    
    match face_idx:
        case 0:  # U
            v00, v01, v11, v10 = v[0], v[1], v[2], v[3]
        case 1:  # D
            v00, v01, v11, v10 = v[3], v[2], v[1], v[0]
        case 2:  # F
            v00, v01, v11, v10 = v[0], v[1], v[2], v[3]
        case 3:  # B
            v00, v01, v11, v10 = v[2], v[1], v[0], v[3]
        case 4:  # L
            v00, v01, v11, v10 = v[1], v[2], v[3], v[0]
        case 5:  # R
            v00, v01, v11, v10 = v[1], v[0], v[3], v[2]
            
    polys = []
    for row in range(3):
        for col in range(3):
            def interp(r, c):
                s = r / 3
                t = c / 3
                p = [(1 - s) * (1 - t) * v00[j] + s * (1 - t) * v10[j] + s * t * v11[j] + (1 - s) * t * v01[j] for j in range(3)]
                if transform_func:
                    p = transform_func(p)
                return p
            corners3d = [interp(row, col), interp(row, col + 1), interp(row + 1, col + 1), interp(row + 1, col)]
            corners2d = [project(c, width, height, fov, viewer_distance) for c in corners3d]
            
            # tuple: sticker grid location, projected corners, original 3D corners
            polys.append(((row, col), corners2d, corners3d))
    return polys

def draw_cube(screen, z_centers, rotated_verts, width, height, fov, viewer_distance):
    faces_pairs = [(face, face_colors[i], z_centers[i], i) for i, face in enumerate(cube_faces)]
    faces_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if (anim_move is not None) and (anim_move['turn'] in {'F','U','D','L','R','B'}):
        turning_face_idx = anim_move['face']
        turning_face_verts = [rotated_verts[j] for j in cube_faces[turning_face_idx]]
        turning_axis = get_face_normal(turning_face_verts)
        turn_angle = anim_move['current_angle'] if not anim_move['prime'] else -anim_move['current_angle']
        transform_adj = lambda p: rodrigues_rotate(p, turning_axis, turn_angle)
        adjacent_mapping = {
            'F': [(0, 'row', 2), (5, 'col', 2), (1, 'row', 2), (4, 'col', 2)],
            'U': [(2, 'row', 0), (5, 'row', 0), (3, 'row', 0), (4, 'row', 0)],
            'D': [(2, 'row', 2), (5, 'row', 2), (3, 'row', 2), (4, 'row', 2)],
            'L': [(2, 'col', 0), (1, 'col', 0), (3, 'col', 0), (0, 'col', 0)],
            'R': [(2, 'col', 2), (1, 'col', 2), (3, 'col', 2), (0, 'col', 2)],
            'B': [(0, 'row', 0), (5, 'col', 0), (1, 'row', 0), (4, 'col', 0)]
        }
        
    # sticker polygons
    polygons_to_draw = []
    for _, _, _, face_idx in faces_pairs:
        sticker_polys = get_sticker_polygons(face_idx, rotated_verts, width, height, fov, viewer_distance)
        for (row, col), poly, corners3d in sticker_polys:
            if anim_move is not None:
                if anim_move['face'] == face_idx:
                    new_poly = [project(transform_adj(c), width, height, fov, viewer_distance) for c in corners3d]
                    transformed_corners = [transform_adj(c) for c in corners3d]
                elif anim_move['turn'] in adjacent_mapping:
                    applied = False
                    for (fid, cond, idx) in adjacent_mapping[anim_move['turn']]:
                        if fid == face_idx and ((cond == 'row' and row == idx) or (cond == 'col' and col == idx)):
                            new_poly = [project(transform_adj(c), width, height, fov, viewer_distance) for c in corners3d]
                            transformed_corners = [transform_adj(c) for c in corners3d]
                            applied = True
                            break
                    if not applied:
                        new_poly = poly
                        transformed_corners = corners3d
                else:
                    new_poly = poly
                    transformed_corners = corners3d
            else:
                new_poly = poly
                transformed_corners = corners3d
            avg_z = sum(c[2] for c in transformed_corners) / 4
            polygons_to_draw.append((avg_z, new_poly, face_colors[cube_colors[face_idx, row, col]]))
    
    if anim_move is not None and anim_move['turn'] in {'F','U','D','L','R','B'}:
        if anim_move['face'] in range(6):
            v = [rotated_verts[j] for j in cube_faces[anim_move['face']]]
            match anim_move['face']:
                case 0:  # U
                    v00, v01, v11, v10 = v[0], v[1], v[2], v[3]
                case 1:  # D
                    v00, v01, v11, v10 = v[3], v[2], v[1], v[0]
                case 2:  # F
                    v00, v01, v11, v10 = v[0], v[1], v[2], v[3]
                case 3:  # B
                    v00, v01, v11, v10 = v[2], v[1], v[0], v[3]
                case 4:  # L
                    v00, v01, v11, v10 = v[1], v[2], v[3], v[0]
                case 5:  # R
                    v00, v01, v11, v10 = v[1], v[0], v[3], v[2]
                    
            face_center = [(v00[i] + v01[i] + v11[i] + v10[i]) / 4 for i in range(3)]
            edge = np.linalg.norm(np.array(v01) - np.array(v00))
            sticker_width = edge / 3.0
            dir_to_center = np.array([0, 0, 0]) - np.array(face_center)
            if np.linalg.norm(dir_to_center) != 0:
                shift = (dir_to_center / np.linalg.norm(dir_to_center)) * sticker_width
            else:
                shift = np.array([0, 0, 0])
                
            inner_face = [ (np.array(pt) + shift).tolist() for pt in [v00, v01, v11, v10] ]
            overall_inner_center = [sum(pt[i] for pt in inner_face) / 4 for i in range(3)]
           
            def bilerp(s, t):
                        return [(1 - s) * ((1 - t) * inner_face[0][k] + t * inner_face[1][k]) + s * ((1 - t) * inner_face[3][k] + t * inner_face[2][k]) for k in range(3)]
           
            # subdivide inner_face into 3x3 cells
            for i in range(3):
                for j in range(3):
                    s0 = i / 3; s1 = (i+1) / 3; t0 = j / 3; t1 = (j+1) / 3

                    cell_corners = [ bilerp(s, t) for s, t in [(s0, t0), (s0, t1), (s1, t1), (s1, t0)] ]
                    
                    # rotate cell_corners around the face center
                    rotated_corners = []
                    for corner in cell_corners:
                        rel = np.array(corner) - np.array(overall_inner_center)
                        rotated_rel = transform_adj(rel.tolist())
                        rotated_corner = np.array(overall_inner_center) + np.array(rotated_rel)
                        rotated_corners.append(rotated_corner.tolist())
                    cell_poly = [ project(pt, width, height, fov, viewer_distance) for pt in rotated_corners ]
                    cell_avg_z = sum(pt[2] for pt in rotated_corners) / 4
                    polygons_to_draw.append((cell_avg_z, cell_poly, BLACK))
                    
            # non-rotating inner face
            for i in range(3):
                for j in range(3):
                    s0 = i / 3; s1 = (i+1) / 3; t0 = j / 3; t1 = (j+1) / 3
                    cell_corners_nr = [bilerp(s, t) for s, t in [(s0, t0), (s0, t1), (s1, t1), (s1, t0)]]
                    
                    # no rotate
                    cell_poly_nr = [project(pt, width, height, fov, viewer_distance) for pt in cell_corners_nr]
                    cell_avg_z_nr = sum(pt[2] for pt in cell_corners_nr) / 4
                    polygons_to_draw.append((cell_avg_z_nr, cell_poly_nr, BLACK))
    
    # painter's algorithm: draws polygons in order from back to front
    polygons_to_draw.sort(key=lambda x: x[0], reverse=True)
    for _, poly_to_draw, col in polygons_to_draw:
        pg.draw.polygon(screen, col, poly_to_draw, 0)
        pg.draw.polygon(screen, BLACK, poly_to_draw, 2)

def get_cube_definition():
    face_letter = {0: "U", 1: "D", 2: "F", 3: "B", 4: "L", 5: "R"}

    order = [0, 5, 2, 1, 4, 3]
    s = ""
    for idx in order:
        if idx == 1:
            for i in range(2, -1, -1):
                for sticker in cube_colors[idx][i]:
                    s += face_letter[sticker]
            continue
        for row in cube_colors[idx]:
            if idx in [5, 3]:
                row = row[::-1]
            for sticker in row:
                s += face_letter[sticker]

    return s

# Add a slider for move turn speed
slider_rect = pg.Rect(20, 20, 200, 20)
slider_knob_rect = pg.Rect(20, 15, 10, 30)
slider_dragging = False

def draw_slider(screen, slider_rect, knob_rect):
    pg.draw.rect(screen, (200, 200, 200), slider_rect)
    pg.draw.rect(screen, BLACK, slider_rect, 2)
    pg.draw.rect(screen, (100, 100, 100), knob_rect)
    pg.draw.rect(screen, BLACK, knob_rect, 2)

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()
solve_button_rect = pg.Rect(WIDTH - 150, 20, 130, 40)
font = pg.font.SysFont(None, 30)

orientation = rotate_matrix_x(math.pi / 2) 
running = True
mouse_down = False
last_mouse_pos = None

move_to_face = {"U": 0, "D": 1, "F": 2, "B": 3, "L": 4, "R": 5}

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                if solve_button_rect.collidepoint(event.pos):
                    cube_str = get_cube_definition()
                    solution = sv.solve(cube_str, timeout=0.3)

                    # parse solution 
                    solution_moves = solution.strip().split()
                    solution_display = ""
                    green = False
                elif slider_knob_rect.collidepoint(event.pos):
                    slider_dragging = True
                else:
                    mouse_down = True
                    last_mouse_pos = pg.mouse.get_pos()
        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False
                slider_dragging = False
        elif event.type == pg.MOUSEMOTION:
            if mouse_down:
                x, y = pg.mouse.get_pos()
                last_x, last_y = last_mouse_pos
                dx = x - last_x
                dy = y - last_y

                rot_y = rotate_matrix_y(-dx * 0.01)
                rot_x = rotate_matrix_x(-dy * 0.01)
                orientation = rot_x @ rot_y @ orientation

                center_x, center_y = WIDTH / 2, HEIGHT / 2
                angle_prev = math.atan2(last_mouse_pos[1] - center_y, last_mouse_pos[0] - center_x)
                angle_curr = math.atan2(y - center_y, x - center_x)
                d_angle = angle_curr - angle_prev

                if abs(d_angle) > 0.2:
                    d_angle = 0.2 if d_angle > 0 else -0.2
                rot_z = rotate_matrix_z(d_angle * 0.25)
                orientation = orientation @ rot_z

                last_mouse_pos = (x, y)
            elif slider_dragging:
                x, _ = pg.mouse.get_pos()
                slider_knob_rect.x = max(slider_rect.x, min(x, slider_rect.x + slider_rect.width - slider_knob_rect.width))
                anim_speed = math.pi / 60 + (math.pi / 30) * ((slider_knob_rect.x - slider_rect.x) / (slider_rect.width - slider_knob_rect.width))
        elif event.type == pg.KEYDOWN and anim_move is None:
            prime = pg.key.get_pressed()[pg.K_LSHIFT] or pg.key.get_pressed()[pg.K_RSHIFT]
            key_to_turn = {pg.K_u: 'U', pg.K_d: 'D', pg.K_f: 'F', pg.K_b: 'B', pg.K_l: 'L', pg.K_r: 'R'}
            if event.key in key_to_turn:
                turn = key_to_turn[event.key]
                total_frames = int(math.pi / 2 / anim_speed)
                anim_move = {
                    'face': move_to_face[turn],
                    'turn': turn,
                    'prime': prime,
                    'current_angle': 0,
                    'target_angle': math.pi / 2,
                    'speed': anim_speed,
                    'current_frame': 0,
                    'total_frames': total_frames
                }
            elif event.key == pg.K_SPACE:
                cube_colors = np.zeros((6, 3, 3), dtype=int)
                for i in range(6):
                    cube_colors[i, :, :] = i
                orientation = rotate_matrix_x(math.pi / 2)

    if (anim_move is None) and solution_moves:
        move = solution_moves.pop(0)
        if move[0] == '(':
            green = True
            continue
        letter = move[0]
        prime = move[1] == '3'
        double = move[1] == '2'
        cur_moves = solution_display.split()
        if len(cur_moves) > 0 and cur_moves[-1][0] == letter:
            cur_moves.pop()
            cur_moves.append(letter + '2')
        else:
            cur_moves.append(letter + ("'" if prime else ""))
        solution_display = " ".join(cur_moves)
        if letter in move_to_face:
            total_frames = int(math.pi / 2 / anim_speed)
            anim_move = {
                'face': move_to_face[letter],
                'turn': letter,
                'prime': prime,
                'current_angle': 0,
                'target_angle': math.pi/2,
                'speed': anim_speed,
                'current_frame': 0,
                'total_frames': total_frames
            }
        if double:
            solution_moves.insert(0, letter + '1')

    if anim_move is not None:
        # ease in-out animation
        t = anim_move['current_frame'] / anim_move['total_frames']
        t = min(max(t, 0), 1)
        angle = anim_move['target_angle'] * (0.5 - 0.5 * math.cos(math.pi * t))
        anim_move['current_angle'] = angle
        anim_move['current_frame'] += 1
        if anim_move['current_frame'] > anim_move['total_frames']:
            anim_move['current_angle'] = anim_move['target_angle']
            execute_turn(anim_move['turn'], anim_move['prime'])
            anim_move = None

    screen.fill(WHITE)
    rotated = []
    for v in cube_vertices:
        v_np = np.array(v)
        r = orientation @ v_np
        rotated.append(r.tolist())
        projected = project(r, WIDTH, HEIGHT, fov=2000, viewer_distance=6)
    
    face_centers = [sum(rotated[idx][2] for idx in face) / 4 for face in cube_faces]
    draw_cube(screen, face_centers, rotated, WIDTH, HEIGHT, fov=800, viewer_distance=6)
    
    # solve button
    pg.draw.rect(screen, (200, 200, 200), solve_button_rect)
    pg.draw.rect(screen, BLACK, solve_button_rect, 2)
    button_text = font.render("Solve Cube", True, BLACK)
    text_rect = button_text.get_rect(center = solve_button_rect.center)
    screen.blit(button_text, text_rect)
    
    if solution_display:
        solution_text = font.render("Solution: " + solution_display, True, (0, 210, 0) if green else BLACK)
        screen.blit(solution_text, (20, HEIGHT - 40))
    
    screen.blit(font.render("Move Speed: " + str(round(anim_speed, 2)), True, BLACK), (20, 50))
    
    draw_slider(screen, slider_rect, slider_knob_rect)
    
    pg.display.flip()
    clock.tick(60)

pg.quit()