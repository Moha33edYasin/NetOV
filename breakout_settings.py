import pygame

def create_board(rows, cols, blocks_rows):
    blocks_start_row = int(rows / 6)
    blocks_end_row = blocks_start_row + blocks_rows
    return [
        [(i - blocks_start_row) for j in range(cols)] 
        if blocks_start_row <= i <= blocks_end_row else  
        [0 for _ in range(cols)] for i in range(rows)
    ]

W, H = 800, 600
NUM_ROW, NUM_COL, NUM_BLOCKS_ROW = 40, 20, 6 # 40, 20, 6
FPS = 60

BLOCED_CODE = [1, 2, 3, 4, 5, 6]

LAZY_RENDERS = {
    1 : lambda screen, obj: pygame.draw.rect(screen, "red", [*obj.pos, obj.width-1, obj.height-1]),
    2 : lambda screen, obj: pygame.draw.rect(screen, "orange", [*obj.pos, obj.width-1, obj.height-1]),
    3 : lambda screen, obj: pygame.draw.rect(screen, "brown", [*obj.pos, obj.width-1, obj.height-1]),
    4 : lambda screen, obj: pygame.draw.rect(screen, "yellow", [*obj.pos, obj.width-1, obj.height-1]),
    5 : lambda screen, obj: pygame.draw.rect(screen, "green", [*obj.pos, obj.width-1, obj.height-1]),
    6 : lambda screen, obj: pygame.draw.rect(screen, "blue", [*obj.pos, obj.width-1, obj.height-1])
}

WORLD_MAP = create_board(NUM_ROW, NUM_COL, NUM_BLOCKS_ROW) # Experiment 2: (20, 10, 3)

CELL_W = W / NUM_COL
CELL_H = H / NUM_ROW

PEDDEL_H = 0.6 * CELL_H
PEDDEL_W = 2 * CELL_W

BALL_RADIUS = 0.35 * W / H * CELL_H

TRAVERSE_JUMP = min(CELL_W * CELL_W, CELL_H * CELL_H) * 1.15
NORMAL_SPEED = TRAVERSE_JUMP * 1
FAST_SPEED1 = TRAVERSE_JUMP * 1.25
FAST_SPEED2 = TRAVERSE_JUMP * 1.75
FAST_SPEED3 = TRAVERSE_JUMP * 2