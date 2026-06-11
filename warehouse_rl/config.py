# default vrijednosti, azuriraju se nakon ucitavanja mape
GRID_W, GRID_H = 16, 12
TILE = 48
WIN_W, WIN_H = GRID_W * TILE, GRID_H * TILE + 80

# hiperparametri
EPISODES = 1200
MAX_STEPS = 300
ALPHA   = 0.7
GAMMA   = 0.95
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 0.997

# sustav nagradi
R_GOAL     =  100.0
R_OBSTACLE = -20.0
R_STEP     =  -0.5
R_REVISIT  =  -1.0
R_PICKUP   =  50.0

GRAPH_UPDATE_INTERVAL = 2000

# azuriraj ovisno o ucitanoj mapis
def apply_map_config(map_config):
    global GRID_W, GRID_H, WIN_W, WIN_H
    GRID_W = map_config["grid_w"]
    GRID_H = map_config["grid_h"]
    WIN_W = GRID_W * TILE
    WIN_H = GRID_H * TILE + 80
