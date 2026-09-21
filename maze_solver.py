import pygame
from markov import *

W, H = 900, 600

_ = 1
B = 2
G = 3

blocked_code = [_]
lazy_render = {
    _ : (lambda screen, obj: pygame.draw.rect(screen, "gray", [*obj.pos, obj.width-1, obj.height-1])),
    B : lambda screen, obj: pygame.draw.rect(screen, "green", [*obj.pos, obj.width, obj.height]),
    G : lambda screen, obj: pygame.draw.rect(screen, "red", [*obj.pos, obj.width, obj.height])
}

world_map = [
    [B, 0, _, 0, _, 0, 0, 0, 0, 0],
    [0, 0, _, 0, _, 0, 0, _, 0, 0],
    [0, 0, _, 0, _, 0, 0, _, 0, 0],
    [0, 0, _, 0, _, 0, 0, _, 0, 0],
    [0, 0, _, 0, 0, 0, 0, _, 0, 0],
    [0, 0, _, _, _, _, 0, _, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, _, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, _, 0, 0],
    [0, 0, _, _, _, _, 0, _, 0, 0],
    [0, 0, _, 0, 0, 0, 0, _, 0, G]
]

cell_w = W / len(world_map[0])
cell_h = H / len(world_map)

def arrow_labels(renderer, action_idx, coordinate):
    j, i = coordinate
    cell_w, cell_h = renderer.CELL_SIZE

    if action_idx == 0:
        p1 = (j, i)
        p2 = (j + 0.4 * cell_w, i)

    elif action_idx == 1:
        p1 = (j, i)
        p2 = (j - 0.4 * cell_w, i)

    elif action_idx == 2:
        p1 = (j, i)
        p2 = (j, i - 0.4 * cell_h)
        
    elif action_idx == 3:
        p1 = (j, i)
        p2 = (j, i + 0.4 * cell_h)

    pygame.draw.line(renderer.screen, renderer.colors[action_idx], p1, p2, int(cell_w / 25))
    pygame.draw.circle(renderer.screen, renderer.colors[action_idx], p2, cell_w / 15)

def label_y(renderer, coordinate):
    return coordinate[1] - 3/8 * renderer.CELL_SIZE[1]

def label_x(renderer, coordinate):
    return coordinate[0] - renderer.CELL_SIZE[0] / 2 + renderer.CELL_SIZE[1] / 8

def render_player(screen):
    if agent.record:
        for state, _, _, state_after, _ in agent.record[:-1]:
            pygame.draw.line(screen, 'red', (
                state[0] * cell_w + cell_w / 2, 
                state[1] * cell_h + cell_h / 2
                ), (
                    state_after[0] * cell_w + cell_w / 2, 
                    state_after[1] * cell_h + cell_h / 2
                    ), 3)

    pygame.draw.circle(screen, 'red', agent.pos, cell_w / 6, 0)

def right(agent): agent.pos[0] += cell_w
def left(agent): agent.pos[0] -= cell_w
def up(agent): agent.pos[1] -= cell_h
def down(agent): agent.pos[1] += cell_h

# agent setup
goal = (W - cell_w / 2, H - cell_h / 2)

agent = Agent(pos=[cell_w / 2, cell_h / 2])

env = Environment(
    world_map, 
    lazy_render=lazy_render,
    BLOCKED_SPACE=blocked_code, 
    CELL_SIZE=[cell_w, cell_h]
)

agent.render_f = render_player
agent.limit("pos", ([cell_w / 2, cell_h / 2], [W - cell_w / 2, H - cell_h / 2]))

agent.set_nn(
    nn(
        Flatten(),
        Dense(64, Leaky_ReLU()),
        Dense(64, Leaky_ReLU()),
        Dense(4)
    )
)

renderer = Renderer(
    RES=(W, H),
    init=pygame.init,
    quit=pygame.quit
)

renderer.configuer_debugger(
    figure=arrow_labels,
    info_y=label_y,
    info_x=label_x,
    colors=["orange", "yellow", "purple", "blue"],
    labels=["R", "L", "U", "D"]
)

agent.define_actions(
    right,
    left,
    up,
    down,
    breaklaw_penalty=-1,
    done_f= lambda : np.array_equal(agent.pos, goal),
    fail_f=lambda : False
)

env.state_f = lambda env, agent : np.array(agent.norm_pos)
env.reward_f = lambda env, agent : 0 if agent.done() else -1 # get as quickly as possible!

env.add_agent(agent)
env.add_renderer(renderer)

agent.compile(
    batch_size=32,
    optim=Adam(lr=5e-4),
    cost=MSE, # J(θ) for π(s)
    dcost=None, # ∇ J(θ) for π(s)
    update_tqn_every=200,
    buffer_capcity=10_000,
    record_capcity=20
)

env.run(
    episodes=500,
    gamma=0.99,
    ε_range=(0.9, 0.05),
    ε_clip_ratio=.4,
    fps=1000
)