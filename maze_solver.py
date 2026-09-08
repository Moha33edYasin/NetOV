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

def arrow_labels(renderer, action_idx, state):
    j, i = state
    cell_w, cell_h = renderer.CELL_SIZE

    if action_idx == 0:
        p1 = (j * cell_w + cell_w / 2, i * cell_h + cell_h / 2),
        p2 = (j * cell_w + .9 * cell_w, i * cell_h + cell_h / 2)

    elif action_idx == 1:
        p1 = (j * cell_w + cell_w / 2, i * cell_h + cell_h / 2),
        p2 = (j * cell_w + .1 * cell_w, i * cell_h + cell_h / 2)

    elif action_idx == 2:
        p1 = (j * cell_w + cell_w / 2, i * cell_h + cell_h / 2)
        p2 = (j * cell_w + cell_w / 2, i * cell_h + .1 * cell_h)
        
    elif action_idx == 3:
        p1 = (j * cell_w + cell_w / 2, i * cell_h + cell_h / 2)
        p2 = (j * cell_w + cell_w / 2, i * cell_h + .9 * cell_h)

    pygame.draw.line(renderer.screen, renderer.colors[action_idx], p1, p2, int(cell_w / 25))
    pygame.draw.circle(renderer.screen, renderer.colors[action_idx], p2, cell_w / 15)

def label_positioning(renderer, state):
    return state[1] * renderer.CELL_SIZE[0] + renderer.CELL_SIZE[1] / 8

def label_spacing(renderer, state):
    return state[0] * renderer.CELL_SIZE[0] + renderer.CELL_SIZE[1] / 8

def render_player(screen):
    if agent.record:
        pygame.draw.line(screen, 'red', (
            agent.rel_start[0] * cell_w + cell_w / 2, 
            agent.rel_start[1] * cell_h + cell_h / 2
            ),
            (
                agent.record[0][0][0] * cell_w + cell_w / 2, 
                agent.record[0][0][1] * cell_h + cell_h / 2
            ), 3)

        for state, _, _, state_after, _ in agent.record[:-1]:
            pygame.draw.line(screen, 'red', (
                state[0] * cell_w + cell_w / 2, 
                state[1] * cell_h + cell_h / 2
                ), (
                    state_after[0] * cell_w + cell_w / 2, 
                    state_after[1] * cell_h + cell_h / 2
                    ), 3)

    pygame.draw.circle(screen, 'red', (
                                    agent.state[0] * cell_w + cell_w / 2,
                                    agent.state[1] * cell_h + cell_h / 2
                                    ), cell_w / 6, 0)

# agent setup
goal = (9, 9)

agent = Agent(start_state=[0, 0])
env = Environment(
    world_map, 
    LAZY_OBJXXT=lazy_render, 
    BLOCKED_SPACE=blocked_code, 
    CELL_SIZE=[cell_w, cell_h]
)

agent.set_renderer(render_player)
agent.interpet_state(lambda agent : (agent.state[0] * cell_w, agent.state[1] * cell_h), "pos")
agent.lower_bound_state([0, 0])
agent.upper_bound_state([len(world_map[0]), len(world_map)])
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
    info_y=label_positioning,
    info_x=label_spacing,
    colors=["orange", "yellow", "purple", "blue"],
    labels=["R", "L", "U", "D"]
)

agent.define_actions(
    lambda s : s + [1, 0], # right
    lambda s : s + [-1, 0], # left
    lambda s : s + [0, -1], # up
    lambda s : s + [0, 1],  # down
    done_f= lambda : np.array_equal(agent.state, goal),
    fail_f=lambda : False
)

env.define_reward(
    lambda agent, a : 0 if agent.done() else -1, # get as quickly as possible!
    lambda agent, a : -2
)

env.add_agent(agent)
env.add_renderer(renderer)

agent.compile(
    optim=Adam(lr=5e-4),
    cost=MSE, # J(θ) for π(s)
    dcost=None, # ∇ J(θ) for π(s)
    batch_size=32,
    update_tqn_every=200,
    buffer_capcity=10_000,
    record_capcity=20
)

env.run(
    episodes=500,
    gamma=0.99,
    ε_range=(1, 0.05),
    ε_clip_ratio=.4,
    fps=1000
)

# ! TRY TO BRANCH NETJET AND ADD INDEPENDENCY FROM THE SEQUENCTIAL NN CLASS.