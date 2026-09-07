import pygame
from markov import *


W, H = 900, 600
FPS = 10000
_ = 1

blocked_code = [_]
lazy_render = {
    _ : (lambda screen, obj: pygame.draw.rect(screen, "red", [*obj.pos, obj.width-1, obj.height-1])),
}

world_map = [
    [0, _, _, _, _, _, _, _, _, _, _, _, _, 0],
    [_, _, _, _, _, _, _, _, _, _, _, _, _, _],
    [0, _, _, _, _, _, _, _, _, _, _, _, _, 0],
    [0, _, _, _, _, _, _, _, _, _, _, _, _, 0],
    [0, 0, 0, _, _, _, _, _, _, _, _, 0, 0, 0],
    [0, 0, 0, 0, _, _, _, _, _, _, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

rect_w = W / (len(world_map[0]))
rect_h = H / (len(world_map))
platform_height = 0.5 * rect_h
platform_width = 2 * rect_w

# ********** debugging *************
def arrow_labels(renderer, action_idx, state):
    cell_w, cell_h = renderer.CELL_SIZE
    j = state[0]

    if action_idx == 0:
        p1 = (j * cell_w + cell_w / 2, renderer.H - cell_h / 2),
        p2 = (j * cell_w + .9 * cell_w, renderer.H - cell_h / 2)

    elif action_idx == 1:
        p1 = (j * cell_w + cell_w / 2, renderer.H - cell_h / 2),
        p2 = (j * cell_w + .1 * cell_w, renderer.H - cell_h / 2)

    pygame.draw.line(renderer.screen, renderer.colors[action_idx], p1, p2, int(cell_w / 25))
    pygame.draw.circle(renderer.screen, renderer.colors[action_idx], p2, cell_w / 15)

def label_y(renderer, state):
    return renderer.H - renderer.CELL_SIZE[1] + renderer.CELL_SIZE[1] / 8

def label_x(renderer, state):
    return state[0] * renderer.CELL_SIZE[0] + renderer.CELL_SIZE[0] / 8

# ********* reward system **********
def pong_reward(agent):
    n_hit = sum([ball.collide(obj) for obj in env.live_lazy_objects])

    if n_hit:
        return n_hit + 1

    return int(ball.collide(agent))

# ******** ball **********
def ball_update(dt):
    if dt > 1: dt = 0.1
    ball.pos[0] += ball.direction[0] * ball.velocity[0] * dt
    ball.pos[1] += ball.direction[1] * ball.velocity[1] * dt

    # check bounds crossing
    if ball.pos[0] <= 0:
        ball.pos[0] = 0
        ball.direction[0] *= -1 
    if ball.pos[0] >= W - ball.radius:
        ball.pos[0] = W - ball.radius
        ball.direction[0] *= -1 
    if ball.pos[1] <= 0 :
        ball.pos[1] = 0
        ball.direction[1] *= -1 
        
    if ball.collide(agent, offset_down=-agent.height / 2, offset_right=-ball.radius / 2, offset_left=ball.radius / 2):
        ball.direction[1] *= -1
        ball.pos[1] = agent.pos[1] - ball.radius
        return
    
    for obj in env.live_lazy_objects:
        if ball.collide(obj):
            dx = ball.pos[0] - obj.pos[0] - obj.width / 2
            dy = ball.pos[1] - obj.pos[1] - obj.height / 2
            if abs(dx) > abs(dy):
                ball.direction[0] *= -1
                if dx > 0:
                    ball.pos[0] = obj.pos[0] + obj.width + ball.radius + 1
                else:
                    ball.pos[0] = obj.pos[0] - ball.radius - 1
            else:
                ball.direction[1] *= -1
                if dy > 0:
                    ball.pos[1] = obj.pos[1] + obj.height + ball.radius + 1
                else:
                    ball.pos[1] = obj.pos[1] - ball.radius - 1
            
            env.kill_lazy_object(obj)
            break

def ball_render(screen):
    pygame.draw.circle(screen, "green", ball.pos, ball.radius)

def random_direction(rng = np.random.default_rng()):
    dirx = rng.uniform(-1 / np.sqrt(2), 1 / np.sqrt(2))
    return [dirx, np.sqrt(1 - dirx * dirx)]

# ******** agent **********
def peddel_render(screen):
    pygame.draw.rect(screen, "blue", [*agent.pos, agent.width, agent.height])


# agent setup
ball = Object(
    pos=[W / 2, H / 2],
    radius=rect_w / 5,
    velocity=[FPS/500 * rect_w, FPS/500 * rect_h],
    direction=random_direction,
)

agent = Agent(
    start_state=[int(len(world_map[0]) / 2)], 
    width=platform_width, 
    height=platform_height
)

env = Environment(
    world_map,
    LAZY_OBJXXT=lazy_render,
    BLOCKED_SPACE=blocked_code,
    CELL_SIZE=[rect_w, rect_h]
)

renderer = Renderer(
    RES=(W, H),
    init=pygame.init, 
    quit=pygame.quit
)

ball.set_update(ball_update)
ball.set_renderer(ball_render)


agent.set_renderer(peddel_render)
agent.interpet_state(lambda agent : (agent.state[0] * rect_w, H - rect_h), "pos")
agent.lower_bound_state(0)
agent.upper_bound_state(len(world_map[0]))
agent.track(ball, "pos", concat=True)

agent.set_nn(
    nn(
        Flatten(),
        Dense(64, Leaky_ReLU()),
        Dense(64, Leaky_ReLU()),
        Dense(2)
    )
)

agent.define_actions(
    lambda s : np.array([s[0] + 1]), # right
    lambda s : np.array([s[0] - 1]), # left
    done_f= lambda : len(env.live_lazy_objects) == 0,
    fail_f=lambda : ball.pos[1] > renderer.H - ball.radius * 0.5
)


env.define_reward(
    lambda agent, a : pong_reward(agent),  # get as quickly as possible!
    lambda agent, a : -1
)

env.add_object(ball)
env.add_agent(agent)
env.add_renderer(renderer)

renderer.configuer_debugger(
    figure=arrow_labels,
    info_y=label_y,
    info_x=label_x,
    colors= ["orange", "purple"],
    labels=["R", "L"]
)

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
    episodes=10,
    gamma=0.99,
    ε_range=(1, 0.05),
    ε_clip_ratio=1,
    fps=FPS
)