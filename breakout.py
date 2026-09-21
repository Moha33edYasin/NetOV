import pygame
from markov import *

W, H = 900, 600
FPS = 60
_ = 1

blocked_code = [_]
lazy_render = {
    _ : lambda screen, obj: pygame.draw.rect(screen, "red", [*obj.pos, obj.width-1, obj.height-1]),
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
ball_collision = True

# ********** debugging *************
def arrow_labels(renderer, action_idx, coordinate):
    if action_idx == 0:
        p1 = (coordinate[0] + renderer.cell_width / 2, renderer.H - renderer.cell_height / 2),
        p2 = (coordinate[0] + .9 * renderer.cell_width, renderer.H - renderer.cell_height / 2)

    elif action_idx == 1:
        p1 = (coordinate[0] + renderer.cell_width / 2, renderer.H - renderer.cell_height / 2),
        p2 = (coordinate[0] + .1 * renderer.cell_width, renderer.H - renderer.cell_height / 2)

    pygame.draw.line(renderer.screen, renderer.colors[action_idx], p1, p2, int(renderer.cell_width / 25))
    pygame.draw.circle(renderer.screen, renderer.colors[action_idx], p2, renderer.cell_width / 15)

def label_y(renderer, coordinate):
    return renderer.H - renderer.CELL_SIZE[1] + renderer.CELL_SIZE[1] / 8

def label_x(renderer, coordinate):
    return coordinate[0] + renderer.CELL_SIZE[0] / 8

# ********* state system **********
def capture_state(env, agent):
    blocks_loc = np.array([obj.pos if obj in env.live_lazy_objects else [0.0, 0.0] for obj in env.lazy_objects])
    n_arr = 1 / np.max(env.lazy_loc, axis=0)

    return np.concatenate(
        [
            [agent.pos[0] / (W - agent.width)],
            np.multiply(ball.pos, [1 / (W - ball.radius), 1 / (H - ball.radius * 0.5)]),
            ball.direction,
            np.multiply(blocks_loc, n_arr).ravel()
        ]
    )

# ******** ball **********
def ball_update(dt):
    if dt >= 0.1: dt = 0.001

    prev_pos = ball.pos.copy()
    ball.pos[0] += ball.direction[0] * ball.speed * dt
    ball.pos[1] += ball.direction[1] * ball.speed * dt

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
        
    if ball.collide(agent, offset_down=-agent.height / 2, offset_right=ball.radius, offset_left=ball.radius):
        agent.grant(1)
        ball.direction[1] *= -1
        ball.pos[1] = agent.pos[1] - ball.radius
        return

    sequared_distances = {}
    for obj in env.live_lazy_objects:
        if ball.collide(obj):
            dx = prev_pos[0] - obj.pos[0] - obj.width / 2
            dy = prev_pos[1] - obj.pos[1] - obj.height / 2
            seq_dist = dx * dx + dy * dy
            sequared_distances[seq_dist] = obj

    if sequared_distances:
        agent.grant(2)
        collision_obj = sequared_distances[min(sequared_distances.keys())]

        dx = prev_pos[0] - collision_obj.pos[0] - collision_obj.width / 2
        dy = prev_pos[1] - collision_obj.pos[1] - collision_obj.height / 2
        if abs(dx) >= obj.width / 2:
            ball.direction[0] *= -1
            if dx > 0:
                ball.pos[0] = collision_obj.pos[0] + collision_obj.width + ball.radius + 1
            else:
                ball.pos[0] = collision_obj.pos[0] - ball.radius - 1
        if abs(dy) >= obj.height / 2:
            ball.direction[1] *= -1
            if dy > 0:
                ball.pos[1] = collision_obj.pos[1] + collision_obj.height + ball.radius + 1
            else:
                ball.pos[1] = collision_obj.pos[1] - ball.radius - 1

        env.kill_lazy_object(collision_obj)

def ball_render(screen):
    pygame.draw.circle(screen, "green", ball.pos, ball.radius)

def random_direction(rng = np.random.default_rng()):
    dirx = rng.uniform(-1 / np.sqrt(2), 1 / np.sqrt(2))
    return [dirx, np.sqrt(1 - dirx * dirx)]

# ******** agent **********
def peddel_right(agent):
    agent.pos[0] += rect_w

def peddel_left(agent):
    agent.pos[0] -= rect_w

def peddel_render(screen):
    pygame.draw.rect(screen, "blue", [*agent.pos, agent.width, agent.height])


# agent setup
ball = Object(
    pos=[W / 2, H / 2],
    radius=rect_w / 5,
    speed=FPS * rect_w * 0.35,
    direction=random_direction,
)

agent = Agent(
    pos=[len(world_map[0]) * rect_w / 2, H - rect_h],
    width=platform_width, 
    height=platform_height
)

env = Environment(
    world_map,
    lazy_render=lazy_render,
    BLOCKED_SPACE=blocked_code,
    CELL_SIZE=[rect_w, rect_h]
)

renderer = Renderer(
    RES=(W, H),
    init=pygame.init, 
    quit=pygame.quit
)

ball.update_f = ball_update
ball.render_f  = ball_render
agent.render_f = peddel_render

agent.limit("pos", ([0, H - rect_h], 
                    [W - platform_width, H - rect_h]))

agent.set_nn(
    nn(
        Flatten(),
        Dense(64, Leaky_ReLU()),
        Dense(64, Leaky_ReLU()),
        Dense(2)
    )
)

agent.define_actions(
    peddel_right,
    peddel_left,
    breaklaw_penalty=-1,
    done_f=lambda : len(env.live_lazy_objects) == 0,
    fail_f=lambda : ball.pos[1] > renderer.H - ball.radius * 0.5
)

env.state_f = capture_state
env.reward_f = lambda env, agent: -2 if agent.fail() else 0

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
    batch_size=32,
    optim=Adam(lr=5e-4),
    cost=MSE, # J(θ) for π(s)
    dcost=None, # ∇ J(θ) for π(s)
    update_tqn_every=200,
    buffer_capcity=10_000,
    record_capcity=20
)

env.run(
    episodes=250,
    gamma=0.99,
    ε_range=(0.9, 0.05),
    ε_clip_ratio=0.7,
    fps=FPS
)