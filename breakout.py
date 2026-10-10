from markov import *
from breakout_settings import *

# ********** debugging *************
def arrow_labels(renderer, action_idx, coordinate):
    if action_idx == 0:
        p1 = (coordinate[0] + renderer.cell_width / 2, renderer.H - renderer.cell_height / 2),
        p2 = (coordinate[0] + .9 * renderer.cell_width, renderer.H - renderer.cell_height / 2)

    if action_idx == 1:
        p1 = (coordinate[0] + renderer.cell_width / 2, renderer.H - renderer.cell_height / 2),
        p2 = (coordinate[0] + renderer.cell_width / 2, renderer.H - renderer.cell_height / 2)

    elif action_idx == 2:
        p1 = (coordinate[0] + renderer.cell_width / 2, renderer.H - renderer.cell_height / 2),
        p2 = (coordinate[0] + .1 * renderer.cell_width, renderer.H - renderer.cell_height / 2)

    pygame.draw.line(renderer.screen, renderer.colors[action_idx], p1, p2, int(renderer.cell_width / 25))
    pygame.draw.circle(renderer.screen, renderer.colors[action_idx], p2, renderer.cell_width / 15)

# ********* state system **********
def capture_state(env, agent):
    return np.concatenate(
        [
            [agent.pos[0] / (W - agent.width), agent.width / PADDEL_W],
            np.multiply(ball.pos, [1 / (W - ball.radius), 1 / H]),
            np.multiply(ball.direction, ball.speed / FAST_SPEED3),
            env.live_map.ravel() / env.map.max()
        ]
    )

# ******** ball **********
def ball_update(dt):
    if dt >= 0.1: dt = 0.01

    if ball.n_paddel_hits == 4:
        ball.speed = max(FAST_SPEED1, ball.speed)

    if ball.n_paddel_hits == 12:
        ball.speed = max(FAST_SPEED2, ball.speed)

    prev_pos = ball.pos.copy()
    ball.pos[0] += ball.direction[0] * ball.speed * dt
    ball.pos[1] += ball.direction[1] * ball.speed * dt

    # check bounds crossing
    if ball.pos[0] <= ball.radius:
        ball.pos[0] = ball.radius
        ball.direction[0] *= -1

    elif ball.pos[0] >= W - ball.radius:
        ball.pos[0] = W - ball.radius
        ball.direction[0] *= -1

    if ball.pos[1] <= ball.radius :
        ball.pos[1] = ball.radius
        ball.direction[1] *= -1
        agent.width = PADDEL_W / 2
        agent.track("pos", ([0, H - CELL_H], 
                    [W - PADDEL_W / 2, H - CELL_H]))

    if ball.collide(agent, dt=dt, offset_down=-ball.radius):
        ball.n_paddel_hits += 1
        agent.grant(0.1)

        distance_x = ball.pos[0] - agent.pos[0] - agent.width / 2

        # Hitting a segment close to edges will raise the ball at a wider angle
        # This should decrease the determinism in the game.
        x_sign = 1 if ball.direction[0] > 0 else -1
        if 0 <= abs(distance_x) < agent.width / 8:
            ball.direction[0] = x_sign * 1 / np.sqrt(2) * 0.35
        elif agent.width / 8 <= abs(distance_x) < agent.width / 4:
            ball.direction[0] = x_sign * 1 / np.sqrt(2) * 0.65
        elif agent.width / 4 <= abs(distance_x) < 3/8 * agent.width:
            ball.direction[0] = x_sign * 1 / np.sqrt(2) * 0.95
        else:
            ball.direction[0] = x_sign * 1 / np.sqrt(2) * 1.25

        ball.direction[1] = -np.sqrt(1 - ball.direction[0] * ball.direction[0])

        ball.pos[1] = agent.pos[1] - ball.radius - 1
        return

    target_brick = ball.multi_collide(env.live_lazy_objects, dt=dt) #sequared_distances[min(sequared_distances.keys())]
    if target_brick != None:
        agent.grant(8 - (target_brick.norm_pos[1] - int(NUM_ROW / 6)))

        # change speed and paddel width if the ball hit the first two rows on the top
        norm_y = target_brick.pos[1] / CELL_H
        if norm_y <= int(NUM_ROW / 6) + 1:
            ball.speed = max(FAST_SPEED3, ball.speed)

        dx = prev_pos[0] - target_brick.pos[0] - target_brick.width / 2
        dy = prev_pos[1] - target_brick.pos[1] - target_brick.height / 2

        if abs(dy) > target_brick.height / 2:
            if ball.direction[1] < 0:
                ball.direction[1] = abs(ball.direction[1])
                ball.pos[1] = target_brick.pos[1] + target_brick.height + ball.radius + 1
            else:
                ball.direction[1] = -abs(ball.direction[1])
                ball.pos[1] = target_brick.pos[1] - ball.radius - 1

        elif abs(dx) > target_brick.width / 2:
            if ball.direction[0] < 0:
                ball.direction[0] = abs(ball.direction[1])
                ball.pos[0] = target_brick.pos[0] + target_brick.width + ball.radius + 1
            else:
                ball.direction[0] = -abs(ball.direction[1])
                ball.pos[0] = target_brick.pos[0] - ball.radius - 1
        
        env.kill_lazy_object(target_brick)

def ball_render(screen):
    pygame.draw.circle(screen, "green", ball.pos, ball.radius)

def random_direction(rng = np.random.default_rng()):
    dirx = rng.uniform(-1 / np.sqrt(2), 1 / np.sqrt(2))
    return [dirx, np.sqrt(1 - dirx * dirx)]

# ******** paddel **********
def paddel_right(agent):
    agent.pos[0] += CELL_W

def paddel_idle(agent): ...

def paddel_left(agent):
    agent.pos[0] -= CELL_W

def paddel_render(screen):
    pygame.draw.rect(screen, "blue", [*agent.pos, agent.width, agent.height])

# agent setup
ball = Object(
    pos=[W / 2, H / 2],
    radius=BALL_RADIUS,
    speed=NORMAL_SPEED,
    direction=random_direction,
    n_paddel_hits=0
)

agent = Agent(
    pos=[NUM_COL * CELL_W / 2, H - CELL_H],
    width=PADDEL_W,
    height=PADDEL_H
)

env = Environment(
    WORLD_MAP,
    require_live_map=True,
    lazy_code=BLOCED_CODE,
    lazy_render=LAZY_RENDERS,
    blocked_space=BLOCED_CODE,
    cell_size=[CELL_W, CELL_H]
)

renderer = Renderer(
    RES=(W, H),
    init=pygame.init,
    quit=pygame.quit,
)

ball.update_f = ball_update
ball.render_f  = ball_render
agent.render_f = paddel_render

agent.track("pos", ([0, H - CELL_H], 
                    [W - PADDEL_W, H - CELL_H]))

agent.set_nn(
    nn(
        Flatten(),
        Dense(64, Leaky_ReLU()),
        Dense(64, Leaky_ReLU()),
        Dense(3)
    )
)

agent.define_actions(
    paddel_right,
    paddel_idle,
    paddel_left,
    breaklaw_penalty=-0.1,
    done_f=lambda : len(env.live_lazy_objects) == 0,
    fail_f=lambda : ball.pos[1] > H
)

env.state_f = capture_state
env.reward_f = lambda env, agent: -5 if agent.fail() else (10 if agent.done() else 0)

env.add_object(ball)
env.add_agent(agent)
env.add_renderer(renderer)

renderer.configuer_debugger(
    figure=arrow_labels,
    colors= ["orange", "black", "purple"]
)

agent.compile(
    batch_size=64,
    optim=Adam(lr=5e-4),
    cost=MSE, # J(θ) for π(s)
    dcost=None, # ∇ J(θ) for π(s)
    update_tqn_every=1000,
    buffer_capcity=100_000
)

env.run(
    episodes=65_000, # Experiement 1: (60_000)
    gamma=0.99,
    ε_range=(1, 0.05),
    ε_clip_ratio=0.9995,
    fps=FPS
)
