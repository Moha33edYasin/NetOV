# NetOV

**NetOV** is an experimental reinforcement-learning environment framework built on top of my neural-network framework, **[NetJet](https://github.com/Moha33edYasin/NetJet/)**.

The goal of NetOV is to provide a lightweight way to construct environments, agents, state representations, actions, rewards, and training loops around neural networks without relying on high-level reinforcement-learning frameworks.

> **Status:** Experimental / Work in Progress  
> * The agent did learn how to solve discrete maze and breakout game (refer to `maze_solver.py` and `breakout.py` to experiment with the actual code).

---

## Relationship with NetJet

NetOV does not implement its neural-network functionality independently. Instead, it builds an RL-oriented interface on top of NetJet.

```text
NetOV
 ├── Environment
     ├─── State
     ├─── General Reward system
     ├─── Objects Management
     ├─── Rendering System
          └─── Renderer
 ├── Object
     ├─── Static property defining
     ├─── Property tracking
     ├─── collision mechanics
 ├── Agent
     ├─── Instantenous reward/penalty recieving. 
     ├─── Action handling
          └── RL training logic
              │
              ▼
            NetJet
       ├── Neural network
       ├── Layers
       ├── Activations
       ├── Optimizers
       └── Loss functions
```

The `markov` module inside NetOV imports NetJet internally, allowing an environment and its agent to use NetJet's neural-network components directly.

For example:

```python
from markov import *

agent.set_nn(
    nn(
        Flatten(),
        Dense(64, Leaky_ReLU()),
        Dense(64, Leaky_ReLU()),
        Dense(2)
    )
)
```

The neural network above is constructed using NetJet components while the surrounding agent, environment, reward, and training logic are handled by NetOV.

---
## Expreiment A: Solving 2D Maze
### Problem Statement:
Here, the agent should find its way out of the constructed environment in the shortest path possible. There 4 possible action: (`right`, `left`, `up`, `down`)

```python
def right(agent): agent.pos[0] += cell_w
def left(agent): agent.pos[0] -= cell_w
def up(agent): agent.pos[1] -= cell_h
def down(agent): agent.pos[1] += cell_h
```

### How reward works?
The environment `env` always give a penalty of `-1` to the agent until it hit the goal (the red square), where no reward will be given. 

```python
env.reward_f = lambda env, agent : 0 if agent.done() else -1
```
If it hit the boundaries of the maze or the walls, it will get an additional `-1` reward, suming to `-2` in total.  

`breaklaw_penalty=-1` in `define_action` specify this additional constraint.

```python
agent.define_actions(
    right,
    left,
    up,
    down,
    breaklaw_penalty=-1,
    done_f= lambda : np.array_equal(agent.pos, goal),
    fail_f=lambda : False
)
```
`done_f` and `fail_f` arguments determine when the agent solved the problem or entirely failed and should restart.

## The Result
  
<img width="898" height="600" alt="20261001-1830-32 7113342" src="https://github.com/user-attachments/assets/7e0c0158-6edd-4178-9c85-aa636dbb1e17" />  
  
  
## Experiment B: Playing Breakout

### Problem Statement
The environment contains:

* A controllable paddle
* A moving ball
* Breakable blocks
* A discrete action space
  
We set the screen resolution to `(900, 600)` and frame rate `FPS` to `60`:  
```python
W, H = 900, 600
FPS = 60
```
Then, we have to specify the blocks.  
We use positive integers as unique codes to help us build the board faster. These codes represent blocked spaces, meaning the agent cannot move into them. 
```python
BLOCED_CODE = [1, 2, 3, 4, 5, 6]
```
> [!NOTE]
> In this Breakout experiment, this is not particularly meaningful, but it makes it clear that the paddle (the agent) cannot move onto the bricks.

To render these static objects (which we will refer to as lazy objects), we create a dictionary that contains all unique codes and maps them to a custom rendering function.

```python
LAZY_RENDERS = {
    1 : lambda screen, obj: pygame.draw.rect(screen, "red", [*obj.pos, obj.width-1, obj.height-1]),
    2 : lambda screen, obj: pygame.draw.rect(screen, "orange", [*obj.pos, obj.width-1, obj.height-1]),
    3 : lambda screen, obj: pygame.draw.rect(screen, "brown", [*obj.pos, obj.width-1, obj.height-1]),
    4 : lambda screen, obj: pygame.draw.rect(screen, "yellow", [*obj.pos, obj.width-1, obj.height-1]),
    5 : lambda screen, obj: pygame.draw.rect(screen, "green", [*obj.pos, obj.width-1, obj.height-1]),
    6 : lambda screen, obj: pygame.draw.rect(screen, "blue", [*obj.pos, obj.width-1, obj.height-1])
}
```

> [!NOTE]  
> `Renderer` currently use Pygame as the its main rendering engine. If you used NetOV `Renderer`, you have to use Pygame when building your rendering functions.

Next, we make a simple grid (`40x20` with `6` rows of brick) to low represent the game board.

```python
world_map = create_board(40, 20, 6)
```

> [!NOTE]
> * You can check the implementation of `create_board` in the `breakout_setting.py`, but it should be straightforward.
> * You can experiment with any layout you want; this is only an example.

Then, we calculate cell width `CELL_W` and height `CELL_H` as well as the paddel width `PADDEL_W` and height `PADDEL_H`.

```python
CELL_W = W / 40
CELL_H = H / 20

PADDEL_H = 0.6 * CELL_H
PADDEL_W = 2 * CELL_W
```

**The goal** is to **hit all the blocks** with the ball by moving the paddle so that the ball bounces toward them.   
We will walk through how to use NetOV to produce the solution together.  

## Core Elements

### Environment

`Environment` manages the simulation.

It is responsible for things such as:

* The world representation
* Live environment objects (in this case, the ball)
* Lazy environment objects (static ones, like the blocks)
* Agents management.
* Rendering (if renderer is provided. NetOV has built-in `Renderer` class that uses Pygame)
* Credit assignment
* Episode execution

To define environment, you pass the map, lazy rendering dictonary, blocked space, and cell size:  

```python
env = Environment(
    WORLD_MAP,
    require_live_map=True,
    lazy_code=BLOCED_CODE, # meaningless here, lol!
    lazy_render=LAZY_RENDERS,
    blocked_space=BLOCED_CODE,
    cell_size=[CELL_W, CELL_H]
)
```

The map is converted into environment *lazy objects* through `Environment`. It allows the environment representation and rendering logic to remain separate.  


Objects and a renderer engine can be added dynamically:
```python
env.add_object(ball)
env.add_agent(agent)
env.add_renderer(renderer)
```

The environment can then run multiple episodes:

```python
env.run(
    episodes=60_000,
    gamma=.99, # discount on future reward
    ε_range=(1, .05),
    ε_clip_ratio=.95, # after what percentage of the episodes ε fall to the end of the interval (in this case, 0.05) or what is called the decay rate.
    fps=FPS
)
```

### Object
We define our ball. (Note: you can add any attributes you need to store when initializing an `Object` or `Agent`; you can think of this as another way to construct a Python class.)

```python
ball = Object(
    pos=[W / 2, H / 2],
    radius=BALL_RADIUS, # this will be used to align the label when rendering
    speed=NORMAL_SPEED, # the first ball speed for the breakout
    direction=random_direction, # random_direction is external function that return random unit vector pointing downward and incline by angle between [-π/4, -3π/4] 
    n_peddel_hits=0
)
```

Its movement is updated independently of the agent.

> [!Note]  
> * Every object should have at least a `pos` argument.  
> * Any property initialized by a function (like `random_direction` above) will call that function everytime the object is resetted.  
> * You can prevent resetting a property by calling `keep` on the object and passing the property's name. If you wish to reset a kept property later, you have to call `init` instead of `reset` on the object. `init` will reset all property regardless of their specifications.   
> * `width`, `height`, and/or `radius` are required for proper collision mechanics and labels rendering.  
> * `velocity` / `speed` with `direction` are recommend. That will help in collision mechanics.  
> * Alternatively, you can use automatic tracking for the position property (using `track` function), and the velocity needed for collision calculations will be calculated automatically. (It is important not to use automatic position tracking if the motion changes repeatedly. Setting `speed` with `direction` or just setting `velocity` is safer).    
   
---

### Agent

An `Agent` is basically an object that represents the learning entity interacting with the environment.

The agent can define:

* Physical representation
* Parameters/Attributes boundaries
* Deep Q-network
* Available actions
* Terminal conditions
* Optimizer and training configuration

Here, we define our agent as follow:
```python
agent = Agent(
    pos=[NUM_COL * CELL_W / 2, H - CELL_H],
    width=PEDDEL_W,
    height=PEDDEL_H
)
```
You can bound certain properties of the agent. In our breakout example, we write:

```python
agent.track("pos", ([0, H - CELL_H], 
                    [W - PEDDEL_W, H - CELL_H]))
```
Also, passing the argument `track_change=True` to `track`, will automatically track property change. We can use this calculate the rate as follow:  

```python
dt = 0.05 # this only an example
agent.track("pos", ([0, H - CELL_H], 
                    [W - PEDDEL_W, H - CELL_H]), track_change=True)

print("Agent velocity:", agent.pos.delta / dt)
```

> [!IMPORTANT]
> Using automatic change track with `change_track=True` will affect how the collision mechanics work. As the older property value is used to calculate the change, it is highly not recommended to set `change_track=True` if the object have a random or frequently changing motion.

## Core Concepts
---
### State Representation

This Breakout experiment uses the paddle's position together with its width, ball position and velocity, and an integar map of the enviroment with the available bricks.  

All these are normalized combined into a single flatten array that will be passed to the agent learning system.  

```python
def capture_state(env, agent):
    return np.concatenate(
        [
            [agent.pos[0] / (W - agent.width), agent.width / PEDDEL_W],
            np.multiply(ball.pos, [1 / (W - ball.radius), 1 / H]),
            np.multiply(ball.direction, ball.speed / FAST_SPEED3),
            env.live_map.ravel() / env.map.max()
        ]
    )
```
We provide the environment `env` with our state definition by writing:

```python
env.state_f = capture_state
```

This is deliberately kept lightweight (no CNN), as I was curious about knowing what could MLP perform in this scenario.

---

### Actions

The agent currently has three actions (right, no action, or left):

```python
agent.define_actions(
    paddel_right,
    paddel_idle, # void function
    paddel_left,
    breaklaw_penalty=-1,
    done_f=lambda : len(env.live_lazy_objects) == 0,
    fail_f=lambda : ball.pos[1] > renderer.H - ball.radius * 0.5
)
```
---

## Neural Network

The agent uses a small neural network created through NetJet:

```python
agent.set_nn(
    nn(
        Flatten(),
        Dense(64, Leaky_ReLU()),
        Dense(64, Leaky_ReLU()),
        Dense(3)
    )
)
```

The final layer produces three values, corresponding to the three possible actions (`paddle_left`, `paddel_idle`, `paddel_right`).

Compiling the network for our current breakout experiment:

```python
agent.compile(
    batch_size=64,
    optim=Adam(lr=5e-4),
    cost=MSE,
    dcost=None,
    update_tqn_every=2000,
    buffer_capcity=10_000
)
```

This experiment uses experience replay, a target-network update mechanism, and an epsilon-based exploration strategy.  

---

## Reward System
The agent receives `-5` if it fail (i.e., the ball falls off); `-0.1` for trying to cross the screen boundaries; `+0.5` if the ball bounces off the paddle; `+2` if the ball hits a brick in the 6th level, with an additional `+1` for hitting a higher level; and `+10` if all bricks are successfully cleared.

The general reward function is discourage failing:  

```python
env.reward_f = lambda env, agent: -5 if agent.fail() else (10 if agent.done() else 0)
```
while `breaklaw_penalty` in `agent.define_actions` is set to `-0.1`.  
And finally, when we update the ball, we grant rewards for desirable hits:  

```python
def ball_update(dt):
    if dt >= 0.1: dt = 0.001
    ...

    ''' The peddel hit the ball '''
    if ball.collide(agent, dt=dt, offset_down=-ball.radius):
        agent.grant(0.5) # grant a reward of +0.5 for hitting the ball
        ...

    ''' The ball hit the blocks '''
    target_brick = ball.multi_collide(env.live_lazy_objects, dt=dt)
    if target_brick != None:
        agent.grant(8 - (target_brick.norm_pos[1] - int(NUM_ROW / 6))) # grant the agent points for hitting a brick in a certain level
        ...
        env.kill_lazy_object(target_brick) # remove the brick from the running environment 
```
To inform the ball object with our update function above, we set:  

```python
ball.update_f = ball_update
```
---

## Collision
You can check if two objects collided by calling `obj1.collide(obj2)` which returns `True` if there is a collision and `False` otherwise. You can pass `dt` if the enviroment is changing with respect to the time and You can adjust the collision box by passing any of `offset_left`, `offset_right`, `offset_top`, and/or `offset_down`.  

There are two types of collisions that I built `collide` to detect:  

1. **By-bound**: Happen when `obj1` touches or enter the collision box area of `obj2`.  

2. **Bypass**: Happen when `obj1` pass through or jump over `obj2`. It uses the intersections between the path of `obj1` and the diagonals of two inscribed quadrilaterals made by the movement of the `obj2`.  In discrete motion (as used by computers), this is extremely important.  

You can also use `obj1.multi_collide(objects)` with multiple objects. You can pass `dt` and the collision-box offsets in the same way as with collide.  

In our breakout expriement, collision handling determines whether the ball:

* Hits the paddle → Changes direction
* Hits a brick → Changes direction → Removes a brick from the environment

### A visual illustration of how collision works
  
<img width="796" height="598" alt="Collision Detection 2" src="https://github.com/user-attachments/assets/3702cda2-1a98-40e7-8196-069a6c353ab8" />  
  
Here, bricks colored in pink indicate detected collisions with the ball, while the green brick is where the actual hit occurs (the nearest collision).  
  
  
Here is how **Bypass** collision works in isolation: (You can see that I've just increased the ball's velocity)  
  
<img width="798" height="598" alt="Collision Detection" src="https://github.com/user-attachments/assets/cdc94f34-8ec0-4a71-8627-bbd54f271571" />  
  
You may notice in our breakout example normal rectangular diagonals are used. This is because we neither define `velocity` / `speed` and `direction` nor we use automatic change tracking (using `track`). Therefore, `multi_collide`/`collide` won't detect the paddle's movement. Instead, it will see isolated positions (snapshots) of the paddle and use the diagonals of the paddle.
  
Bricks are removed after successful collision by this line:

```python
env.kill_lazy_object(obj)
```

This makes the environment dynamic: the set of available objects changes during an episode.

---

## Rendering and Debugging

NetOV separates simulation logic from rendering.

The framework provides a `Renderer` object:

```python
renderer = Renderer(
    RES=(W, H),
    init=pygame.init, 
    quit=pygame.quit
)
```

Objects and agents can define their own rendering functions:

```python
ball.render_f = ball_render
agent.renderer_f = peddel_render
```
> [!NOTE]
> * The current prototype also includes a debugging overlay that visualizes the agent's available actions in all possible states based on their `Q` return, if there is a restriction via `track` on the agent's position subject to debugging. Otherwise, it visualizes the Q of the available actions in the current state.  
> * The net reward will be shown near each agent.  
> * The hyperparamter (ε and gamma) will be shown in the top-left corner of the screen.  

```python
renderer.configuer_debugger(
    figure=arrow_labels, # function to draw figure (here, it an arrow pointing to direction of the most likely next action)
    info_y=label_y, # function of y coordinate of the info written with respect to the position of an agent in a certain state 
    info_x=label_x, # function of x coordinate of the info written with respect to the position of an agent in a certain state
    colors=["orange", "purple"], # line colors (also affect the figure color)
    labels=["R", "L"] # info lables
)
```

> [!NOTE]  
> * This is useful for inspecting what the agent is choosing during training rather than treating the learning process as a black box.
> * Inspecting is done by pressing the `SPACE` bar while the simulation is running.  
> * For multiple agents, you can debug any agent by pressing keyboard key corresponding to that agent index (e.g. `1` for the first agent, and so on), and then `SPACE` to inspect.

---

## The Result
<img width="800" height="598" alt="20261003-1030-14 5205950" src="https://github.com/user-attachments/assets/1d13cdc2-daca-4136-b1a3-cbfd2ec82a12" />


## One Important Caveat
NetOV is still under active development.

The current goal is **not** to present a finished RL library, but to build and evaluate the underlying abstractions through increasingly complex experiments.

---

## Dependencies

The current prototype uses:  

* Python
* NumPy
* Pygame
* NetJet

**NetOV** itself provides the environment and reinforcement-learning abstractions, while **NetJet** provides the background neural-network functionality.  
