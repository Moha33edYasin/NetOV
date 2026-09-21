# NetOV

**NetOV** is an experimental reinforcement-learning environment framework built on top of my neural-network framework, **[NetJet](https://github.com/Moha33edYasin/NetJet/)**.

The goal of NetOV is to provide a lightweight way to construct environments, agents, state representations, actions, rewards, and training loops around neural networks without relying on high-level reinforcement-learning frameworks.

> **Status:** Experimental / Work in Progress  
> * The agent did learn how to solve discrete maze and breakout game (refer to `maze_solver.py` and `breakout.py`to experiment with the actual code).

---

## Relationship with NetJet

NetOV does not implement its neural-network functionality independently. Instead, it builds an RL-oriented interface on top of NetJet.

```text
NetOV
 ├── Environment
     ├─── State
     ├─── Reward system
 ├── Renderer
 ├── Object
 ├── Agent
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
The agent gets always `-1` until it hit the goal (the red square), where no reward will be given. 

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
We use `1` as a unique code that will help us build faster, and it will be a blocked space, meaning the agent cannot move to.  

```python
_ = 1
blocked_code = [_]
```
To render these static objects (which will refer to as lazy objects), we make a dictonary that contain all unique codes and map them to a custom render function.  

```python
lazy_render = {
    _ : lambda screen, obj: pygame.draw.rect(screen, "red", [*obj.pos, obj.width-1, obj.height-1]),
}
```

> [!NOTE]
> NetOV `Renderer` currently use Pygame as the its main rendering engine. If you used NetOV `Renderer`, you have to use Pygame when building your rendering functions.

Next, we make a simple grid to low represent the game board.

```python
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
```
> [!Note]
> You can experiment with any layout you want. This is only an example.

Then, we calculate cell width `rect_w` and height `rect_h` as well as the peddel width `platform_width` and height `platform_height`.

```python
rect_w = W / (len(world_map[0]))
rect_h = H / (len(world_map))
platform_height = 0.5 * rect_h
platform_width = 2 * rect_w
```

**The goal** is to **hit all the blocks** with the ball by moving the peddel so the ball bounce off toward them.   


## Core Elements

### Environment

`Environment` manages the simulation itself.

It is responsible for things such as:

* The world representation
* Live environment objects (in this case, the ball)
* Lazy environment objects (static ones, like the blocks)
* Agents management.
* Rendering (if renderer is provided. NetOV has built-in `Renderer` class that uses Pygame)
* Credit Assignment
* Episode execution

To define environment, you pass the map, lazy rendering dictonary, blocked space, and cell size:  

```python
env = Environment(
    world_map,
    lazy_render=lazy_render,
    BLOCKED_SPACE=blocked_code,
    CELL_SIZE=[rect_w, rect_h]
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
    episodes=80,
    gamma=.99, # discount on future reward
    ε_range=(.9, .05),
    ε_clip_ratio=.75, # after what percentage of the episodes ε fall to the end of the interval (in this case, 0.05)
    fps=FPS
)
```

---

### Agent

An `Agent` represents the learning entity interacting with the environment.

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
    pos=[len(world_map[0]) * rect_w / 2, H - rect_h],
    width=platform_width, 
    height=platform_height
)
```
You can bound certain properties of the agent. In our breakout example, we write:

```python
agent.limit("pos", ([0, H - rect_h], 
                    [W - platform_width, H - rect_h]))
```


### Object
We define our ball. (note: you can put any attribute you need to store when initializing `Object` or `Agent`, you think of it as another way to construct your python class)

```python
ball = Object(
    pos=[W / 2, H / 2],
    radius=rect_w / 5,
    speed=FPS * rect_w * 0.35,
    direction=random_direction,
)
```

Its movement is updated independently of the agent.

---

### State Representation

This Breakout experiment uses the paddle's normalized position together with ball normalized position, direction, and live blocks' normalized positions.

All these are combined into a single flatten array that will be passed
to the agent learning system.

```python
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
```
We inform the environement with our definition, simply by writing:
```python
env.state_f = capture_state
```
This is deliberately kept lightweight (no CNN), as I was curious about knowing what could MLP perform in this scenario.

---

### Actions

The agent currently has two actions (right and left):

```python
agent.define_actions(
    peddel_right,
    peddel_left,
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
        Dense(2)
    )
)
```

The final layer produces two values, corresponding to the two possible actions.

Compiling the network for our current breakout experiment:

```python
agent.compile(
    batch_size=32,
    optim=Adam(lr=5e-4),
    cost=MSE,
    dcost=None,
    update_tqn_every=200,
    buffer_capcity=10_000,
    record_capcity=20
)
```

This experiment uses experience replay, a target-network update mechanism, and an epsilon-based exploration strategy.  

---

## Reward System
The agent will get `-2` if it `fail` (aka. the ball fall off); `-1` for trying cross screen boundaries, `+1` if the ball bounce of the peddel, `+2` if the ball collided with a block.

The general reward function is discourage failing:  

```python
env.reward_f = lambda env, agent: -2 if agent.fail() else 0
```
while `breaklaw_penalty` in `agent.define_actions` is set to `-1`.  
And finally, when we update the ball, we grant reward to desirable hits:  

```python
def ball_update(dt):
    if dt >= 0.1: dt = 0.001
    ...

    ''' The peddel hit the ball '''
    if ball.collide(agent, offset_down=-agent.height / 2, offset_right=ball.radius, offset_left=ball.radius):
        agent.grant(1) # grant a reward of +1 for hitting the ball
        ...

    ''' The ball hit the blocks '''
    if ball.collide(block):
        agent.grant(2) # grant the agent +2 points for hitting a block 
        ...
        env.kill_lazy_object(block) # remove the block from the running environment 
```
To inform the ball object with our update function above, we set:  

```python
ball.update_f = ball_update
```
---

## Physics and Environment Logic
Collision handling determines whether the ball:

* Reaches a screen boundary
* Hits the paddle
* Hits a block
* Changes direction
* Removes a block from the environment

Blocks are removed after successful collision:

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

The current prototype also includes a debugging overlay that visualizes the agent's available actions in all possible states based on their `Q` return, if there is a restriction on the agent's, subject to debugging, position. Otherwise, it visualize the qaulity of the available actions in the current state:

```python
renderer.configuer_debugger(
    figure=arrow_labels, # function to draw figure (here, it an arrow pointing to most likely action to accur)
    info_y=label_y, # function of y coordinate of the info written with respect to the position of an agent in a certain state 
    info_x=label_x, # function of x coordinate of the info written with respect to the position of an agent in a certain state
    colors=["orange", "purple"], # line colors (also affect the figure color)
    labels=["R", "L"] # info lables
)
```

> [!NOTE]  
> This is useful for inspecting what the agent is choosing during training rather than treating the learning process as a black box.  
> Inspecting is done by pressing the `SPACE` bar while the simulation is running.  
> For multiple agents, you can debug any agent by pressing keyboard key corresponding to that agent index, and then `SPACE` to inspect.

---

## Why I Built NetOV

NetOV is an extension of my work on NetJet.

While developing neural networks from scratch, I wanted to explore reinforcement learning without immediately moving to a high-level RL framework. As I am enginnering the environment interface, I experiment with the entire pipeline:

```text
Environment
     ↓
State representation
     ↓
Neural network
     ↓
Action
     ↓
Environment transition
     ↓
Reward
     ↓
Learning update
     ↺
```

This project is my personal research and engineering project: I am testing how a general-purpose neural-network framework can be extended into an environment for reinforcement-learning experiments.

---

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