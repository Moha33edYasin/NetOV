# NetOV

**NetOV** is an experimental reinforcement-learning environment framework built on top of my neural-network framework, **[NetJet](../NetJet)**.

The goal of NetOV is to provide a lightweight way to construct environments, agents, state representations, actions, rewards, and training loops around neural networks without relying on high-level reinforcement-learning frameworks.

> **Status:** Experimental / Work in Progress
> The agent did learn how to solve discrete maze environment.
> The current Breakout environment is a prototype used to test the framework's abstractions and reinforcement-learning workflow. The agent does **not** reliably learn Breakout yet.

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

## Current Experiment: Breakout

The current prototype uses Pygame to create a simplified Breakout-style environment.

The environment contains:

* A controllable paddle
* A moving ball
* Breakable blocks
* A discrete action space
* A reward function based on interactions with the ball and blocks
* Terminal conditions for winning and failing
* A neural-network-driven agent
* Experience replay and target-network updates

The current experiment is primarily intended to test whether NetOV can provide a clean interface between:

**environment → state → neural network → action → reward → training**

rather than being a finished game-playing system.

---

## Example Environment

The game board is represented using a simple grid:

```python
world_map = [
    [0, _, _, _, _, _, _, _, _, _, _, _, _, 0],
    [_, _, _, _, _, _, _, _, _, _, _, _, _, _],
    [0, _, _, _, _, _, _, _, _, _, _, _, _, 0],
    [0, _, _, _, _, _, _, _, _, _, _, _, _, 0],
    [0, 0, 0, _, _, _, _, _, _, _, 0, 0, 0],
    [0, 0, 0, 0, _, _, _, _, _, _, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...
]
```

The map is converted into environment objects through `Environment`.

```python
env = Environment(
    world_map,
    LAZY_OBJXXT=lazy_render,
    BLOCKED_SPACE=blocked_code,
    CELL_SIZE=[rect_w, rect_h]
)
```

This allows the environment representation and rendering logic to remain separate.

---

## Core Abstractions

### `Environment`

`Environment` manages the simulation itself.

It is responsible for things such as:

* The world representation
* Live environment objects
* Agents
* Rendering
* Rewards
* Episode execution
* Terminal conditions

Objects can be added dynamically:

```python
env.add_object(ball)
env.add_agent(agent)
env.add_renderer(renderer)
```

The environment can then run multiple episodes:

```python
env.run(
    episodes=10,
    gamma=0.99,
    ε_range=(1, 0.05),
    ε_clip_ratio=1,
    fps=FPS
)
```

---

### `Agent`

An `Agent` represents the learning entity interacting with the environment.

The agent can define:

* Its initial state
* Physical representation
* State interpretation
* State boundaries
* Tracked environment objects
* Neural network
* Available actions
* Terminal conditions
* Optimizer and training configuration

For example:

```python
agent = Agent(
    start_state=[int(len(world_map[0]) / 2)],
    width=platform_width,
    height=platform_height
)
```

The agent's discrete position state is converted into a physical screen position:

```python
agent.interpet_state(
    lambda agent: (agent.state[0] * rect_w, H - rect_h),
    "pos"
)
```

The agent can also observe another object's properties:

```python
agent.track(ball, "pos", concat=True)
```

This allows the ball's position to become part of the information available to the learning system.

---

### State Representation

The current Breakout experiment uses the paddle's state together with information tracked from the ball.

The framework allows an agent's internal state and tracked object information to be combined before being passed through the neural network.

This is deliberately kept lightweight so that different state representations can be experimented with without rewriting the environment itself.

---

### Actions

The agent currently has two actions:

```python
agent.define_actions(
    lambda s: np.array([s[0] + 1]),  # right
    lambda s: np.array([s[0] - 1]),  # left
    done_f=lambda: len(env.live_lazy_objects) == 0,
    fail_f=lambda: ball.pos[1] > renderer.H - ball.radius * 0.5
)
```

The output of the neural network therefore corresponds to the two available actions:

```text
0 → Move Right
1 → Move Left
```

The agent also defines separate functions for successful completion and failure.

---

## Neural Network

The agent uses a small neural network created through NetJet:

```python
nn(
    Flatten(),
    Dense(64, Leaky_ReLU()),
    Dense(64, Leaky_ReLU()),
    Dense(2)
)
```

The final layer produces two values, corresponding to the two possible actions.

The current experiment uses:

```python
agent.compile(
    optim=Adam(lr=5e-4),
    cost=MSE,
    dcost=None,
    batch_size=32,
    update_tqn_every=200,
    buffer_capcity=10_000,
    record_capcity=20
)
```

This experiment uses experience replay, a target-network update mechanism, and an epsilon-based exploration strategy.

The underlying neural-network operations are provided by NetJet rather than an external deep-learning library.

---

## Reward Function

The reward function is currently designed to encourage interaction with blocks and the paddle:

```python
def pong_reward(agent):
    n_hit = sum(
        [ball.collide(obj) for obj in env.live_lazy_objects]
    )

    if n_hit:
        return n_hit + 1

    return int(ball.collide(agent))
```

The environment provides separate reward values for the corresponding conditions:

```python
env.define_reward(
    lambda agent, a: pong_reward(agent),
    lambda agent, a: -1
)
```

This reward design is still experimental and is likely to change as the learning behavior of the agent is investigated.

---

## Physics and Environment Logic

The ball is represented as an environment object:

```python
ball = Object(
    pos=[W / 2, H / 2],
    radius=rect_w / 5,
    velocity=[FPS / 500 * rect_w, FPS / 500 * rect_h],
    direction=random_direction,
)
```

Its movement is updated independently of the agent.

Collision handling determines whether the ball:

* Reaches a screen boundary
* Hits the paddle
* Hits a block
* Changes direction
* Removes a block from the environment

For example, blocks are removed after collision:

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
ball.set_renderer(ball_render)
agent.set_renderer(peddel_render)
```

The current prototype also includes a debugging overlay that visualizes the agent's available actions:

```python
renderer.configuer_debugger(
    figure=arrow_labels,
    info_y=label_y,
    info_x=label_x,
    colors=["orange", "purple"],
    labels=["R", "L"]
)
```

This is useful for inspecting what the agent is choosing during training rather than treating the learning process as a black box.
Inspecting is done by pressing the SPACE bar while the simulation is running.
---

## Why I Built NetOV

NetOV is an extension of my work on NetJet.

While developing neural networks from scratch, I wanted to explore reinforcement learning without immediately moving to a high-level RL framework. Building the environment interface myself lets me experiment with the entire pipeline:

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

This project is therefore also an engineering experiment: I am testing how a general-purpose neural-network framework can be extended into an environment for reinforcement-learning experiments.

---

## Current Limitations

NetOV is still under active development.

The current Breakout experiment has several unresolved issues:

* The agent does not yet learn a reliable Breakout policy.
* The reward design is still experimental.
* Collision handling and physics are still being refined.
* State representation can be improved.
* Training performance is not yet optimized.
* The RL abstractions may change as more environments are implemented.

The current goal is **not** to present a finished RL library, but to build and evaluate the underlying abstractions through increasingly complex experiments.

---

## Planned Direction

Future experiments are intended to improve the framework itself as well as its learning capabilities.

Potential directions include:

* Improving the Breakout state representation
* Refining reward shaping
* Experimenting with alternative RL formulations
* Improving collision and physics handling
* Testing additional environments
* Reducing training overhead
* Expanding the agent/environment interface
* Investigating more flexible state-action representations

---

## Dependencies

The current prototype uses:

* Python
* NumPy
* Pygame
* NetJet
* C++ components used internally by NetJet where applicable

NetOV itself provides the environment and reinforcement-learning abstractions, while NetJet provides the underlying neural-network functionality.

---

## Project Status

**Experimental / In Development**

NetOV is currently a personal research and engineering project.

The Breakout implementation should be considered a **prototype and framework test**, not a completed reinforcement-learning benchmark. Its main purpose at this stage is to demonstrate the architecture and to provide a foundation for further experimentation.
