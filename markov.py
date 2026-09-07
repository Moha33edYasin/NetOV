import pygame
import numpy as np

from netjet.Layers import *
from netjet.models import *
from netjet.methods import *
from time import perf_counter


class Object:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for attr, val in kwargs.items():
            if callable(val):
                setattr(self, attr, val)
            else:
                setattr(self, attr, deepcopy(val))

    def reset(self):
        for attr, val in self.kwargs.items():
            if callable(val):
                setattr(self, attr, val())
            else:
                setattr(self, attr, deepcopy(val))
            
        if hasattr(self, "pos"):
            self.pos = self.kwargs["pos"].copy()

    def set_update(self, update_f):
        self.update = update_f

    def set_renderer(self, render_f):
        self.render = render_f

    # def collide(self, obj, offset_left=0, offset_right=0, offset_top=0, offset_down=0):

    #     if hasattr(self, "radius"):
    #         radial_ray = self.radius + offset_left
    #         if hasattr(obj, "radius"):

    #             dx = self.pos[0] - obj.pos[0]
    #             dy = self.pos[1] - obj.pos[1]

    #             return dx * dx + dy * dy >= radial_ray * radial_ray

    #         elif hasattr(obj, "width") and hasattr(obj, "height"):
    #             lower_bound_x = obj.pos[0] - offset_left
    #             upper_bound_x = obj.pos[0] + obj.width + offset_right
    #             lower_bound_y = obj.pos[1] - offset_top
    #             upper_bound_y = obj.pos[1] + obj.width + offset_down

    #             closest_x = max(lower_bound_x, min(upper_bound_x, self.pos[0])) 
    #             closest_y = max(lower_bound_y, min(upper_bound_y, self.pos[1]))

    #             dx = closest_x - obj.pos[0]
    #             dy = closest_y - obj.pos[1]

    #             return dx * dx + dy * dy >= radial_ray * radial_ray

    #     if hasattr(self, "radius"):
    #         w, h = self.radius, self.radius
    #     elif hasattr(self, "width") and hasattr(self, "height"):
    #         w, h = self.width, self.height

    #     if hasattr(obj, "radius"):
    #         ow, oh = obj.radius, obj.radius
    #     elif hasattr(obj, "width") and hasattr(obj, "height"):
    #         ow, oh = obj.width, obj.height

    #     lower_bound_x = obj.pos[0] - w - offset_left
    #     upper_bound_x = obj.pos[0] + ow + w + offset_right
    #     lower_bound_y = obj.pos[1] - h - offset_top
    #     upper_bound_y = obj.pos[1] + oh + h + offset_down

    #     return (lower_bound_x <= self.pos[0] <= upper_bound_x and
    #             lower_bound_y <= self.pos[1] <= upper_bound_y)

    def collide(self, obj, offset_left=0, offset_right=0, offset_top=0, offset_down=0):
        if hasattr(self, "radius"):
            w, h = self.radius, self.radius
        elif hasattr(self, "width") and hasattr(self, "height"):
            w, h = self.width, self.height

        if hasattr(obj, "radius"):
            ow, oh = obj.radius, obj.radius
        elif hasattr(obj, "width") and hasattr(obj, "height"):
            ow, oh = obj.width, obj.height

        lower_bound_x = obj.pos[0] - w - offset_left
        upper_bound_x = obj.pos[0] + ow + w + offset_right
        lower_bound_y = obj.pos[1] - h - offset_top
        upper_bound_y = obj.pos[1] + oh + h + offset_down

        return (lower_bound_x <= self.pos[0] <= upper_bound_x and
                lower_bound_y <= self.pos[1] <= upper_bound_y)


class Agent(Object):
    def __init__(self, start_state=[], **kwargs):
        super().__init__(**kwargs)
        self.start = np.array(start_state)
        self.state = self.start.copy()
        self.rel_start = self.start.copy()
        self.env = None

        # For tracking agents progress
        self.actions = []
        self.reply_buffer = []
        self.record = []
        self.to_concat = []
        self.to_process = []
        self.grad_step = 0
        self.accu_reward = 0

        # Neural networks for tunning the model
        self.dqn = None
        self.tqn = None

        # Hyperparameters
        self.batch_size = None
        self.update_tqn_every = None
        self.buffer_capcity = None
        self.record_capcity = None

    @property
    def norm_pos(self):
        return [int(x / cx) for x, cx in zip(self.pos, self.env.CELL_SIZE)]

    @property
    def more_info(self):
        return np.array([getattr(obj, attr) for obj, attr in self.to_concat]).flatten()
    
    @property
    def input_state(self):
        return np.concatenate((self.state, self.more_info))

    @property
    def tracked(self):
        return self.to_concat + self.to_process

    def interpet_state(self, corrlation, intended_attr=""):
        self.state_related_attr = intended_attr
        setattr(self.__class__, intended_attr, property(corrlation))

    def with_info(self, states):
        return np.concatenate(
                [states, np.broadcast_to(self.more_info, (len(states), len(self.more_info)))], 
                axis=1
            )       

    def reset(self):
        self.state = self.rel_start = self.start
        self.accu_reward = 0
        self.record.clear()

    def track(self, obj, attr, concat=True):
        if concat:
            self.to_concat.append((obj, attr))
        else:
            self.to_process.append((obj, attr))
    
    def set_nn(self, nn):
        self.dqn = nn
        self.tqn = self.dqn.copy()

    def define_actions(self, *actions, done_f=None, fail_f=None):
        self.actions = actions
        self.done = done_f
        self.fail = fail_f

    def lower_bound_state(self, bounds):
        self.lower_bound = bounds

    def upper_bound_state(self, bounds):
        self.upper_bound = bounds

    def take_action(self, ε=0.5, rng=np.random.default_rng()):
        # take random actions initially, then gradually build the policy based on the Q function
        random_pick = rng.choice([1, 0], p=[ε, 1 - ε])
        if random_pick:
            i = rng.integers(len(self.actions))
        else:
            i = self.Q(self.input_state).argmax()

        return i
    
    def Q(self, states):
        if states.ndim == 1:
            self.dqn.set(1)
            out = self.dqn.feedforward(states)
            self.dqn.set(0)
            return out
        
        return self.dqn.feedforward(states)

    def Q_target(self, states):
        return self.tqn.feedforward(states)
     
    def collect_Qtarget(self, gamma=1, batch_size=1, rng=np.random.default_rng()):
        batch_seq = rng.choice(np.array(self.reply_buffer, dtype=object), size=batch_size, replace=False)
        input_states = np.array(list(batch_seq[:, 0]))
        input_xstates = np.array(list(batch_seq[:, 3]))

        Qbatch_seq = self.Q(input_states)
        Qtarget_seq = self.Q_target(input_xstates)

        # Double DQN
        # A_max = self.Q(next_states).argmax(axis=1)
        # Qtarget_seq = Qtarget_seq[np.arange(batch_size), A_max]
        temp_tq = []

        for (_, a, r, _, done), Q, Qt in zip(batch_seq, Qbatch_seq, Qtarget_seq):
            if done:
                Q[self.actions.index(a)] = r
            else:
                Q[self.actions.index(a)] = r + gamma * np.max(Qt)
            temp_tq.append(Q)

        return np.array(temp_tq, dtype=float)

    def compile(self, optim=None, batch_size=1, cost=None, dcost=None, update_tqn_every=None, buffer_capcity=1000, record_capcity=None):
        self.batch_size = batch_size
        self.buffer_capcity = buffer_capcity

        self.update_tqn_every = 2 * self.env.norm_size if update_tqn_every == None else update_tqn_every
        self.record_capcity = int(self.env.norm_size / 2) if record_capcity == None else record_capcity

        # compile the networks
        if not self.dqn.is_compiled:
            self.dqn.compile(
                input_shape=(batch_size, len(self.start) + np.asarray(self.to_concat).size),
                cost=cost, # J(θ) for π(s)
                dcost=dcost, # ∇ J(θ) for π(s)
                optimizer=optim
            )

        if not self.tqn.is_compiled:
            self.tqn.compile(input_shape=(batch_size, len(self.start) + np.asarray(self.to_concat).size))
            self.tqn.set(2)
            self.tqn.copy_from(self.dqn)
                   
    def learn(self, gamma=1):
        # Temporal difference
        if len(self.reply_buffer) > self.batch_size:
            temp_tq = self.collect_Qtarget(gamma, batch_size=self.batch_size)
            self.dqn.backprop(temp_tq) # Q-learning

            self.grad_step = (self.grad_step + 1) % self.update_tqn_every
            if self.grad_step == 0: self.tqn.copy_from(self.dqn)

        if len(self.reply_buffer) >= self.buffer_capcity:
            self.reply_buffer.pop(0)

        if len(self.record) >= self.record_capcity:
            self.accu_reward += np.array(self.record, dtype=object)[:, 2].sum()
            self.rel_start = self.record[-1][3] 
            self.record.clear()


class Environment:
    def __init__(self, map, LAZY_OBJXXT={}, BLOCKED_SPACE=[], CELL_SIZE=None):
        self.map = np.array(map)
        self.norm_width = len(map[0])
        self.norm_height = len(map)
        self.norm_size = self.norm_height * self.norm_width
        self.CELL_SIZE =  CELL_SIZE

        self.agents = []
        self.objects = []
        self.lazy_objects = []

        self.live_agents = []
        self.live_objects = []
        self.live_lazy_objects = []

        self.renderer = None

        self.running = False
        self.require_renderer = False
        self.progress_inspect = False

        # functions
        self.reward = None # to distrubte reward
        self.penalty = None # to punish/reward the model if it end up in a impremessible state

        self.BLOCKED_SPACE = BLOCKED_SPACE
        self.LAZY_CODE = list(LAZY_OBJXXT.keys())
        for i, row in enumerate(map):
            for j, col in enumerate(row):
                if col in self.LAZY_CODE:
                    lazy_obj = Object(pos=[j * CELL_SIZE[0], i * CELL_SIZE[1]], width=CELL_SIZE[0], height=CELL_SIZE[1])
                    lazy_obj.set_renderer(LAZY_OBJXXT[self.map[i, j]])
                    self.lazy_objects.append(lazy_obj)

    def step(self, ε, dt):
        agents_step = []

        for agent in self.agents:
            action = agent.actions[agent.take_action(ε)]
            state = agent.state
            next_state = action(state)
            input_state = agent.input_state    
            agents_step.append((action, next_state, input_state))

        for obj in self.objects: obj.update(dt) # update all active objects

        for agent, (action, next_state, input_state) in zip(self.agents, agents_step):
            if ((np.asarray(next_state) < np.asarray(agent.lower_bound)).any() or 
                (np.asarray(next_state) >= np.asarray(agent.upper_bound)).any()):
                trans = (input_state, action, self.penalty(agent, action), agent.input_state, False)
            else:
                agent.state = next_state
                if self.map[*agent.norm_pos[::-1]] in self.BLOCKED_SPACE:
                    agent.state = state
                    trans = (input_state, action, self.penalty(agent, action), agent.input_state, False)
                else:
                    trans = (input_state, action, self.reward(agent, action), agent.input_state, agent.done())
        
            agent.record.append(trans)
            agent.reply_buffer.append(trans)

    def reset(self):
        for agent in self.agents: agent.reset()
        for obj in self.objects: obj.reset()

        self.live_agents = self.agents.copy()
        self.live_objects = self.objects.copy()
        self.live_lazy_objects = self.lazy_objects.copy()

    def define_reward(self, reward_f, undo_penalty):
        self.reward = reward_f
        self.penalty = undo_penalty

    def add_object(self, object):
        self.objects.append(object)

    def add_agent(self, agent):
        self.agents.append(agent)
        agent.env = self

    def add_renderer(self, renderer):
        self.renderer = renderer
        self.require_renderer = True
        renderer.env = self
        renderer.CELL_SIZE = renderer.cell_width, renderer.cell_height = self.CELL_SIZE

    def kill_lazy_object(self, obj):
        self.live_lazy_objects.remove(obj)

    def kill_object(self, obj):
        self.live_objects.remove(obj)

    def kill_agent(self, agent):
        self.live_agents.remove(agent)

    def run(self, episodes=1, gamma=1, ε_range=(1, 0.05), ε_clip_ratio=0.5, fps=60):
        self.running = True
        ε_start, ε_final = ε_range
        initial_time = perf_counter()

        self.renderer.init()
        for episode in range(episodes):
            ε =  max(ε_final, ε_start - (episode / (ε_clip_ratio * episodes)) * (ε_start - ε_final))

            # run a sequence of actions and update
            self.reset()
            while self.live_agents and self.running:
                dt = self.renderer.render(fps, ε, gamma)

                self.step(ε, dt)
                for agent in self.live_agents:
                    agent.learn(gamma)
                    if agent.done(): self.kill_agent(agent)
                    if agent.fail(): self.reset()

                if episode > episodes - 3: self.renderer.progress_inspect = True

                if not self.renderer.running:
                    print(f"(!) The environment is stopped.         ({perf_counter() - initial_time :.2f}s)")
                    self.renderer.quit()
                    return

            print("Episode #:", episode, "   Total Reward per Child:", [child.accu_reward for child in self.agents])
        print(f"(*) Simulation Finished.            ({perf_counter() - initial_time :.2f}s)")


class Renderer:
    def __init__(self, RES=(800, 600), init=pygame.init, render=None, debug=None, quit=pygame.quit):
        self.env = None
        self.init_core = init
        self.render_core = render
        self.debug_core = debug
        self.quit_core = quit

        # pygame components
        self.W, self.H = RES
        self.screen = pygame.display.set_mode(RES)
        self.clock = pygame.time.Clock()
        self.font_aliases = None

        self.progress_inspect = False
        self.running = False
        self.agent_to_debug = 0

    def pre_process(func):
        def wrapper(instance, *args, **kwargs):
            if instance.env.require_renderer:
                return func(instance, *args, **kwargs)
        return wrapper

    @pre_process
    def init(self): 
        self.init_core()
        self.font_aliases = pygame.font.SysFont(None, int(self.CELL_SIZE[0] / 8) + 1)

    def configuer_debugger(self, figure, info_y, info_x, colors=[], labels=[]):
        self.figure = figure
        self.info_y = info_y
        self.info_x = info_x
        self.colors = colors
        self.labels = labels

    @pre_process
    def debug(self, n_agent):
        agent = self.env.agents[n_agent]

        if isinstance(agent.lower_bound, (int, float)) and isinstance(agent.upper_bound, (int, float)):
            axes = [np.arange(agent.lower_bound, agent.upper_bound)]
        else:
            axes = [np.arange(s, e) for s, e in zip(agent.lower_bound, agent.upper_bound)]

        grid = np.meshgrid(*axes, indexing='ij')
        states = np.column_stack([g.ravel() for g in grid])
        input_states = agent.with_info(states)

        agent.dqn.resize_batch(states.shape[0])
        Qs = agent.Q(input_states)
        agent.dqn.resize_batch(agent.batch_size)

        for q, state in zip(Qs, states):
            lines = [
                f"{l}: {sub_q :.2f}" for l, sub_q in zip(self.labels, q)
            ]

            y = self.info_y(renderer=self, state=state)
            for line, color in zip(lines, self.colors):
                text_surface = self.font_aliases.render(line, True, color)
                self.screen.blit(text_surface, (self.info_x(renderer=self, state=state), y))
                y += self.font_aliases.get_linesize()

            action_idx = np.argmax(q)
            self.figure(renderer=self, action_idx=action_idx, state=state)

    @pre_process
    def render(self, fps, ε, gamma):
        self.running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.progress_inspect = True
                else:
                    self.agent_to_debug = event.key - pygame.K_1

        # fill the screen with a color to wipe away anything from last frame
        self.screen.fill("white")

        for obj in self.env.live_lazy_objects:
            obj.render(self.screen, obj)

        for obj in self.env.live_objects: 
            obj.render(self.screen)

        if self.progress_inspect: 
            self.debug(self.agent_to_debug)
        else:
            for agent in self.env.live_agents: agent.render(self.screen)

        # info logging
        font = pygame.font.SysFont(None, int(self.cell_width / 2) + 1)
        text_surface = font.render(f"   ε={round(ε, 3)}   gamma={gamma}", True, "black")
        text_rect = text_surface.get_rect()
        text_rect.topleft = (0, 0)
        self.screen.blit(text_surface, text_rect)

        for obj in self.env.objects: obj.render(self.screen)

        pygame.display.flip()
        
        # Program sleeps here until an event happens
        while self.progress_inspect:
            # This freezes the program and waits for the next event
            event = pygame.event.wait()
            
            if event.type == pygame.QUIT:
                self.env.running = False
                self.progress_inspect = False  # Break the freeze loop to exit
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.progress_inspect = False  # Break the freeze loop and resume

        dt = self.clock.tick(fps) / 1000.0  # limits FPS
        return dt

    @pre_process
    def quit(self): self.quit_core()

    del pre_process

# the gradient is vanishing and the Q value goes to one actions (R or D depend on dqn activations)
# * Solved: it was really about anchoring the TD to the terminal state that will have target value of (r) which I forget to add
# ! Now, I need to add nn for target values, and fix the nn to accept (x, y) and generlize 
# ! add replay buffer (it will take a batch and sample it to the DQN, just like supervised learning)
# * for clarity: rearrage the nn input to take (s, a) --> Q, not s --> [Q_a]
# ! I suspect that Huber is implemented wrongly