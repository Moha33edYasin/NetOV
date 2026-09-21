import pygame
import numpy as np

from netjet.Layers import *
from netjet.models import *
from netjet.methods import *
from time import perf_counter


class RestrictedListParam(list):
    def __init__(self, data, limits, agent=None):
        super().__init__(data)
        self.lower_limit, self.upper_limit = limits
        self.agent = agent

    @property
    def env(self): return self.agent.env

    def __setitem__(self, key, value):
        lower = self.lower_limit[key]
        upper = self.upper_limit[key]
        original_value = super().__getitem__(key)

        if value < lower or value > upper:
            self.agent.breaklaw_punish()

        super().__setitem__(key, np.clip(value, a_min=lower, a_max=upper))

        if self.env != None:
            if self.env.map[*self.agent.norm_pos[::-1]] in self.env.BLOCKED_SPACE:
                super().__setitem__(key, original_value)
                self.agent.breaklaw_punish()

class Object:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.update_f = None
        self.render_f = None
        for attr, val in kwargs.items():
            if callable(val):
                setattr(self, attr, val())
            else:
                setattr(self, attr, deepcopy(val))

    def reset(self):
        for attr, val in self.kwargs.items():
            if callable(val):
                setattr(self, attr, val())
            else:
                setattr(self, attr, deepcopy(val))

    def collide(self, obj, offset_left=0, offset_right=0, offset_top=0, offset_down=0):
        ow, oh = 0, 0
        w, h = 0, 0

        if hasattr(self, "width"):
            w = self.width
        if hasattr(self, "height"):
            h = self.height
        elif hasattr(self, "radius"):
            w, h = self.radius, self.radius

        if hasattr(obj, "width"):
            ow = obj.width
        if hasattr(obj, "height"):
            oh = obj.height
        elif hasattr(obj, "radius"):
            ow, oh = obj.radius, obj.radius

        lower_bound_x = obj.pos[0] - w - offset_left
        upper_bound_x = obj.pos[0] + ow + w + offset_right
        lower_bound_y = obj.pos[1] - h - offset_top
        upper_bound_y = obj.pos[1] + oh + h + offset_down

        return ((lower_bound_x <= self.pos[0] <= upper_bound_x) and
                (lower_bound_y <= self.pos[1] <= upper_bound_y))


class Agent(Object):
    def __init__(self, **kwargs):
        self.__dict__["_restricted_attrs"] = {}
        self.env = None
        
        super().__init__(**kwargs)

        # For tracking agents progress
        self.actions = []
        self.reply_buffer = []
        self.record = []
        self.grad_step = 0

        self.granted_points = 0
        self.breaklaw_penalty = 0
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
    def input_state(self):
        return self.env.state_f(self.env, self)

    def __setattr__(self, name, value):
        if name in self._restricted_attrs:
            if self._restricted_attrs.get(name) != None:
                lower_limit, upper_limit = self._restricted_attrs[name]

                if np.any(value < lower_limit) or np.any(value > upper_limit):
                    self.breaklaw_punish()

                if isinstance(value, (np.ndarray, list)):
                    value = RestrictedListParam(np.clip(value, a_min=lower_limit, a_max=upper_limit), [lower_limit, upper_limit], agent=self)
                else:
                    value = np.clip(value, a_min=lower_limit, a_max=upper_limit)

            if hasattr(self, "env") and self.env != None:
                original_value = self.__getattribute__(name)
                super().__setattr__(name, value)

                if self.env.map[*self.norm_pos[::-1]] in self.env.BLOCKED_SPACE:
                    ''' Failed to set the attribute '''
                    super().__setattr__(name, original_value)
                    self.breaklaw_punish()
        else:
            super().__setattr__(name, value)

    def grant(self, n_points):
        self.granted_points += n_points

    def breaklaw_punish(self):
        self.granted_points = self.breaklaw_penalty

    def reset(self):
        super().reset()
        self.granted_points = 0
        self.accu_reward = 0
        self.record.clear()

    def set_nn(self, nn):
        self.dqn = nn
        self.tqn = self.dqn.copy()

    def define_actions(self, *actions, breaklaw_penalty=-1, done_f=None, fail_f=None):
        self.actions = actions
        self.breaklaw_penalty = breaklaw_penalty
        self.done = done_f
        self.fail = fail_f

    def limit(self, attr_name, limits=None):
        self._restricted_attrs[attr_name] = limits
        attr = getattr(self, attr_name)
        if isinstance(attr, (np.ndarray, list)):
            super().__setattr__(attr_name, RestrictedListParam(attr, limits, agent=self))

    def take_action(self, ε=0.5, rng=np.random.default_rng()):
        # take random actions initially, then gradually build the policy based on the Q function
        random_pick = rng.choice([1, 0], p=[ε, 1 - ε])
        if random_pick:
            i = rng.integers(len(self.actions))
        else:
            i = self.Q(self.input_state).argmax()

        return self.actions[i]
    
    def Q(self, states):
        if np.array(states).ndim == 1:
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

    def compile(self, batch_size=1, optim=None, cost=None, dcost=None, update_tqn_every=None, buffer_capcity=1000, record_capcity=None):
        self.batch_size = batch_size
        self.buffer_capcity = buffer_capcity
        
        self.update_tqn_every = 2 * self.env.norm_size if update_tqn_every == None else update_tqn_every
        self.record_capcity = int(self.env.norm_size / 2) if record_capcity == None else record_capcity
        input_shape = (batch_size, *np.array(self.env.state_f(self.env, self)).shape)

        # compile the networks
        if not self.dqn.is_compiled:
            self.dqn.compile(
                input_shape=input_shape,
                cost=cost, # J(θ) for π(s)
                dcost=dcost, # ∇ J(θ) for π(s)
                optimizer=optim
            )

        if not self.tqn.is_compiled:
            self.tqn.compile(input_shape=input_shape)
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
            self.record.clear()


class Environment:
    def __init__(self, map, lazy_render={}, BLOCKED_SPACE=[], CELL_SIZE=None):
        self.map = np.array(map)
        self.norm_width = len(map[0])
        self.norm_height = len(map)
        self.norm_size = self.norm_height * self.norm_width
        self.CELL_SIZE =  CELL_SIZE

        self.agents = []
        self.active_objects = []
        self.lazy_objects = []
        self.lazy_loc = []

        self.live_agents = []
        self.live_active_objects = []
        self.live_lazy_objects = []

        self.renderer = None

        self.running = False
        self.require_renderer = False
        self.progress_inspect = False

        # functions
        self.state_f = None # to produce state reperesentation for the agent(s)
        self.reward_f = None # to distrubte punishment/reward

        self.BLOCKED_SPACE = BLOCKED_SPACE
        self.LAZY_CODE = list(lazy_render.keys())
        for i, row in enumerate(map):
            for j, col in enumerate(row):
                if col in self.LAZY_CODE:
                    loc = [j * CELL_SIZE[0], i * CELL_SIZE[1]]
                    lazy_obj = Object(pos=loc, width=CELL_SIZE[0], height=CELL_SIZE[1])
                    lazy_obj.render_f = lazy_render[self.map[i, j]]
                    self.lazy_objects.append(lazy_obj)
                    self.lazy_loc.append(loc)

    def step(self, ε, dt):
        agents_step = []

        for agent in self.agents:
            state = self.state_f(self, agent)
            action = agent.take_action(ε)
            agents_step.append((state, action))

        for obj in self.active_objects: obj.update_f(dt) # update all active objects

        for agent, (state, action) in zip(self.agents, agents_step):
            # check if it make a successful action by checking if any of the agentic attributes changed
            action(agent)

            next_state = self.state_f(self, agent)
            net_reward = agent.granted_points

            if self.reward_f != None:
                net_reward += self.reward_f(self, agent)

            trans = (state, action, net_reward, next_state, agent.done())        
            agent.record.append(trans)
            agent.reply_buffer.append(trans)
            agent.granted_points = 0

    def active_objects_reset(self):
        for obj in self.active_objects: obj.reset()
        self.live_active_objects = self.active_objects.copy()

    def reset(self):
        for agent in self.agents: agent.reset()
        self.live_agents = self.agents.copy()

        self.active_objects_reset()
        self.live_lazy_objects = self.lazy_objects.copy()

    def add_object(self, object):
        self.active_objects.append(object)

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
        self.live_active_objects.remove(obj)

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
                    if agent.fail(): self.active_objects_reset()

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

        # fonts
        self.hyperparam_font = None
        self.reward_font = []
        self.font_aliases = []

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
        cell_size = self.cell_width * self.cell_height

        self.init_core()

        self.hyperparam_font = pygame.font.SysFont(None, int(cell_size / 80) + 1)
        for agent in self.env.agents:
            if hasattr(agent, "width") and hasattr(agent, "height"):
                size = agent.width * agent.height / 35
            elif hasattr(agent, "width"):
                size = agent.width * 0.75
            elif hasattr(agent, "height"):
                size = agent.height * 0.75
            elif hasattr(agent, "radius"):
                size = agent.radius * 0.75
            else:
                size = cell_size / 50

            self.reward_font.append(pygame.font.SysFont(None, int(size / 2) + 1))
            self.font_aliases.append(pygame.font.SysFont(None, int(size / (2 * len(agent.actions))) + 1))

    def configuer_debugger(self, figure, info_y, info_x, colors=[], labels=[]):
        self.figure = figure
        self.info_y = info_y
        self.info_x = info_x
        self.colors = colors
        self.labels = labels

    @pre_process
    def debug(self, n_agent):
        agent = self.env.agents[n_agent]
        limits = agent._restricted_attrs.get("pos")

        if limits:
            lower, upper = limits
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                axes = [np.arange(lower / self.CELL_SIZE[0], upper / self.CELL_SIZE[0] + 1)]
            else:
                axes = [np.arange(s / n, e / n + 1) for s, e, n in zip(lower, upper, self.CELL_SIZE)]

            grid = np.meshgrid(*axes, indexing='ij')
            coordinates = np.column_stack([g.ravel() for g in grid]) * self.CELL_SIZE

            states = []
            pos = list(agent.pos)
            for p in coordinates:
                agent.pos = p
                states.append(self.env.state_f(self.env, agent))

            agent.pos = pos
            states = np.array(states)
            agent.dqn.resize_batch(states.shape[0])

            Qs = agent.Q(states)
            agent.dqn.resize_batch(agent.batch_size)

            for q, p in zip(Qs, coordinates):
                lines = [
                    f"{l}: {sub_q :.2f}" for l, sub_q in zip(self.labels, q)
                ]

                y = self.info_y(renderer=self, coordinate=p)
                for line, color in zip(lines, self.colors):
                    text_surface = self.font_aliases[n_agent].render(line, True, color)
                    self.screen.blit(text_surface, (self.info_x(renderer=self, coordinate=p), y))
                    y += self.font_aliases[n_agent].get_linesize()

                action_idx = np.argmax(q)
                self.figure(renderer=self, action_idx=action_idx, coordinate=p)
        else:
            print("(!) No limits to agent's position are found.")
            state = self.env.state_f(self.env, agent)

            agent.dqn.set(1)
            Q = agent.Q(state).squeeze()
            agent.dqn.set(0)

            lines = [
                f"{l}: {sub_q :.2f}" for l, sub_q in zip(self.labels, Q)
            ]

            y = self.info_y(renderer=self, coordinate=agent.pos)
            for line, color in zip(lines, self.colors):
                text_surface = self.font_aliases.render(line, True, color)
                self.screen.blit(text_surface, (self.info_x(renderer=self, coordinate=agent.pos), y))
                y += self.font_aliases.get_linesize()

            action_idx = np.argmax(Q)
            self.figure(renderer=self, action_idx=action_idx, coordinate=agent.pos)

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
                    agent_to_debug = event.key - pygame.K_1
                    if agent_to_debug < len(self.env.agents):
                        self.agent_to_debug = agent_to_debug
                    else:
                        print(f"(!) Sorry, enable to find agent no. ({agent_to_debug}).")

        # fill the screen with a color to wipe away anything from last frame
        self.screen.fill("white")

        for obj in self.env.live_lazy_objects:
            obj.render_f(self.screen, obj)

        for obj in self.env.live_active_objects: 
            obj.render_f(self.screen)

        if self.progress_inspect:
            self.debug(self.agent_to_debug)
        else:
            for i, agent in enumerate(self.env.live_agents):
                agent.render_f(self.screen)

                # show the reward / penalty on the screen close to the agent
                if agent.reply_buffer:
                    reward = agent.reply_buffer[-1][2]
                    
                    if reward > 0:
                        text_surface = self.reward_font[i].render(f"+{reward}", True, "green")
                    elif reward == 0:
                        continue
                    else:
                        text_surface = self.reward_font[i].render(f"{reward}", True, "red")

                    if hasattr(agent, "width"):
                        label_x = agent.pos[0] + agent.width - self.cell_width / 2
                    elif hasattr(agent, "radius"):
                        label_x = agent.pos[0] + agent.radius - self.cell_width / 2
                    else:
                        label_x = agent.pos[0]

                    text_rect = text_surface.get_rect(center=(label_x, agent.pos[1] - self.cell_height / 2))
                    self.screen.blit(text_surface, text_rect)

        # info logging
        text_surface = self.hyperparam_font.render(f"   ε={round(ε, 3)}   gamma={gamma}", True, "black")
        text_rect = text_surface.get_rect()
        text_rect.topleft = (0, 0)
        self.screen.blit(text_surface, text_rect)

        for obj in self.env.live_active_objects: obj.render_f(self.screen)

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