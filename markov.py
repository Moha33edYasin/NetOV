import pygame
import numpy as np

from netjet.Layers import *
from netjet.models import *
from netjet.methods import *
from time import perf_counter


class RestrictedListParam(list):
    def __init__(self, data, limits, obj=None, track_change=False):
        super().__init__(data)
        self.lower_limit, self.upper_limit = limits
        self.agent = obj
        self.track_change = track_change
        if track_change: self.delta = []

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

        if self.track_change:
            self.delta[key] = super().__getitem__(key) - original_value

class Buffer():
    def __init__(self, capacity, state_shape, float_dtype=np.float32, int_dtype=np.int16):
        self.capacity = capacity

        self.states = np.empty((capacity, *state_shape), dtype=float_dtype)
        self.actions = np.empty((capacity,), dtype=int_dtype)
        self.next_states = np.empty((capacity, *state_shape), dtype=float_dtype)
        self.rewards = np.empty((capacity,), dtype=float_dtype)
        self.dones = np.empty((capacity,), dtype=np.bool_)

        self.idx = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, transition):
        state, action_idx, next_state, reward, done = transition
        capacity = self.__dict__["capacity"]

        self.__dict__["states"][self.idx] = state
        self.__dict__["actions"][self.idx] = action_idx
        self.__dict__["next_states"][self.idx] = next_state
        self.__dict__["rewards"][self.idx] = reward
        self.__dict__["dones"][self.idx] = done

        self.__dict__["idx"] = (self.__dict__["idx"] + 1) % capacity
        self.__dict__["size"] = min(self.__dict__["size"] + 1, capacity)

    def choice(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)

        states = self.states[idx]
        actions = self.actions[idx]
        next_states = self.next_states[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

        return states, actions, next_states, rewards, dones

    def __getattribute__(self, name):
        if name in ["states", "actions", "next_states", "rewards", "dones"]:
            return super().__getattribute__(name)[:self.__dict__["size"]]
        else:
            return super().__getattribute__(name)

class Object:
    def __init__(self, **kwargs):
        self.__dict__["_kept_attrs"] = []
        self.__dict__["_restricted_attrs"] = {}
        self.__dict__["_tracked_attrs"] = []
        self.__dict__["env"] = None
        
        self.__dict__["kwargs"] = kwargs
        self.__dict__["update_f"] = None
        self.__dict__["render_f"] = None
        
        self.__dict__["bound_collisions"] = {}
        self.__dict__["pass_collisions"] = {}
        self.__dict__["intersections"] = []
        
        for attr, val in kwargs.items():
            if callable(val):
                self.__dict__[attr] = val()
            else:
                self.__dict__[attr] = deepcopy(val)

    @property
    def norm_pos(self):
        return [int(x / cx) for x, cx in zip(self.pos, self.env.CELL_SIZE)]

    def __setattr__(self, name, value):
        if hasattr(self, name): 
            original_value = super().__getattribute__(name)
            if name in self._restricted_attrs:
                lower_limit, upper_limit = self._restricted_attrs[name]

                if np.any(value < lower_limit) or np.any(value > upper_limit):
                    self.breaklaw_punish()

                if isinstance(value, (np.ndarray, list)):
                    track = name in self._tracked_attrs
                    value = RestrictedListParam(np.clip(value, a_min=lower_limit, a_max=upper_limit), [lower_limit, upper_limit], obj=self, track_change=track)
                else:
                    value = np.clip(value, a_min=lower_limit, a_max=upper_limit)

                if hasattr(self, "env") and self.env != None:
                    super().__setattr__(name, value)

                    if self.env.map[*self.norm_pos[::-1]] in self.env.BLOCKED_SPACE:
                        ''' Failed to set the attribute '''
                        super().__setattr__(name, original_value)
                        self.breaklaw_punish()
            else:
                super().__setattr__(name, value)

            if name in self._tracked_attrs:
                attr_value = self.__getattribute__(name)
                
                if type(attr_value) == RestrictedListParam:
                    attr_value.delta = np.subtract(attr_value, original_value)
                else:
                    setattr(self, f"{name}_delta", attr_value - original_value)
        else:
            super().__setattr__(name, value)

    def keep(self, *args):
        for arg in args: self._kept_attrs.append(arg)

    def track(self, attr_name, limits=None, track_change=False):
        if limits != None: 
            self._restricted_attrs[attr_name] = limits
            attr = getattr(self, attr_name)
            if isinstance(attr, (np.ndarray, list)):
                super().__setattr__(attr_name, RestrictedListParam(attr, limits, obj=self, track_change=track_change))

        if track_change: self._tracked_attrs.append(attr_name)

    def init(self):
        for attr, val in self.kwargs.items():
            if callable(val):
                setattr(self, attr, val())
            else:
                setattr(self, attr, deepcopy(val))

    def reset(self):
        for attr, val in self.kwargs.items():
            if attr not in self._kept_attrs:
                if callable(val):
                    setattr(self, attr, val())
                else:
                    setattr(self, attr, deepcopy(val))

    def get_velocity(self, dt):
        if type(self.pos) == RestrictedListParam and self.pos.track_change:
            return self.pos.delta / dt
        elif hasattr(self, "pos_delta"):
            return self.pos_delta / dt
        return [0, 0]

    def get_collision_rect(self, offset_left=0, offset_right=0, offset_top=0, offset_down=0):
        w, h = 0, 0

        if hasattr(self, "width"):
            w = self.width
        if hasattr(self, "height"):
            h = self.height
        elif hasattr(self, "radius"):
            w, h = self.radius, self.radius

        left = self.pos[0] - offset_left
        right = self.pos[0] + w + offset_right
        top = self.pos[1] - offset_top
        down = self.pos[1] + h + offset_down
        
        return left, right, top, down

    def collision_lines_intersect(self, m1, m2, p1, p2, left, right, top, down):
        x1, y1 = p1
        x2, y2 = p2

        if (m1 == m2 == None and x2 == x1):
            return True
        elif m1 == None:
            X = x1
            Y = m2 * (X - x2) + y2
        elif m2 == None:
            X = x2
            Y = m1 * (X - x1) + y1
        elif m1 == m2 and y1 == m1 * (x1 - x2) + y2:
            return True        
        else:
            X = (y2 - y1 - (m2 * x2 - m1 * x1)) / (m1 - m2)
            Y = m1 * (X - x1) + y1


        if top <= Y <= down and left <= X <= right:
            self.intersections.append((X, Y))
            return True
        return False

    def by_bound_collide(self, left, right, top, down):
        return left <= self.pos[0] <= right and top <= self.pos[1] <= down

    def by_pass_collide(self, δ1, δ2, left, right, top, down):
        x1, y1 = self.pos.copy()
        x2, y2 = self.pos.copy() + δ1

        if δ2[0] > 0:
            ox1 = _ox1 = left
            uox1 = _uox1 = right
            ox2 = _ox2 = right + δ2[0]
            uox2 = _uox2 = left + δ2[0]
        else:
            ox1 = _ox1 = right
            uox1 = _uox1 = left
            ox2 = _ox2 = left + δ2[0]
            uox2 = _uox2 = right + δ2[0]

        if δ2[1] > 0:
            oy1 = uoy1 = top
            _oy1 = _uoy1 = down
            oy2 = uoy2 = down + δ2[1]
            _oy2 = _uoy2 = top + δ2[1]
        else:
            oy1 = uoy1 = down
            _oy1 = _uoy1 = top
            oy2 = uoy2 = top + δ2[1]
            _oy2 = _uoy2 = down + δ2[1]

        if abs(x1 - left) > abs(x1 - right):
            near_x = right
            far_x = left
        else:
            near_x = left
            far_x = right

        if abs(y1 - top) > abs(y1 - down):
            near_y = down
            far_y = top
        else:
            near_y = top
            far_y = down

        if (
            ((x1 <= near_x and far_x <= x2 or x1 >= near_x and far_x >= x2) and
             (y1 <= near_y and far_y <= y2 or y1 >= near_y and far_y >= y2))
            
            or
            
            ((near_x <= x1 <= far_x or near_x >= x1 >= far_x or 
              near_x <= x2 <= far_x or near_x >= x2 >= far_x) and (y1 <= near_y <= y2 or y1 >= near_y >= y2))
            
            or
            
            ((near_y <= y1 <= far_y or near_y >= y1 >= far_y or 
              near_y <= y2 <= far_y or near_y >= y2 >= far_y) and (x1 <= near_x <= x2 or x1 >= near_x >= x2))
        ):
            dx, dy = δ1

            odx = ox2 - ox1
            ody = oy2 - oy1

            _odx = _ox2 - _ox1
            _ody = _oy2 - _oy2

            uodx = uox2 - uox1
            uody = uoy2 - uoy2

            _uodx = _uox2 - _uox1
            _uody = _uoy2 - _uoy2

            m = dy / dx if dx != 0 else None
            om = ody / odx if odx != 0 else None
            _om = _ody / _odx if _odx != 0 else None
            uom = uody / uodx if uodx != 0 else None
            _uom = _uody / _uodx if _uodx != 0 else None

            self.intersections.clear()
            return (    
                self.collision_lines_intersect(m, om, (x1, y1), (ox1, oy1), left, right, top, down) |
                self.collision_lines_intersect(m, _om, (x1, y1), (_ox1, _oy1), left, right, top, down) |
                self.collision_lines_intersect(m, uom, (x1, y1), (uox1, uoy1), left, right, top, down) |
                self.collision_lines_intersect(m, _uom, (x1, y1), (_uox1, _uoy1), left, right, top, down)
            )

    def collide(self, obj, dt=1, offset_left=0, offset_right=0, offset_top=0, offset_down=0):
        w, h = 0, 0
        if hasattr(self, "width"):
            w = self.width
        if hasattr(self, "height"):
            h = self.height
        elif hasattr(self, "radius"):
            w, h = self.radius, self.radius

        left, right, top, down = obj.get_collision_rect( 
                                                        offset_left + w, 
                                                        offset_right + w,
                                                        offset_top + h,
                                                        offset_down + h
                                                    )

        if hasattr(self, "velocity"):
            v = self.velocity
        elif hasattr(self, "direction") and hasattr(self, "speed"):
            v = np.multiply(self.direction, self.speed)
        else:
            v = self.get_velocity(dt)

        if hasattr(obj, "velocity"):
            ov = obj.velocity
        elif hasattr(obj, "direction") and hasattr(obj, "speed"):
            ov = np.multiply(obj.direction, obj.speed)
        else:
            ov = obj.get_velocity(dt)

        return (self.by_bound_collide(left, right, top, down) or 
                self.by_pass_collide(np.multiply(v, dt), np.multiply(ov, dt), left, right, top, down))

    def multi_collide(self, objects, dt=1, offset_left=0, offset_right=0, offset_top=0, offset_down=0):
        w, h = 0, 0
        if hasattr(self, "width"):
            w = self.width
        if hasattr(self, "height"):
            h = self.height
        elif hasattr(self, "radius"):
            w, h = self.radius, self.radius

        if hasattr(self, "velocity"):
            v = self.velocity
        elif hasattr(self, "direction") and hasattr(self, "speed"):
            v = np.multiply(self.direction, self.speed)
        else:
            v = self.get_velocity(dt)

        δ1 = np.multiply(v, dt)

        self.pass_collisions.clear()
        self.bound_collisions.clear()
        
        for obj in objects:
            left, right, top, down = obj.get_collision_rect( 
                offset_left + w, 
                offset_right + w,
                offset_top + h,
                offset_down + h
            )

            if hasattr(obj, "velocity"):
                ov = obj.velocity
            elif hasattr(obj, "direction") and hasattr(obj, "speed"):
                ov = np.multiply(obj.direction, obj.speed)
            else:
                ov = obj.get_velocity(dt)

            δ2 = np.multiply(ov, dt)

            if self.by_bound_collide(left, right, top, down):
                dx, dy = np.subtract(obj.pos, self.pos)
                self.bound_collisions[dx * dx + dy * dy] = obj

            elif self.by_pass_collide(δ1, δ2, left, right, top, down):
                displacements = np.subtract(self.intersections, self.pos)    
                # directed = displacements * (v / np.abs(v))
                distances = [dx * dx + dy * dy for dx, dy in displacements]
                # if ahead: self.bypass_collisions[min(ahead)] = obj
                self.pass_collisions[min(distances)] = obj

        if self.bound_collisions:
            collision_obj = self.bound_collisions[min(self.bound_collisions.keys())]
            return collision_obj
        
        if self.pass_collisions:
            collision_obj = self.pass_collisions[min(self.pass_collisions.keys())]
            return collision_obj
                
        return None


class Agent(Object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # For tracking agents progress
        self.actions = []
        self.record = []
        self.reply_buffer = None
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
    def input_state(self):
        return self.env.state_f(self.env, self)

    def grant(self, n_points):
        self.granted_points += n_points

    def breaklaw_punish(self):
        self.granted_points += self.breaklaw_penalty

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

    def take_action(self, ε=0.5, rng=np.random.default_rng()):
        # take random actions initially, then gradually build the policy based on the Q function
        random_pick = rng.choice([1, 0], p=[ε, 1 - ε])
        if random_pick:
            i = rng.integers(len(self.actions))
        else:
            i = self.Q(self.input_state).argmax()

        return self.actions[i], i
    
    def Q(self, states):
        if np.array(states).ndim == 1:
            self.dqn.set(1)
            out = self.dqn.feedforward(states)
            self.dqn.set(0)
            return out
        
        return self.dqn.feedforward(states)

    def Q_target(self, states):
        return self.tqn.feedforward(states)
     
    def collect_Qtarget(self, gamma=1, batch_size=1):
        states, actions, next_states, rewards, dones = self.reply_buffer.choice(batch_size)

        Qbatch_seq = self.Q(states)
        Qtarget_seq = self.Q_target(next_states)

        # Double DQN
        # A_max = self.Q(next_states).argmax(axis=1)
        # Qtarget_seq = Qtarget_seq[np.arange(batch_size), A_max]
        temp_tq = []

        for a, r, done, Q, Qt in zip(actions, rewards, dones, Qbatch_seq, Qtarget_seq):
            if done:
                Q[a] = r
            else:
                Q[a] = r + gamma * np.max(Qt)
            temp_tq.append(Q)

        return np.array(temp_tq, dtype=float)

    def compile(self, batch_size=1, optim=None, cost=None, dcost=None, update_tqn_every=None, buffer_capcity=1000, record_capcity=None):
        self.batch_size = batch_size
        self.update_tqn_every = -1 if update_tqn_every == None else update_tqn_every
        self.record_capcity = -1 if record_capcity == None else record_capcity
        
        input_shape = (batch_size, *np.array(self.env.state_f(self.env, self)).shape)
        self.reply_buffer = Buffer(buffer_capcity, input_shape[1:])

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
        if self.reply_buffer.size > self.batch_size:
            temp_tq = self.collect_Qtarget(gamma, batch_size=self.batch_size)
            self.dqn.backprop(temp_tq) # Q-learning

            self.grad_step = (self.grad_step + 1) % self.update_tqn_every
            if self.grad_step == 0: self.tqn.copy_from(self.dqn)

        if len(self.record) >= self.record_capcity: self.record.pop(0)


class Environment:
    def __init__(self, map, require_live_map=True, lazy_code=None, lazy_render={}, blocked_space=[], cell_size=None):
        self.map = np.array(map)

        if require_live_map: self.live_map = np.array(map)

        self.norm_width = len(map[0])
        self.norm_height = len(map)
        self.norm_size = self.norm_height * self.norm_width
        self.CELL_SIZE =  cell_size

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
        self.release_ε = False

        # functions
        self.state_f = None # to produce state reperesentation for the agent(s)
        self.reward_f = None # to distrubte punishment/reward

        self.BLOCKED_SPACE = blocked_space
        self.LAZY_CODE = lazy_code if lazy_code != None else list(lazy_render.keys())
        self.NONE_CODE = None

        for i, row in enumerate(map):
            for j, col in enumerate(row):
                if col in self.LAZY_CODE:
                    loc = [j * cell_size[0], i * cell_size[1]]
                    lazy_obj = Object(pos=loc, width=cell_size[0], height=cell_size[1])
                    lazy_obj.env = self

                    if lazy_render: lazy_obj.render_f = lazy_render[self.map[i, j]]

                    self.lazy_objects.append(lazy_obj)
                    self.lazy_loc.append(loc)

                elif self.NONE_CODE == None:
                    self.NONE_CODE = col

    def step(self, ε, dt):
        agents_step = []

        for agent in self.agents:
            state = self.state_f(self, agent)
            action, action_idx = agent.take_action(ε)
            agents_step.append((state, action_idx))
            
            action(agent) # This will excute action only if it was succesful.
        
        for obj in self.active_objects: obj.update_f(dt) # update all active objects

        for agent, (state, action_idx) in zip(self.agents, agents_step):
            next_state = self.state_f(self, agent)
            net_reward = agent.granted_points

            if self.reward_f != None:
                net_reward += self.reward_f(self, agent)

            trans = (state, action_idx, next_state, net_reward, agent.done())        
            agent.record.append(trans)
            agent.reply_buffer.push(trans)
            agent.accu_reward += net_reward
            agent.granted_points = 0

    def active_objects_init(self):
        for obj in self.active_objects: obj.init()
        self.live_active_objects = self.active_objects.copy()

    def active_objects_reset(self):
        for obj in self.active_objects: obj.reset()
        self.live_active_objects = self.active_objects.copy()

    def reset(self):
        for agent in self.agents: agent.init()
        self.live_agents = self.agents.copy()

        self.active_objects_init()
        self.live_lazy_objects = self.lazy_objects.copy()

    def add_object(self, obj):
        self.active_objects.append(obj)
        obj.env = self

    def add_agent(self, agent):
        self.agents.append(agent)
        agent.env = self

    def add_renderer(self, renderer):
        self.renderer = renderer
        self.require_renderer = True
        renderer.env = self
        renderer.CELL_SIZE = renderer.cell_width, renderer.cell_height = self.CELL_SIZE

    def kill_lazy_object(self, obj):
        if hasattr(self, "live_map"):
            map_pos = (np.array(obj.pos) / np.array(self.CELL_SIZE)).astype(int)
            self.live_map[*map_pos[::-1]] = self.NONE_CODE
        self.live_lazy_objects.remove(obj)

    def kill_object(self, obj):
        self.live_active_objects.remove(obj)

    def kill_agent(self, agent):
        self.live_agents.remove(agent)

    def run(self, episodes=1, gamma=1, ε_range=(1.0, 0.0), ε_clip_ratio=0.5, fps=60):
        self.running = True
        ε_start, ε_final = ε_range
        initial_time = t_start = perf_counter()

        if self.require_renderer:
            self.renderer.init()

        for episode in range(episodes):
            # run a sequence of actions and update
            self.reset()
            while self.live_agents and self.running:
                if self.release_ε:
                    ε = ε_final
                else:
                    ε =  max(ε_final, ε_start - (episode / (ε_clip_ratio * episodes)) * (ε_start - ε_final))

                if self.require_renderer: 
                    dt = self.renderer.render(fps, ε, gamma)  
                else:
                    end_start = perf_counter()
                    dt = end_start - t_start
                    t_start = end_start
                
                self.step(ε, dt)

                for agent in self.live_agents:
                    agent.learn(gamma)
                    if agent.fail() or agent.done(): self.kill_agent(agent)

                if episode == episodes - 5:
                    if self.require_renderer: self.renderer.progress_inspect = True
                    self.agents[self.renderer.agent_to_debug].dqn.save(f"BreakoutA{self.map.shape}{self.renderer.agent_to_debug}")

                if not self.running or (self.require_renderer and not self.renderer.running):
                    self.renderer.quit()
                    print(f"(!) The environment is stopped.         ({perf_counter() - initial_time :.2f}s)")
                    return

            print("Episode #:", episode, "   Total Reward per Child:", [np.round(agent.accu_reward, 2) for agent in self.agents])
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
        self.screen = None
        self.clock = None

        # fonts
        self.hyperparam_font = None
        self.reward_font = None
        self.font_aliases = []

        self.progress_inspect = False
        self.running = False
        self.agent_to_debug = 0

    def init(self):
        self.init_core()

        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()

        self.hyperparam_font = pygame.font.SysFont(None, int(self.H / 15) + 1)
        self.reward_font = pygame.font.SysFont(None, int(self.H / 18) + 1)
        for agent in self.env.agents:
            self.font_aliases.append(pygame.font.SysFont(None, int(self.cell_height / (1.5 * len(agent.actions))) + 1))

    def configuer_debugger(self, figure, info_y=None, info_x=None, colors=[], labels=[]):
        self.figure = figure
        self.info_y = info_y
        self.info_x = info_x
        self.colors = colors
        self.labels = labels

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

            if agent.pos.track_change:
                delta = agent.pos.delta.copy()

            for p in coordinates:
                agent.pos = p
                states.append(self.env.state_f(self.env, agent))

            agent.pos = pos
            if agent.pos.track_change:
                agent.pos.delta = delta

            states = np.array(states)
            agent.dqn.resize_batch(states.shape[0])

            Qs = agent.Q(states)
            agent.dqn.resize_batch(agent.batch_size)

            for q, p in zip(Qs, coordinates):
                if self.info_y != None and self.info_x != None:
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

            if self.show_info:
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

    def render(self, fps, ε, gamma):
        self.running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.progress_inspect = True
                elif event.key == pygame.K_PAGEDOWN:
                    self.env.release_ε = not self.env.release_ε
                elif event.key == pygame.K_s:
                    self.env.agents[self.agent_to_debug].dqn.save(f"BreakoutA{self.env.map.shape}{self.agent_to_debug}")
                    self.env.agents[self.agent_to_debug].tqn.save(f"BreakoutT{self.env.map.shape}{self.agent_to_debug}")
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
            for agent in self.env.live_agents:
                agent.render_f(self.screen)

                # show the reward / penalty on the screen close to the agent
                if agent.reply_buffer:
                    reward = agent.reply_buffer.rewards[-1]

                    if reward > 0:
                        text_surface = self.reward_font.render(f"+{reward :.2g}", True, "green")
                    elif reward == 0:
                        continue
                    else:
                        text_surface = self.reward_font.render(f"{reward :.2g}", True, "red")

                    if hasattr(agent, "width"):
                        label_x = agent.pos[0] + agent.width - self.cell_width / 2
                    elif hasattr(agent, "radius"):
                        label_x = agent.pos[0] + agent.radius - self.cell_width / 2
                    else:
                        label_x = agent.pos[0]

                    if hasattr(agent, "radius"):
                        text_rect = text_surface.get_rect(center=(label_x, agent.pos[1] - agent.radius - self.reward_font.get_height()))
                    else:
                        text_rect = text_surface.get_rect(center=(label_x, agent.pos[1] - self.reward_font.get_height()))
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

    def quit(self): self.quit_core()