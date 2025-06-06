import os
import numpy as np
import traci
import sumolib
import gym
from traffic_signal import Signal


class MultiSignal(gym.Env):
    def __init__(self, run_name, map_name, net, state_fn, reward_fn, route=None, gui=False, end_time=3600,
                 step_length=10, yellow_length=4, step_ratio=1, max_distance=200, lights=(), log_dir='/', libsumo=False,
                 warmup=0, gymma=False):
        self.libsumo = libsumo
        self.gymma = gymma  # gymma expects sequential list of states/rewards instead of dict
        print(map_name, net, state_fn.__name__, reward_fn.__name__)
        self.log_dir = log_dir
        self.net = net
        self.route = route
        self.gui = gui
        self.state_fn = state_fn
        self.reward_fn = reward_fn
        self.max_distance = max_distance
        self.warmup = warmup

        self.end_time = end_time
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.step_ratio = step_ratio
        self.connection_name = run_name + '-' + map_name + '---' + state_fn.__name__ + '-' + reward_fn.__name__
        self.map_name = map_name

        # Run some steps in the simulation with default light configurations to detect phases
        if self.route is not None:
            if 'grid4x4' in self.route:
                self.route += '/grid4x4'
            elif 'arterial4x4' in self.route:
                self.route += '/arterial4x4'
            sumo_cmd = [sumolib.checkBinary('sumo'), '-n', net, '-r', self.route + '_1.rou.xml', '--no-warnings', 'True']
        else:
            sumo_cmd = [sumolib.checkBinary('sumo'), '-c', net, '--no-warnings', 'True']
        print('[DEBUG]:', sumo_cmd)
        if self.libsumo:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label = self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)
        self.signal_ids = self.sumo.trafficlight.getIDList()
        print("lights", len(self.signal_ids), self.signal_ids)

        # this should work on all SUMO versions
        self.phases = {
            lightID: [
                p
                for p in self.sumo.trafficlight.getAllProgramLogics(lightID)[0].getPhases()
                if "y" not in p.state and "g" in p.state.lower()
            ]
            for lightID in self.signal_ids
        }


        self.signals = dict()

        self.all_ts_ids = lights if len(lights) > 0 else self.sumo.trafficlight.getIDList()
        self.ts_starter = len(self.all_ts_ids)
        self.signal_ids = []

        # Pull signal observation shapes
        self.obs_shape = dict()
        self.observation_space = list()
        self.action_space = list()
        for ts in self.all_ts_ids:
            self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts])
        for ts in self.all_ts_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)
        observations = self.state_fn(self.signals)
        self.ts_order = list()
        for ts in observations:
            o_shape = observations[ts].shape
            self.obs_shape[ts] = o_shape
            o_shape = gym.spaces.Box(low=-np.inf, high=np.inf, shape=o_shape)
            self.ts_order.append(ts)
            self.observation_space.append(o_shape)
            if ts == 'top_mgr' or ts == 'bot_mgr': continue  # Not a traffic signal
            self.action_space.append(gym.spaces.Discrete(len(self.phases[ts])))

        self.n_agents = self.ts_starter

        self.run = 0
        self.metrics = []
        self.wait_metric = dict()

        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.connection_name = run_name + '-' + map_name + '-' + str(len(lights)) + '-' + state_fn.__name__ + '-' + reward_fn.__name__
        if not os.path.exists(log_dir+self.connection_name):
            os.makedirs(log_dir+self.connection_name)
        self.sumo_cmd = None
        print('Connection ID', self.connection_name)

    def step_sim(self):
        # The monaco scenario expects .25s steps instead of 1s, account for that here.
        for _ in range(self.step_ratio):
            self.sumo.simulationStep()
        
    def reset(self):
        if self.run != 0:
            if not self.libsumo: traci.switch(self.connection_name)
            traci.close()
            self.save_metrics()
        self.metrics = []

        self.run += 1

        # Start a new simulation
        self.sumo_cmd = []
        if self.gui:
            self.sumo_cmd.append(sumolib.checkBinary('sumo-gui'))
            self.sumo_cmd.append('--start')
        else:
            self.sumo_cmd.append(sumolib.checkBinary('sumo'))
        if self.route is not None:
            self.sumo_cmd += ['-n', self.net, '-r', self.route + '_'+str(self.run)+'.rou.xml']
        else:
            self.sumo_cmd += ['-c', self.net]
        self.sumo_cmd += ['--random', '--time-to-teleport', '-1', '--tripinfo-output',
                          os.path.join(self.log_dir, self.connection_name, 'tripinfo_' + str(self.run) + '.xml'),
                          '--tripinfo-output.write-unfinished',
                          '--no-step-log', 'True',
                          '--no-warnings', 'True']
        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

        for _ in range(self.warmup):
            self.step_sim()

        # 'Start' only signals set for control, rest run fixed controllers
        if self.run % 30 == 0 and self.ts_starter < len(self.all_ts_ids): self.ts_starter += 1
        self.signal_ids = []
        for i in range(self.ts_starter):
            self.signal_ids.append(self.all_ts_ids[i])

        for ts in self.signal_ids:
            self.signals[ts] = Signal(self.map_name, self.sumo, ts, self.yellow_length, self.phases[ts])
            self.wait_metric[ts] = 0.0
        for ts in self.signal_ids:
            self.signals[ts].signals = self.signals
            self.signals[ts].observe(self.step_length, self.max_distance)

        if self.gymma:
            states = self.state_fn(self.signals)
            rets = list()
            for ts in self.ts_order:
                rets.append(states[ts])
            return rets

        return self.state_fn(self.signals)

    def step(self, act):
        if self.gymma:
            dict_act = dict()
            for i, ts in enumerate(self.ts_order):
                dict_act[ts] = act[i]
            act = dict_act

        # Send actions to their signals
        for signal in self.signals:
            self.signals[signal].prep_phase(act[signal])

        for step in range(self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].set_phase()
        for step in range(self.step_length - self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].observe(self.step_length, self.max_distance)

        # observe new state and reward
        observations = self.state_fn(self.signals)
        rewards = self.reward_fn(self.signals)

        self.calc_metrics(rewards)

        done = self.sumo.simulation.getTime() >= self.end_time
        if self.gymma:
            obss, rww = list(), list()
            for ts in self.ts_order:
                obss.append(observations[ts])
                rww.append(rewards[ts])
            return obss, rww, [done], {'eps': self.run}
        return observations, rewards, done, {'eps': self.run}

    def calc_metrics(self, rewards):
        queue_lengths = dict()
        max_queues = dict()
        for signal_id in self.signals:
            signal = self.signals[signal_id]
            queue_length, max_queue = 0, 0
            for lane in signal.lanes:
                queue = signal.full_observation[lane]['queue']
                if queue > max_queue: max_queue = queue
                queue_length += queue
            queue_lengths[signal_id] = queue_length
            max_queues[signal_id] = max_queue
        self.metrics.append({
            'step': self.sumo.simulation.getTime(),
            'reward': rewards,
            'max_queues': max_queues,
            'queue_lengths': queue_lengths
        })

    def save_metrics(self):
        log = os.path.join(self.log_dir, self.connection_name+ os.sep + 'metrics_' + str(self.run) + '.csv')
        print('saving', log)
        with open(log, 'w+') as output_file:
            for line in self.metrics:
                csv_line = ''
                for metric in ['step', 'reward', 'max_queues', 'queue_lengths']:
                    csv_line = csv_line + str(line[metric]) + ', '
                output_file.write(csv_line + '\n')

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.libsumo: traci.switch(self.connection_name)
        traci.close()
        self.save_metrics()
