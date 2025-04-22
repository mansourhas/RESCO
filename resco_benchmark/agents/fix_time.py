import time
from resco_benchmark.agents.agent import SharedAgent, Agent
from resco_benchmark.config.signal_config import signal_configs


class FixedTimeAgent(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        """
        Initializes the Fixed-time agent.
        :param config: Configuration parameters.
        :param obs_act: Observations and actions mapping.
        :param map_name: Name of the traffic map.
        :param thread_number: Thread number for parallel execution.
        """
        super().__init__(config, obs_act, map_name, thread_number)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        self.agent = FixedPhaseAgent(signal_configs[map_name]['phase_pairs'])

    def run(self):
        """
        Runs the agent in a loop to control traffic signals at fixed intervals.
        """
        while True:
            actions = self.agent.act()
            self.apply_actions(actions)
            time.sleep(60)  # Wait for 60 seconds before switching phases

class FixedPhaseAgent(Agent):
    def __init__(self, phase_pairs):
        """
        Initializes the Fixed-phase agent.
        :param phase_pairs: List of valid phase pairs for the intersection.
        """
        super().__init__()
        self.phase_pairs = phase_pairs
        self.current_phase = 0

    def act(self, observations=None, valid_acts=None):
        """
        Selects the next phase in a round-robin manner.
        :param observations: (Optional) Observations from the environment.
        :param valid_acts: (Optional) Valid actions for each intersection.
        :return: List of actions for all intersections.
        """
        actions = [self.current_phase] * len(self.phase_pairs)
        self.current_phase = (self.current_phase + 1) % len(self.phase_pairs)
        return actions

    def observe(self, observation, reward, done, info):
        """
        Observation method (not used for fixed-time control).
        """
        pass

    def save(self, path):
        """
        Save method (not used for fixed-time control).
        """
        pass
