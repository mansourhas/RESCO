import numpy as np
import networkx as nx
from resco_benchmark.agents.agent import SharedAgent, Agent
from resco_benchmark.config.signal_config import signal_configs


class GraphBasedAgent(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        """
        Initializes the graph-based traffic control agent.
        """
        super().__init__(config, obs_act, map_name, thread_number)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        self.phase_pairs = signal_configs[map_name]['phase_pairs']
        
        # Get intersection connectivity information from the config
        self.intersection_info = config.get('intersection_info', {})
        self.agent = GraphSolverAgent(self.phase_pairs, self.intersection_info)


class GraphSolverAgent(Agent):
    def __init__(self, phase_pairs, intersection_info):
        super().__init__()
        self.phase_pairs = phase_pairs
        self.intersection_info = intersection_info
        self.graph = nx.DiGraph()
        
        # Determine number of intersections from available data
        num_intersections = len(phase_pairs)  # or another appropriate value
        self._initialize_graph(num_intersections)

            
    def _initialize_graph(self, num_intersections):
        """Initialize the graph with nodes for each intersection."""
        # Add a node for each intersection
        for i in range(num_intersections):
            self.graph.add_node(i)  # Explicitly add each node
            
        # Create connections between intersections
        for i in range(num_intersections):
            for j in range(num_intersections):
                if i != j:
                    self.graph.add_edge(i, j, weight=1.0)
        
        self.graph_initialized = True

    
    def _update_graph_weights(self, observations):
        """Update graph edge weights based on observations."""
        # Initialize the graph if needed
        if not hasattr(self, 'graph_initialized') or not self.graph_initialized:
            self._initialize_graph(len(observations))
        
        # Update weights safely
        for i, observation in enumerate(observations):
            # Ensure the node exists
            if i not in self.graph:
                continue
                
            queue_sum = sum(observation)
            # Update outgoing edges
            for neighbor in list(self.graph.neighbors(i)):  # Use list() to avoid iteration issues
                self.graph[i][neighbor]['weight'] = queue_sum

    
    def act(self, observations, valid_acts=None, reverse_valid=None):
        """
        Determine actions for all intersections based on graph analysis.
        """
        # Update graph weights based on observations
        self._update_graph_weights(observations)
        
        acts = []
        for i, observation in enumerate(observations):
            # Use available valid actions if provided
            if valid_acts is not None:
                available_acts = valid_acts[i]
            else:
                available_acts = range(len(self.phase_pairs))
            
            # Calculate betweenness centrality to identify critical intersections
            centrality = nx.betweenness_centrality(self.graph, weight='weight')
            
            # For each intersection, choose the phase with the highest congestion
            best_action = None
            best_score = -float('inf')
            
            for act_idx in available_acts:
                pair = self.phase_pairs[act_idx]
                # Calculate score based on queue length and centrality
                queue_sum = observation[pair[0]] + observation[pair[1]]
                intersection_centrality = centrality.get(i, 0)
                score = queue_sum * (1 + intersection_centrality)
                
                if score > best_score:
                    best_score = score
                    best_action = act_idx
            
            acts.append(best_action if best_action is not None else 0)
        
        return acts
    
    def observe(self, observation, reward, done, info):
        """Observation method (not used in this implementation)."""
        pass

    def save(self, path):
        """Save method (not used in this implementation)."""
        pass
