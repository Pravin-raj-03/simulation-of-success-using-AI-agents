import numpy as np
import random
from agent import Agent
from config import WEALTH_BINS, NETWORK_BINS

class RLAgent(Agent):
    def __init__(self, agent_id=None, epsilon=1.0, alpha=0.1, gamma=0.95, q_table=None):
        super().__init__(agent_id)
        # RL Hyperparameters (Heterogeneous Learning)
        self.epsilon = epsilon 
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # 1. Learning Rate (Alpha) linked to Adaptability
        # High Adaptability (100) -> Fast Learner (Alpha 0.2)
        # Low Adaptability (10) -> Slow Learner (Alpha 0.05)
        if alpha == 0.1: # If default
            self.alpha = 0.05 + (self.adaptability / 100.0) * 0.15
        else:
            self.alpha = alpha

        # 2. Discount Factor (Gamma) linked to IQ
        # High IQ (140) -> Long term planner (Gamma 0.99)
        # Low IQ (80) -> Short term thinker (Gamma 0.80)
        if gamma == 0.95: # If default
            # Normalization: Map IQ 70-150 to 0-1 range.
            # CRITICAL: Gamma must be < 1.0 for Bellman convergence.
            normalized_iq = max(0, min(1, (self.talent - 70) / 80)) 
            self.gamma = 0.80 + (normalized_iq * 0.19)
        else:
            self.gamma = gamma
        
        # Q-Table: Shared or Individual
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = {}
        
        self.last_state = None
        self.last_action = None
    
    def get_state(self, market_state, phase_name):
        # State: (WealthBin, NetworkBin, Market, Phase)
        
        # Wealth perception is relative to "survival" vs "rich"
        if self.wealth < WEALTH_BINS[0]: w_bin = 'Poor'
        elif self.wealth < WEALTH_BINS[1]: w_bin = 'Mid'
        elif self.wealth < WEALTH_BINS[2]: w_bin = 'Rich'
        else: w_bin = 'Ultra'
        
        # Network Status
        net_size = len(self.network)
        if net_size < NETWORK_BINS[0]: net_bin = 'Isolated'
        elif net_size < NETWORK_BINS[1]: net_bin = 'Connected'
        else: net_bin = 'Influential'
        
        return (w_bin, net_bin, market_state, phase_name)
    
    def choose_action(self, state, available_actions):
        # Initialize
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}
            
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            # Exploitation: Max Q
            return max(self.q_table[state], key=self.q_table[state].get)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    

    
    def learn(self, current_state, reward):
        """Update Q-Value based on the result of the LAST action taken."""
        if self.last_state is None or self.last_action is None:
            return

        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))
        
        prev_q = self.q_table[self.last_state][self.last_action]
        
        # Max future Q (for current state, which is the 'next state' of the previous step)
        if current_state not in self.q_table:
             # Just initialized, 0
             max_future_q = 0
        else:
             max_future_q = max(self.q_table[current_state].values())

        new_q = prev_q + self.alpha * (reward + self.gamma * max_future_q - prev_q)
        self.q_table[self.last_state][self.last_action] = new_q

    def reset_state(self):
        """Resets the agent's life state for a new episode simulation, while keeping learning (Q-Table)."""
        self.age = 18
        self.alive = True
        self.wealth = 0 # Or inherit some?
        self.network = []
        self.reputation = 0.0
        # self.energy = 100 # Energy removed
        self.last_state = None
        self.last_action = None
