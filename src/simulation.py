import pandas as pd
import numpy as np
from tqdm import tqdm
from agent import Agent
from environment import EducationPhase, CareerPhase, OpportunityPhase, DeclinePhase
from config import PHASE_DURATIONS, POPULATION_SIZE, MARKET_CYCLES, MARKOV_TRANSITION_MATRIX, CONNECTION_LIMIT

class Simulation:
    def __init__(self, pop_size=POPULATION_SIZE, seed=None, enable_interactions=True):
        if seed is not None:
            np.random.seed(seed)
        
        self.enable_interactions = enable_interactions
        self.agents = [Agent() for _ in range(pop_size)]
        self.agent_map = {a.id: a for a in self.agents}
        self.data_log = []
        self.current_step = 0
        self.market_state = 'NORMAL'
        
        # Lifecycle Schedule
        self.schedule = []
        for _ in range(PHASE_DURATIONS['EDUCATION']): self.schedule.append(EducationPhase())
        for _ in range(PHASE_DURATIONS['CAREER_EARLY']): self.schedule.append(CareerPhase(stage="Early"))
        for _ in range(PHASE_DURATIONS['OPPORTUNITY']): self.schedule.append(OpportunityPhase())
        for _ in range(PHASE_DURATIONS['DECLINE']):
            if np.random.random() < 0.5: self.schedule.append(DeclinePhase())
            else: self.schedule.append(CareerPhase(stage="Late"))

    def update_market(self):
        # Dynamic Markov Chain Market
        # Uses the current state to determine probabilities of the next state
        current_probs = MARKOV_TRANSITION_MATRIX[self.market_state]
        self.market_state = np.random.choice(MARKET_CYCLES, p=current_probs)

    def interaction_step(self):
        """Phase 2: Agents network with each other."""
        # Only active agents interact
        active_agents = [a for a in self.agents if a.alive]
        num_interactions = len(active_agents) // 2
        
        # Shuffle and pair
        indices = np.random.permutation(len(active_agents))
        
        for i in range(0, len(indices)-1, 2):
            a1 = active_agents[indices[i]]
            a2 = active_agents[indices[i+1]]
            
            # Connection Probability: Similarity + Reputation
            # Rich/Famous people are harder to connect with unless you are also rich
            score_a1 = a1.reputation + (a1.wealth / 10000)
            score_a2 = a2.reputation + (a2.wealth / 10000)
            
            # Simple logic: if both have compatiblity (e.g. IQ similar) or high luck
            iq_diff = abs(a1.talent - a2.talent)
            iq_diff = abs(a1.talent - a2.talent)
            prob = 0.05 # Reduced Base (was 0.1) to make active networking more valuable
            if iq_diff < 10: prob += 0.2 # Birds of a feather
            if score_a1 > 10 or score_a2 > 10: prob += 0.2 # Social climbers
            
            if np.random.random() < prob:
                if len(a1.network) < CONNECTION_LIMIT: a1.network.append(a2.id)
                if len(a1.network) < CONNECTION_LIMIT: a1.network.append(a2.id)
                if len(a2.network) < CONNECTION_LIMIT: a2.network.append(a1.id)

    def add_strategic_connection(self, agent):
        """Finds a high-value connection for the agent."""
        # Selection: Sample 10 random agents, pick best (highest Rep/Wealth)
        candidates = np.random.choice(self.agents, 10)
        best_candidate = None
        max_score = -float('inf')
        
        for cand in candidates:
            if cand.id == agent.id: continue
            if cand.id in agent.network: continue
            
            # Score based on Utility
            score = cand.reputation + (cand.wealth/5000)
            if score > max_score:
                max_score = score
                best_candidate = cand
        
        if best_candidate:
            if len(agent.network) < CONNECTION_LIMIT:
                agent.network.append(best_candidate.id)
            # Mutual connection for simplicity
            if len(best_candidate.network) < CONNECTION_LIMIT:
                best_candidate.network.append(agent.id)

        # Update 'Network Quality' for Opportunity Phase
        # Update 'Network Quality' for Opportunity Phase
        # Just for this agent
        if not agent.network:
            agent.network_quality = 0
            return
        
        # Calculate average wealth of network
        net_wealth = 0
        count = 0
        for peer_id in agent.network:
            if peer_id in self.agent_map:
                peer = self.agent_map[peer_id]
                if peer.alive:
                    net_wealth += peer.wealth
                    count += 1
        
        avg_net_wealth = net_wealth / max(1, count)
        # Normalize quality (arbitrary scaling, say 1M is max quality)
        agent.network_quality = min(1.0, avg_net_wealth / 500000)

    def apply_competition(self, performance_map):
        """Phase 2: Zero-Sum Competition in Career."""
        # Rank agents by performance
        # Valid only for agents who succeeded/produced performance
        sorted_agents = sorted(performance_map.items(), key=lambda x: x[1], reverse=True)
        count = len(sorted_agents)
        if count < 10: return
        
        top_10_cutoff = int(count * 0.1)
        bottom_10_cutoff = int(count * 0.9)
        
        # Promotions
        for i in range(top_10_cutoff):
            a_id, _ = sorted_agents[i]
            agent = self.agent_map[a_id]
            agent.wealth += 5000 # Bonus
            agent.reputation += 1.0
            
        # Layoffs (adding insult to injury, they might have already earned 0)
        for i in range(bottom_10_cutoff, count):
            a_id, _ = sorted_agents[i]
            agent = self.agent_map[a_id]
            agent.wealth -= 1000 # Severance cost / debt?
            agent.reputation -= 0.5

    def run(self):
        print(f"Starting Phase 2 Simulation: {len(self.agents)} agents.")
        
        for env in tqdm(self.schedule, desc="Simulating Life"):
            self.current_step += 1
            self.update_market()
            
            # Interaction Step
            if self.enable_interactions:
                self.interaction_step()
            
            performance_map = {}
            
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # 1. Decision Making
                action = 'Work' # Default
                if hasattr(agent, 'choose_action'):
                    # Observe State
                    state = agent.get_state(self.market_state, env.name)
                    action = agent.choose_action(state, ['Work', 'Rest', 'Network', 'Risk'])
                    
                    # Store (for logging/learning)
                    agent.last_state = state
                    agent.last_action = action
                
                # 2. Process Action
                results = env.process_agent(agent, self.market_state, action=action)
                
                # Log Action for Analysis (Optional: only log representative sample to save memory?)
                # We'll log aggregation in collect_aggregates, or store on agent?
                if not hasattr(agent, 'action_history'): agent.action_history = {'Work':0, 'Risk':0, 'Network':0, 'Rest':0}
                agent.action_history[action] = agent.action_history.get(action, 0) + 1
                
                # 2. Update Resources
                # 3. Update Resources
                agent.update_resources(results['wealth_gain'], results['reputation_gain'])
                
                # Energy removed. Logic is now probabilistic only.
                
                # Handle Strategic Networking
                if results.get('network_success', False):
                    self.add_strategic_connection(agent)
                
                # 3. Store Performance for Competition
                if 'Career' in env.name and self.enable_interactions:
                    performance_map[agent.id] = results.get('performance', 0)
                    
                # 4. Check Survival
                agent.check_survival()
                agent.age += 1
            
            # Apply Competition if Career Phase
            if 'Career' in env.name and self.enable_interactions:
                self.apply_competition(performance_map)
                
            self.collect_aggregates(env.name)
            
        return pd.DataFrame(self.data_log)

    def collect_aggregates(self, env_name):
        alive_agents = [a for a in self.agents if a.alive]
        survival_rate = len(alive_agents) / len(self.agents) if self.agents else 0
        
        if not alive_agents: return

        # Faster numpy access?
        wealths = [a.wealth for a in alive_agents]
        # energies = [a.energy for a in alive_agents]
        
        stats = {
            'step': self.current_step,
            'phase': env_name,
            'market': self.market_state,
            'survival_rate': survival_rate,
            'avg_wealth': np.mean(wealths),
            'wealth_gini': self.calculate_gini(wealths)
        }
        self.data_log.append(stats)

    @staticmethod
    def calculate_gini(array):
        array = np.array(array, dtype=float)
        if np.amin(array) < 0: array -= np.amin(array)
        array += 1e-10
        array = np.sort(array)
        n = array.shape[0]
        index = np.arange(1, n + 1)
        total_sum = np.sum(array)
        if total_sum <= 1e-9: return 0.0 # Handle zero wealth case
        return ((np.sum((2 * index - n - 1) * array)) / (n * total_sum))

    def run_step_for_rl(self, agent_action_map=None):
        """
        Manually run one step of the simulation.
        agent_action_map: dict {agent_id: action_name}
        This is needed so the training loop can decide actions for agents,
        then the environment processes them.
        """
        if self.current_step >= len(self.schedule):
            return False # Done
            
        env = self.schedule[self.current_step]
        self.update_market()
        
        # Interaction Step (Optional in RL training if focused on individual learning? Let's keep it)
        if self.enable_interactions:
            self.interaction_step()
            
        performance_map = {}
        
        for agent in self.agents:
            if not agent.alive: continue
            
            # --- AGENT DECISION INJECTION ---
            # If agent is RLAgent, it needs to CHOOSE action based on state.
            # But normally `env.process_agent` calculates outcome.
            # We need `env` to respect the chosen action.
            # Currently Environment logic is "Auto-Pilot" (Calculates performance based on attributes).
            # To make RL meaningful, we must allow Action -> Modifies Performance/Cost.
            
            # Action Space: ['Work', 'Rest', 'Network', 'Risk']
            chosen_action = 'Work' # Default
            if hasattr(agent, 'choose_action'):
                # State observation
                state = agent.get_state(self.market_state, env.name)
                # But wait, agent.choose_action relies on Q-table.
                # Let's let the agent decide here internally if it's an RLAgent.
                chosen_action = agent.choose_action(state, ['Work', 'Rest', 'Network', 'Risk'])
                agent.last_state = state
                agent.last_action = chosen_action
            
            # Pass action to environment? 
            results = env.process_agent(agent, self.market_state, action=chosen_action)
            
            # Sanitize Results
            wg = results.get('wealth_gain', 0)
            rg = results.get('reputation_gain', 0)
            if np.isnan(wg) or np.isinf(wg): wg = 0
            if np.isnan(rg) or np.isinf(rg): rg = 0
            
            # Update Resources
            agent.update_resources(wg, rg)
            # Update Resources
            agent.update_resources(wg, rg)
            
            # Energy removed. 
            
            # Handle Strategic Networking
            if results.get('network_success', False):
                self.add_strategic_connection(agent) 
            
            # Reward Calculation for RL
            if hasattr(agent, 'learn'):
                # Define Reward Function
                # Reward = Wealth Gain? + Survival? - Burnout?
                reward = wg / 1000.0 # Normalize
                reward = wg / 1000.0 # Normalize
                if not results.get('success', False): reward -= 0.1
                # if agent.energy < 20: reward -= 0.5 # REMOVED (No energy)
                
                # Observe Next State
                next_state = agent.get_state(self.market_state, env.name) # Phase hasn't changed yet actually
                agent.learn(next_state, reward)
                # Decay Epsilon
                agent.decay_epsilon()
            
            # Action logging for RL step as well
            
            # Action logging for RL step as well
            if not hasattr(agent, 'action_history'): agent.action_history = {'Work':0, 'Risk':0, 'Network':0, 'Rest':0}
            agent.action_history[chosen_action] = agent.action_history.get(chosen_action, 0) + 1
            
            # Competition Tracking
            if 'Career' in env.name and self.enable_interactions:
                performance_map[agent.id] = results.get('performance', 0)
                
            agent.check_survival()
            agent.age += 1
            
        if 'Career' in env.name and self.enable_interactions:
            self.apply_competition(performance_map)
            
        self.current_step += 1
        return True
