import numpy as np
from config import (
    PATH_DEPENDENCE_FACTOR, SECTORS, NETWORKING_BASE_COST, 
    NETWORKING_SUCCESS_RATE, RASCH_PARAMS, PHASE_DIFFICULTY, 
    MARKET_MODIFIERS, BASE_SALARY
)

def sigmoid(x):
    """Logistic Function for Rasch Model"""
    return 1 / (1 + np.exp(-x))

class Environment:
    def __init__(self, name):
        self.name = name
    
    def calculate_success_probability(self, agent, difficulty, market_state):
        """
        Rasch Model: P(Success) = sigmoid(Ability - Difficulty)
        Reference: "Standard Logistic Rasch Model" for Item Response Theory.
        Ability = Sum of Weighted Z-Scores (Talent + Prep + Luck)
        """
        # 1. Normalize Ability Components (Z-Score approximation)
        norm_talent = (agent.talent - 100) / 15.0
        norm_prep = (agent.adaptability - 50) / 15.0
        
        # Luck: Mean 0.5 -> Shift to -1 to +1 scale
        luck_bias = (agent.luck - 0.5) * 2.0 
        
        # Total Ability (Weighted by Config)
        ability = (
            (norm_talent * RASCH_PARAMS['TALENT_WEIGHT']) + 
            (norm_prep * RASCH_PARAMS['PREP_WEIGHT']) + 
            (luck_bias * RASCH_PARAMS['LUCK_WEIGHT'])
        )
        
        # 2. Difficulty (Logits)
        log_odds = ability - difficulty
        probability = sigmoid(log_odds)
        
        return probability

    def process_agent(self, agent, market_state='NORMAL', action='Work'):
        if action == 'Rest':
             return {
                'wealth_gain': -5, # Cost of living
                'reputation_gain': 0,
                'success': True,
                'performance': 0
            }
            
        elif action == 'Network':
            # Networking Check
            difficulty = PHASE_DIFFICULTY['CAREER_STABLE'] + 0.5 # Slightly harder than stable job
            prob = self.calculate_success_probability(agent, difficulty, market_state)
            is_success = np.random.random() < prob
            
            desc = "Successfully expanded professional network." if is_success else "Attempted to network, but gained little traction."
            return {
                'wealth_gain': -15, 
                'reputation_gain': 0.5 if is_success else 0.1,
                'success': is_success,
                'description': desc,
                'network_success': is_success, 
                'performance': 0
            }
        
        res = self._process_phase_logic(agent, market_state, action)
        if 'description' not in res:
            res['description'] = f"Standard activity in {self.name}."
        return res

    def _process_phase_logic(self, agent, market_state, action):
        raise NotImplementedError

    def post_process_population(self, agents):
        pass

class EducationPhase(Environment):
    def __init__(self):
        super().__init__("Education")
    
    def _process_phase_logic(self, agent, market_state='NORMAL', action='Work'):
        difficulty = PHASE_DIFFICULTY['EDUCATION']
        
        prob = self.calculate_success_probability(agent, difficulty, market_state)
        success = np.random.random() < prob
        
        # Performance is just the probability * 100 + noise
        performance = (prob * 100) + np.random.normal(0, 5)
        
        if success:
            desc = "Graduated with honors!" if performance > 90 else "Completed educational tier successfully."
        else:
            desc = "Struggled with academic requirements."
            
        return {
            'wealth_gain': 0,
            'reputation_gain': 0.5 if success else -0.1,
            'success': success,
            'description': desc,
            'performance': performance
        }

class CareerPhase(Environment):
    def __init__(self, stage="Early"):
        super().__init__(f"Career ({stage})")
        self.stage = stage
        
    def _process_phase_logic(self, agent, market_state='NORMAL', action='Work'):
        if agent.sector is None:
            # High talent prefers Volatile
            if agent.talent > 115: agent.sector = 'VOLATILE'
            else: agent.sector = 'STABLE'
        
        sector_params = SECTORS[agent.sector]
        
        # Determine Base Difficulty from Config
        base_difficulty = PHASE_DIFFICULTY['CAREER_STABLE']
        if agent.sector == 'VOLATILE':
            base_difficulty = PHASE_DIFFICULTY['CAREER_VOLATILE']
            
        # Apply Market Modifiers
        market_mod = MARKET_MODIFIERS.get(market_state, 0.0)
        difficulty = base_difficulty - market_mod # Negative modifier = Easier, so Subtract?
        # WAIT: Config says BOOM = -0.5. 
        # Logic: Difficulty = Base + Modifier? 
        # If Boom (-0.5), Diff should decrease. So Add.
        difficulty = base_difficulty + market_mod
        
        if action == 'Risk': difficulty += 1.0 # Constant risk penalty
        
        # Calculate Success
        prob = self.calculate_success_probability(agent, difficulty, market_state)
        success = np.random.random() < prob
        
        wealth_gain = 0
        desc = ""
        if success:
            # Reward
            base_salary = BASE_SALARY
            multiplier = sector_params['reward_mult']
            if action == 'Risk': 
                multiplier *= 2.0
                desc = f"High-risk project in {agent.sector} paid off handsomely!"
            else:
                desc = f"Steady progress in the {agent.sector} sector."
            
            wealth_gain = base_salary * multiplier
        else:
            # Failure Cost
            if action == 'Risk': 
                wealth_gain = -500
                desc = f"Aggressive move in {agent.sector} failed; incurred costs."
            else:
                desc = f"Difficult year in the {agent.sector} sector; minimal growth."
        
        return {
            'wealth_gain': wealth_gain,
            'reputation_gain': wealth_gain / 2000,
            'success': success,
            'description': desc,
            'performance': wealth_gain
        }

class OpportunityPhase(Environment):
    def __init__(self):
        super().__init__("Opportunity & Risk")
    
    def generate_opportunities(self, agent):
        """
        Poisson Process for Opportunity Generation
        Lambda is modulated by Luck.
        """
        import numpy as np
        from config import LUCK_OPPORTUNITY_FACTOR
        
        # Base arrival rate (e.g., 0.05 per step = 1 opportunity every 20 steps for average)
        base_lambda = 0.05 
        
        # Luck Multiplier:
        # Agent Luck 0.0 -> Multiplier 0.5 (Unlucky)
        # Agent Luck 0.5 -> Multiplier 1.0 (Neutral)
        # Agent Luck 1.0 -> Multiplier 1.5 (Lucky)
        # We can adjust the slope with LUCK_OPPORTUNITY_FACTOR
        luck_mod = (agent.luck - 0.5) * 2.0 * LUCK_OPPORTUNITY_FACTOR
        # Effective lambda
        lambda_agent = base_lambda * (1 + luck_mod)
        lambda_agent = max(0.01, lambda_agent) # Minimum floor
        
        # Poisson Check
        if np.random.random() < lambda_agent:
            return True
        return False

    def _process_phase_logic(self, agent, market_state='NORMAL', action='Work'):
        # Investment Logic
        difficulty = PHASE_DIFFICULTY['OPPORTUNITY']
        
        # Network helps reduce difficulty
        net_quality = getattr(agent, 'network_quality', 0)
        difficulty -= (net_quality * 1.5) 
        
        prob = self.calculate_success_probability(agent, difficulty, market_state)
        success = np.random.random() < prob
        
        # Windfall Chance (Pure Luck)
        # 1% chance to succeed regardless of difficulty if Luck > 0.8
        if not success and agent.luck > 0.8:
            if np.random.random() < 0.01:
                success = True
                # Log windfall?
        
        capital = max(100, agent.wealth * 0.1)
        wealth_gain = 0
        desc = ""
        
        if success:
            # Normal Distribution Return (Emergent Wealth)
            roi = np.random.normal(1.5, 0.5)
            roi = max(-0.5, roi) 
            wealth_gain = capital * roi
            
            if roi > 2.0:
                desc = "Major investment windfall! A stroke of incredible luck."
            else:
                desc = "Capitalized on a solid market opportunity."
        else:
             loss_pct = np.random.normal(0.3, 0.1)
             loss_pct = max(0.0, min(1.0, loss_pct))
             wealth_gain = -capital * loss_pct
             desc = "Investment opportunity soured; lost significant capital."
            
        return {
            'wealth_gain': wealth_gain,
            'reputation_gain': wealth_gain / 5000,
            'success': success,
            'description': desc,
            'performance': wealth_gain
        }

class DeclinePhase(Environment):
    def __init__(self):
        super().__init__("Decline & Crisis")
    
    def _process_phase_logic(self, agent, market_state='NORMAL', action='Work'):
        difficulty = PHASE_DIFFICULTY['DECLINE']
        
        prob_survival = self.calculate_success_probability(agent, difficulty, market_state)
        
        damage = 0
        desc = "Managed to navigate local crises without loss."
        if np.random.random() > prob_survival:
            damage = 1000
            desc = "Hit by a significant life crisis; wealth depleted."
            
        return {
            'wealth_gain': -damage,
            'reputation_gain': 0,
            'success': damage == 0,
            'description': desc,
            'performance': 0
        }
