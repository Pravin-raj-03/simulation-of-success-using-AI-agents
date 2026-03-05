import uuid
import numpy as np
from config import (
    INIT_IQ_MEAN, INIT_IQ_STD, INIT_STRENGTH_MEAN, INIT_STRENGTH_STD,
    INIT_LUCK_MEAN, INIT_LUCK_STD, INIT_ENERGY, INIT_WEALTH,
    BURNOUT_THRESHOLD, RECOVERY_RATE
)
from utils import truncated_normal

class Agent:
    def __init__(self, agent_id=None):
        self.id = agent_id if agent_id else str(uuid.uuid4())
        
        # Inherent Attributes (Fixed or slowly changing)
        # 0-150 Scale for IQ, 0-100 for Strength/Adaptability
        # Inherent Attributes (Fixed or slowly changing)
        # Talent (formerly IQ): 0-150 Scale. Represents raw cognitive/physical ability.
        self.talent = truncated_normal(INIT_IQ_MEAN, INIT_IQ_STD, 50, 180)
        
        # Adaptability / Preparation: Represents "Hard Work" & "Readiness".
        # High score means agent is better prepared for difficulty.
        self.adaptability = truncated_normal(50, 15, 10, 100) 
        
        self.strength = truncated_normal(INIT_STRENGTH_MEAN, INIT_STRENGTH_STD, 10, 100)
        
        # Dynamic Attributes
        self.luck = truncated_normal(INIT_LUCK_MEAN, INIT_LUCK_STD, 0, 1) # 0 to 1, 0.5 neutral
        
        # WEALTH INITIALIZATION (Normal Distribution)
        # Emergent outcome, not forced Pareto.
        self.wealth = np.random.normal(INIT_WEALTH + 1000, 200) # Mean 1000, Std 200
        self.wealth = max(100, self.wealth) # Floor
        
        self.reputation = 0.0
        self.age = 0
        self.alive = True
        
        # Phase 2: Social & Career
        self.network = [] # List of agent IDs
        self.sector = None # 'STABLE' or 'VOLATILE'
        
        # Stats tracking
        self.history = []

    def update_resources(self, monetary_reward, social_reward, incident_desc=None):
        self.wealth += monetary_reward
        self.reputation += social_reward
        
        event_entry = {
            'age': self.age, 
            'event': 'reward' if not incident_desc else 'incident', 
            'wealth_delta': monetary_reward, 
            'rep_delta': social_reward
        }
        if incident_desc:
            event_entry['description'] = incident_desc
            
        self.history.append(event_entry)

    def check_survival(self):
        # Base survival probability
        survival_prob = 1.0
        
        # Example factors
        if self.age > 80:
            pass # survival_prob -= 0.2
            
        # if np.random.random() > survival_prob:
        #     self.alive = False
        #     self.history.append({'age': self.age, 'event': 'death'})
        
        # IMMORTALITY MODE: Always Alive
        self.alive = True
        return self.alive

    def __repr__(self):
        return (f"Agent(ID={self.id[:4]}, Talent={self.talent:.1f}, Wealth={self.wealth:.1f}, "
                f"Alive={self.alive})")
