
# Simulation Configuration

# Population
POPULATION_SIZE = 1000
NUM_GENERATIONS = 1  # For now, single generation life-cycle

# Simulation Steps (Life Phases)
# Each phase has a duration in 'steps' (e.g., months or decision points)
PHASE_DURATIONS = {
    "EDUCATION": 20,
    "CAREER_EARLY": 40,
    "OPPORTUNITY": 40,
    "DECLINE": 20
}

# Attribute Initialization (Mean, Std Dev)
# Normalized roughly 0-100 or 0-1 scale depending on attribute
INIT_IQ_MEAN = 100
INIT_IQ_STD = 15

INIT_STRENGTH_MEAN = 50
INIT_STRENGTH_STD = 10

INIT_LUCK_MEAN = 0.5  # Neutral luck
INIT_LUCK_STD = 0.2

INIT_ENERGY = 100
INIT_WEALTH = 0

# Core Dynamics
PATH_DEPENDENCE_FACTOR = 0.1  # How much wealth increases future opportunity
BURNOUT_THRESHOLD = 20        # Energy level below which performance drops
RECOVERY_RATE = 25            # Energy regained per 'rest' action (Entropy breaker)

# Phase 2: Interaction & Market
MARKET_CYCLES = ['BOOM', 'NORMAL', 'RECESSION']
# Markov Transition Matrix: [From_State][To_State]
# Order: BOOM, NORMAL, RECESSION
MARKOV_TRANSITION_MATRIX = {
    'BOOM':      [0.6, 0.35, 0.05], # Likely stay Boom, or cool to Normal. Rare crash.
    'NORMAL':    [0.1, 0.8, 0.1],   # Mostly stay Normal.
    'RECESSION': [0.05, 0.35, 0.6]  # Recessions are sticky, but eventually recover.
}

SECTORS = {
    'STABLE': {'risk': 0.05, 'reward_mult': 1.0, 'burnout_factor': 0.8}, # Gov/Service
    'VOLATILE': {'risk': 0.3, 'reward_mult': 3.0, 'burnout_factor': 1.5} # Tech/Finance
}

CONNECTION_LIMIT = 50 # Max network size
NETWORKING_BASE_COST = 10 # Energy cost to network
NETWORKING_SUCCESS_RATE = 0.3 # Base chance to make a good connection

# Randomness
# Randomness
RANDOM_SEED = 42

# --- SCIENTIFIC MODEL CONSTANTS ---

# 1. Rasch Model Weights (Ability Calculation)
# Ability = (Talent * W_T) + (Prep * W_P) + (Luck * W_L)
RASCH_PARAMS = {
    'TALENT_WEIGHT': 1.0,  # Primary Driver
    'PREP_WEIGHT': 1.0,    # Consistency Driver
    'LUCK_WEIGHT': 1.0     # Equal weight to Talent (Skill Multiplier)
}

# Luck Modifiers
LUCK_OPPORTUNITY_FACTOR = 1.0 # Multiplier for opportunity arrival rate (1.0 = Luck doubles opportunities)

# 2. Phase Difficulty (Logits)
# 0.0 = 50% success for average agent. >0 = Harder. <0 = Easier.
PHASE_DIFFICULTY = {
    'EDUCATION': -1.0,      # Easy (Most pass)
    'CAREER_STABLE': 0.5,   # Moderate
    'CAREER_VOLATILE': 1.5, # Hard
    'OPPORTUNITY': 2.0,     # Very Hard (Investment/Startup)
    'DECLINE': 1.0          # Health Shocks
}

# 3. Market Modifiers (Impact on Difficulty)
MARKET_MODIFIERS = {
    'BOOM': -0.5,      # Makes things easier
    'NORMAL': 0.0,
    'RECESSION': 1.0   # Makes things much harder
}

# 4. Economics
BASE_SALARY = 1000

# 5. RL Agent Parameters
WEALTH_BINS = [500, 5000, 20000] # Poor, Mid, Rich, Ultra
NETWORK_BINS = [5, 20] # Isolated, Connected, Influential

