import numpy as np
import pandas as pd
from simulation import Simulation
from rl_agent import RLAgent
from config import POPULATION_SIZE, RANDOM_SEED

# We need to modify Simulation loop slightly to allow custom agent actions
# Or we can create a specialized training loop here that manually steps the agents.

def train_rl_agents(episodes=5): # Short run for verification
    print(f"Training RL Agents for {episodes} episodes...")
    
    # We will use one population that "reincarnates" to keep their Q-tables?
    # Or share a global Q-table?
    # Individual learning is slower. Let's start with individual.
    
    # Initialize "Global Best Brain" to track average learning
    global_q_table = {} 
    
    training_data = []

    # Baseline Run (Random Actions)
    print("Running Baseline (Random Actions)...")
    sim_base = Simulation(pop_size=100, seed=42, enable_interactions=True)
    # RLAgents with epsilon=1.0 (Random) that never decay
    sim_base.agents = [RLAgent(agent_id=str(i), epsilon=1.0) for i in range(100)]
    sim_base.agent_map = {a.id: a for a in sim_base.agents}
    while sim_base.run_step_for_rl(): pass
    
    survivors_base = sim_base.agents # All agents survive
    if survivors_base:
        baseline_wealth = np.mean([a.wealth for a in survivors_base])
    else:
        baseline_wealth = 0
        
    print(f"Baseline Random Wealth: {baseline_wealth:,.0f}")

    print(f"Training RL Agents for {episodes} episodes...")

    # Shared Brain
    global_q_table = {}

    for episode in range(episodes):
        # Custom Simulation Logic for RL training
        sim = Simulation(pop_size=100, seed=None, enable_interactions=True) # Smaller batch for speed
        
        # Decay epsilon based on episode progress
        # LINEAR Annealing for simplicity in this loop logic
        eps = max(0.05, 1.0 - (episode / (episodes * 0.8)))
        
        # Pass shared Q-table reference
        sim.agents = [RLAgent(agent_id=str(i), epsilon=eps, q_table=global_q_table) for i in range(100)]
        sim.agent_map = {a.id: a for a in sim.agents}
        
        while True:
            running = sim.run_step_for_rl()
            if not running: break
        
        # After run, collect stats
        # survivors = [a for a in sim.agents if a.alive] # All alive
        all_wealth = [a.wealth for a in sim.agents]
        
        if all_wealth:
            avg_wealth = np.mean(all_wealth)
        else:
            avg_wealth = 0 # Extinction
            
        max_wealth_ever = max(all_wealth) if all_wealth else 0
        
        # Record
        training_data.append({
            'Episode': episode, 
            'Avg_Wealth': avg_wealth,
            'Max_Wealth': max_wealth_ever,
            'Epsilon': eps,
            'Baseline': baseline_wealth
        })
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Avg Wealth = {avg_wealth:,.0f} | Max Wealth = {max_wealth_ever:,.0f}")
            
    # Debug Q-Table
    print(f"\nGlobal Q-Table Size: {len(global_q_table)} states")
    for s in list(global_q_table.keys())[:5]:
        print(f"State {s}: {global_q_table[s]}")
    
    vals = []
    for s in global_q_table:
        vals.extend(global_q_table[s].values())
    
    if vals:
        print(f"Q-Value Stats: Min={min(vals):.3f}, Max={max(vals):.3f}, Mean={np.mean(vals):.3f}")
    else:
        print("Q-Table is EMPTY!")
        
    df = pd.DataFrame(training_data)
    
    # Plot Learning Curve with Baseline
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Wealth', color='tab:blue')
    ax1.plot(df['Episode'], df['Avg_Wealth'], color='tab:blue', label='RL Agent')
    ax1.axhline(y=baseline_wealth, color='r', linestyle='--', label='Random Baseline')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon (Exploration)', color='tab:gray')
    ax2.plot(df['Episode'], df['Epsilon'], color='tab:gray', linestyle=':', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='tab:gray')
    
    plt.title("RL Agent Learning Curve vs Baseline")
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('rl_learning_curve.png') # Overwrite
    print("Saved improved rl_learning_curve.png")
    
    return df

if __name__ == "__main__":
    train_rl_agents()
