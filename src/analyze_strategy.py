import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import Simulation
from rl_agent import RLAgent
from config import POPULATION_SIZE, RANDOM_SEED

def analyze_strategy():
    print("--- Strategic Analysis: Do Geniuses Learn to Take Risks? ---")
    
    # Use RLAgents specifically
    # Simulation creates standard Agents by default, need to override
    sim = Simulation(pop_size=1000, seed=42)
    
    # Replace agents with RLAgents
    # We maintain the random attributes by copying them or just re-initializing
    # Better to just re-init carefully or modify Simulation to accept agent class
    # Create RLAgents
    print("Initializing RL Population...")
    rl_agents = []
    for i in range(100): # Reduce pop size for speed during training
        a = RLAgent(agent_id=f"RL_{i}", epsilon=1.0)
        # Faster decay for this experiment
        a.epsilon_decay = 0.95 
        rl_agents.append(a)
    
    # Training Loop
    NUM_EPISODES = 50
    print(f"Training Agents for {NUM_EPISODES} Episodes...")
    
    for episode in range(NUM_EPISODES):
        # Reset Agents for new life, but KEEP Q-Table
        for a in rl_agents:
            a.alive = True
            a.age = 18
            a.wealth = 0
            a.reputation = 0
            a.network = []
            a.network_quality = 0
            # Epsilon decays naturally over steps, but we might want to reset? 
            # No, let it decay.
        
        sim = Simulation(pop_size=len(rl_agents), seed=42+episode)
        sim.agents = rl_agents
        sim.agent_map = {a.id: a for a in sim.agents}
        sim.run()
        
        # Monitor Epsilon
        if episode % 10 == 0:
            avg_eps = np.mean([a.epsilon for a in rl_agents])
            print(f"Episode {episode}: Avg Epsilon = {avg_eps:.4f}")

    print("Training Complete. Running Final Test Episode...")
    # Final Run for Analysis
    for a in rl_agents:
        a.alive = True
        a.age = 18
        a.wealth = 0
        a.reputation = 0
        a.network = []
        a.epsilon = 0.05 # Force exploitation
        a.action_history = {} # Reset history logging
        
    sim = Simulation(pop_size=len(rl_agents), seed=999)
    sim.agents = rl_agents
    sim.agent_map = {a.id: a for a in sim.agents}
    sim.run()
    
    # Collect Data
    data = []
    for a in sim.agents:
        if not hasattr(a, 'action_history'): continue
        
        total_actions = sum(a.action_history.values())
        if total_actions == 0: continue
        
        risk_rate = a.action_history.get('Risk', 0) / total_actions
        work_rate = a.action_history.get('Work', 0) / total_actions
        net_rate = a.action_history.get('Network', 0) / total_actions
        
        data.append({
            'Talent': a.talent,
            'Wealth': a.wealth,
            'Risk_Rate': risk_rate,
            'Work_Rate': work_rate,
            'Network_Rate': net_rate,
            'Group': 'Genius' if a.talent > 115 else ('Average' if a.talent > 90 else 'Low')
        })
        
    df = pd.DataFrame(data)
    
    print("\n--- Strategy Analysis Results (Post-Training) ---")
    summary_df = df.groupby('Group')[['Risk_Rate', 'Work_Rate', 'Network_Rate', 'Wealth']].mean()
    print(summary_df)
    with open('strategy_report.txt', 'w') as f:
        f.write(summary_df.to_string())
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Risk Taking vs Talent
    # Bin talent
    df['Talent_Bin'] = pd.cut(df['Talent'], bins=10)
    summary = df.groupby('Talent_Bin')[['Risk_Rate', 'Wealth']].mean().reset_index()
    summary['Talent_Mid'] = summary['Talent_Bin'].apply(lambda x: x.mid)
    
    sns.lineplot(data=summary, x='Talent_Mid', y='Risk_Rate', marker='o', color='red', label='Risk Taking Rate')
    plt.title('Do Talented Agents Take More Risks?')
    plt.xlabel('Talent Score')
    plt.ylabel('Proportion of "Risk" Actions')
    plt.grid(True)
    plt.savefig('strategy_risk_vs_talent.png')
    print("Saved strategy_risk_vs_talent.png")
    
    # Gini Coefficient Logic is in data_log of sim
    results_df = pd.DataFrame(sim.data_log)
    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x='step', y='wealth_gini')
        plt.title('Evolution of Inequality (Gini)')
        plt.xlabel('Time Step')
        plt.ylabel('Gini Coefficient')
        plt.grid(True)
        plt.savefig('inequality_evolution.png')
        print("Saved inequality_evolution.png")

if __name__ == "__main__":
    analyze_strategy()
