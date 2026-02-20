import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import Simulation
from rl_agent import RLAgent
from config import POPULATION_SIZE

def train_individual_learning(episodes=20, pop_size=200):
    print(f"--- Starting Individual Learning Simulation (N={pop_size}, Episodes={episodes}) ---")
    
    # 1. Initialize Population ONCE
    # We want to keep the SAME agents to see if they learn over "lifetimes"
    # Or, we can simulate "Reincarnation" where they keep their Q-table but reset status.
    
    # Let's create a base simulation just to generate agents with proper attributes
    params_sim = Simulation(pop_size=pop_size, seed=42)
    base_agents = params_sim.agents # These have random Talent/Adaptability
    
    # Convert to RLAgents, preserving attributes
    rl_agents = []
    for i, a in enumerate(base_agents):
        # Create RLAgent with same attributes
        new_a = RLAgent(agent_id=str(i), alpha=0.1, gamma=0.95) # Alpha/Gamma will be auto-set by attributes in __init__
        new_a.talent = a.talent
        new_a.adaptability = a.adaptability
        new_a.luck = a.luck
        new_a.id = str(i)
        
        # Explicitly initialize individual Q-Table
        new_a.q_table = {} 
        rl_agents.append(new_a)

    # --- INJECT HEAD-TO-HEAD TEST AGENTS ---
    # 1. The Genius (Max Adapt, Avg Luck)
    genius = RLAgent(agent_id="Genius", alpha=0.1, gamma=0.95)
    genius.adaptability = 95
    genius.talent = 95
    genius.luck = 0.5
    genius.q_table = {}
    rl_agents.append(genius)

    # 2. High Competence + High Luck (The "Bit Dumber but Luckier")
    lucky_comp = RLAgent(agent_id="LuckyCompetent", alpha=0.1, gamma=0.95)
    lucky_comp.adaptability = 75 # -20 points
    lucky_comp.talent = 75
    lucky_comp.luck = 0.95 # +45 points
    lucky_comp.q_table = {}
    rl_agents.append(lucky_comp)

    # 3. Average + Extreme Luck
    lucky_avg = RLAgent(agent_id="LuckyAvg", alpha=0.1, gamma=0.95)
    lucky_avg.adaptability = 50
    lucky_avg.talent = 50
    lucky_avg.luck = 0.99
    lucky_avg.q_table = {}
    rl_agents.append(lucky_avg)
    
    print("Injected 3 Test Agents: Genius, LuckyCompetent, LuckyAvg")
        
    print("Agents initialized with individual Q-Tables.")
    
    history_data = []
    
    # 2. Training Loop
    for episode in range(episodes):
        # Create a new Simulation run, but INJECT our existing agents
        # We need to reset their mutable state (Wealth, Age, etc) but KEEP Q-Table
        
        current_sim = Simulation(pop_size=pop_size, seed=None, enable_interactions=True)
        
        # Replace agents with our persistent RL agents, but reset their status
        for a in rl_agents:
            a.reset_state() # We need to ensure this method exists or do it manually
            # Manual Reset if reset_state doesn't exist or is insufficient
            a.wealth = 10000
            a.age = 18
            a.alive = True
            a.network = []
            a.energy = 100
            
            # Decay Epsilon (Learning varies by age/experience, but here by episode)
            # a.decay_epsilon() 
        
        current_sim.agents = rl_agents
        current_sim.agent_map = {a.id: a for a in rl_agents}
        
        # Run the Life Cycle
        while current_sim.run_step_for_rl():
            pass
            
        # Collect Stats at end of Episode
        wealths = [a.wealth for a in rl_agents]
        avg_wealth = np.mean(wealths)
        max_wealth = np.max(wealths)
        
        print(f"Episode {episode+1}/{episodes}: Avg Wealth = ${avg_wealth:,.0f} | Max = ${max_wealth:,.0f}")
        
        for a in rl_agents:
            history_data.append({
                'Episode': episode + 1,
                'AgentID': a.id,
                'Adaptability': a.adaptability,
                'Talent': a.talent,
                'Luck': a.luck,
                'Wealth': a.wealth,
                'Q_Size': len(a.q_table)
            })
            
    # 3. Analysis
    df = pd.DataFrame(history_data)
    
    # Bin Adaptability
    df['Adaptability_Bin'] = pd.cut(df['Adaptability'], bins=[0, 30, 70, 100], labels=['Low', 'Avg', 'High'])
    
    # Plot 1: Wealth Evolution by Adaptability
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Episode', y='Wealth', hue='Adaptability_Bin', style='Adaptability_Bin', markers=True, ci=None)
    plt.title('Wealth Growth by Adaptability (Individual Learning)')
    plt.ylabel('Average Wealth')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('individual_learning_wealth.png')
    print("Saved individual_learning_wealth.png")
    
    # Plot 2: Learning Efficiency (Q-Table Size vs Wealth)
    # Are smarter agents exploring more effectively?
    final_df = df[df['Episode'] == episodes]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_df, x='Adaptability', y='Wealth', size='Q_Size', sizes=(20, 200), alpha=0.6)
    plt.title(f'Final Wealth vs Adaptability (Episode {episodes})')
    plt.yscale('log')
    plt.savefig('individual_learning_scatter.png')
    print("Saved individual_learning_scatter.png")
    
    # Stats
    high_adapt_wealth = final_df[final_df['Adaptability_Bin'] == 'High']['Wealth'].mean()
    low_adapt_wealth = final_df[final_df['Adaptability_Bin'] == 'Low']['Wealth'].mean()
    
    print("\n--- Final Results ---")
    print(f"Avg Wealth (High Adaptability): ${high_adapt_wealth:,.0f}")
    print(f"Avg Wealth (Low Adaptability):  ${low_adapt_wealth:,.0f}")
    if low_adapt_wealth > 0:
        ratio = high_adapt_wealth / low_adapt_wealth
        print(f"High Adaptability Advantage: {ratio:.1f}x")
    else:
        print("High Adaptability Advantage: Infinite (Low Adapt Wealth is 0)")
        
    # --- New Analysis: Adaptability vs Luck ---
    print("\n--- Adaptability vs Luck Analysis ---")
    
    # Stratify by Luck AND Adaptability
    # quadrant analysis
    # Groups:
    # 1. High Adapt (>70) & High Luck (>0.7)  -> "Blessed & Smart"
    # 2. High Adapt (>70) & Low Luck (<0.3)   -> "Unlucky Genius"
    # 3. Low Adapt (<30) & High Luck (>0.7)   -> "Lucky & Stuck"
    # 4. Low Adapt (<30) & Low Luck (<0.3)    -> "Doomed"
    
    def classify(row):
        adapt = row['Adaptability']
        luck = row['Luck']
        if adapt > 70 and luck > 0.7: return 'Blessed & Smart'
        if adapt > 70 and luck < 0.3: return 'Unlucky Genius'
        if adapt < 30 and luck > 0.7: return 'Lucky & Stuck'
        if adapt < 30 and luck < 0.3: return 'Doomed'
        return 'Average'
        
    final_df['Archetype'] = final_df.apply(classify, axis=1)
    
    # Calculate Mean Wealth per Archetype
    archetype_stats = final_df.groupby('Archetype')['Wealth'].mean().sort_values(ascending=False)
    print("\nAverage Wealth by Archetype:")
    print(archetype_stats.apply(lambda x: f"${x:,.0f}"))
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=archetype_stats.index, y=archetype_stats.values, palette='viridis')
    plt.title('Impact of Adaptability vs Luck on Final Wealth')
    plt.ylabel('Average Wealth')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('adaptability_vs_luck.png')
    print("Saved adaptability_vs_luck.png")

    # Save Report
    with open('comparison_report.txt', 'w') as f:
        f.write("--- Final Results ---\n")
        f.write(f"Avg Wealth (High Adaptability): ${high_adapt_wealth:,.0f}\n")
        f.write(f"Avg Wealth (Low Adaptability):  ${low_adapt_wealth:,.0f}\n")
        if low_adapt_wealth > 0:
             f.write(f"High Adaptability Advantage: {ratio:.1f}x\n")
        
        f.write("\n--- Adaptability vs Luck Analysis ---\n")
        f.write(archetype_stats.apply(lambda x: f"${x:,.0f}").to_string())

        f.write("\n\n--- Head-to-Head: Genius vs Lucky Competent ---\n")
        # Find our injected agents
        for a in rl_agents:
            if a.id in ["Genius", "LuckyCompetent", "LuckyAvg"]:
                f.write(f"Agent {a.id}: Wealth=${a.wealth:,.0f} | Adapt={a.adaptability} | Luck={a.luck}\n")
    
    print("Saved comparison_report.txt")
        
if __name__ == "__main__":
    train_individual_learning()
