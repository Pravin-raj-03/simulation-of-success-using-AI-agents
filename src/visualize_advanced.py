import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from simulation import Simulation
from config import RANDOM_SEED, POPULATION_SIZE

def generate_advanced_plots(seed=RANDOM_SEED, pop_size=POPULATION_SIZE):
    print("Running simulation for advanced analysis...")
    sim = Simulation(pop_size, seed, enable_interactions=True)
    sim.run() # This runs the full sim
    
    agents = sim.agents
    active_agents = [a for a in agents if a.alive] # Analysis on survivors usually
    # Or maybe all? Let's use survivors for wealth, but all for some things.
    
    # Dataframe for general analysis
    data = []
    for a in agents:
        data.append({
            'ID': a.id,
            'Wealth': a.wealth,
            'Sector': a.sector if a.sector else 'None',
            'Network_Size': len(a.network),
            'Luck': a.luck,
            'IQ': a.talent,
            'Alive': a.alive
        })
    df = pd.DataFrame(data)

    sns_available = False
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        sns_available = True
    except ImportError:
        plt.style.use('ggplot')

    # 1. Sector Performance (Boxplot)
    plt.figure(figsize=(10, 6))
    if sns_available:
        sns.boxplot(x='Sector', y='Wealth', data=df, showfliers=False)
        sns.stripplot(x='Sector', y='Wealth', data=df, color=".25", alpha=0.3)
    else:
        # Manual grouping
        sectors = df['Sector'].unique()
        data_to_plot = [df[df['Sector'] == s]['Wealth'] for s in sectors]
        plt.boxplot(data_to_plot, labels=sectors, showfliers=False)
        
    plt.title('Wealth by Career Sector')
    plt.ylabel('Final Wealth')
    plt.tight_layout()
    plt.savefig('sector_wealth.png')
    print("Saved sector_wealth.png")

    # 2. Network Size vs Wealth (Scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Network_Size'], df['Wealth'], alpha=0.5, c=df['Luck'], cmap='viridis')
    plt.colorbar(label='Luck')
    plt.xlabel('Network Size (Connections)')
    plt.ylabel('Final Wealth')
    plt.title('Impact of Social Network on Wealth')
    plt.yscale('symlog')
    plt.tight_layout()
    plt.savefig('network_wealth.png')
    print("Saved network_wealth.png")

    # 3. Trajectories (Line Plot of specific agents)
    # We need to extract history. 
    # Agent.history is a list of events. We need to reconstruct wealth over time.
    # This is tricky because history events are like "reward", "expend_energy". 
    # Let's pick 5 random survivors and 5 random dead agents.
    
    sample_agents = np.random.choice(agents, 10, replace=False)
    
    plt.figure(figsize=(12, 6))
    
    for agent in sample_agents:
        # Reconstruct wealth curve
        # Start wealth = 0
        w = 0
        w_curve = [0]
        # Iterate history
        # History format: {'age': age, 'event': ..., 'wealth_delta': ...}
        # Events without wealth_delta don't change wealth
        
        # Sort history by age/sequence? It should be appended in order.
        
        current_age = 0
        
        # This reconstruction is approximate if history doesn't capture every step perfectly or if multiple events per step
        # Ideally, we should have logged wealth at each step in the agent, but history is event-based.
        # Let's map age to wealth.
        
        # Filter for wealth events
        wealth_events = [e for e in agent.history if 'wealth_delta' in e]
        
        # Accumulate
        ages = []
        wealths = []
        current_w = 0
        
        for e in wealth_events:
            current_w += e.get('wealth_delta', 0)
            ages.append(e['age'])
            wealths.append(current_w)
            
        label = f"{agent.sector} (Net:{len(agent.network)})"
        style = '-' if agent.alive else ':'
        plt.plot(ages, wealths, linestyle=style, alpha=0.7, label=label)
        
    plt.title('Wealth Trajectories of Sample Agents (Solid=Alive, Dotted=Dead)')
    plt.xlabel('Age (Steps)')
    plt.ylabel('Wealth')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('wealth_trajectories.png')
    print("Saved wealth_trajectories.png")

if __name__ == "__main__":
    generate_advanced_plots()
