import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import RANDOM_SEED, POPULATION_SIZE
from simulation import Simulation

def analyze_luck_v2(seed=RANDOM_SEED, pop_size=POPULATION_SIZE):
    print("Running simulation (v2) for luck analysis...")
    sim = Simulation(pop_size, seed)
    sim.run()
    
    agents = sim.agents
    data = []
    
    # Track opportunities as well if possible, but let's stick to wealth first
    for a in agents:
        # Count opportunities from history?
        # environment.py doesn't seem to log "opportunity" event in history yet, 
        # but let's check wealth.
        data.append({
            'Luck': a.luck,
            'Talent': a.talent,
            'Final_Wealth': a.wealth,
            'Alive': a.alive
        })
    
    df = pd.DataFrame(data)
    
    # Correlation
    corr = df['Luck'].corr(df['Final_Wealth'])
    print(f"Correlation (Luck vs Wealth): {corr:.4f}")
    
    # Top 10% vs Bottom 10%
    top_luck = df[df['Luck'] > 0.9]['Final_Wealth'].mean()
    bot_luck = df[df['Luck'] < 0.1]['Final_Wealth'].mean()
    print(f"Avg Wealth (Luck > 0.9): {top_luck:,.2f}")
    print(f"Avg Wealth (Luck < 0.1): {bot_luck:,.2f}")
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.scatter(df['Luck'], df['Final_Wealth'], alpha=0.5, c=df['Talent'], cmap='viridis')
    plt.colorbar(label='Talent')
    plt.xlabel('Luck')
    plt.ylabel('Wealth')
    plt.title(f'Luck vs Wealth (Corr: {corr:.2f})')
    plt.yscale('symlog')
    plt.savefig('luck_analysis_v2.png')
    print("Saved luck_analysis_v2.png")

if __name__ == "__main__":
    analyze_luck_v2()
