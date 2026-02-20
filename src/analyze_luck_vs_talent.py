import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from simulation import Simulation
from config import RANDOM_SEED, POPULATION_SIZE

def analyze_groups(seed=RANDOM_SEED, pop_size=POPULATION_SIZE):
    print("Re-running simulation for group comparison...")
    sim = Simulation(pop_size, seed)
    sim.run()
    
    # Extract Data
    agents = sim.agents
    data = []
    for a in agents:
        # Define Talent as IQ (Mental) + Strength (Physical)
        # Normalize roughly: IQ is mean 100, Strength mean 50. 
        # Z-score-ish approach or just sum raw? Let's use IQ as primary proxy for "Talent" per user context, 
        # but let's do a composite to be robust. 
        # talent_score = a.iq + a.strength
        # Actually simplest is just IQ as "Talent" for this specific question as it's the classic trope.
        
        data.append({
            'Luck': a.luck,
            'Talent': a.iq, # Using IQ as Talent
            'Final_Wealth': a.wealth,
            'Alive': a.alive
        })
    
    df = pd.DataFrame(data)
    
    # Define Thresholds (Top/Bottom 25%)
    high_luck_thresh = df['Luck'].quantile(0.75)
    low_luck_thresh = df['Luck'].quantile(0.25)
    
    high_talent_thresh = df['Talent'].quantile(0.75)
    low_talent_thresh = df['Talent'].quantile(0.25)
    
    # Create Groups
    # Group 1: High Luck + Low Talent
    g1 = df[(df['Luck'] > high_luck_thresh) & (df['Talent'] < low_talent_thresh)]
    
    # Group 2: Low Luck + High Talent
    g2 = df[(df['Luck'] < low_luck_thresh) & (df['Talent'] > high_talent_thresh)]
    
    print(f"\n--- Comparison Results ({len(g1)} agents vs {len(g2)} agents) ---")
    print(f"Group 1 (Fortunate Fools) - Avg Wealth:: {g1['Final_Wealth'].mean():,.2f}")
    print(f"Group 2 (Unlucky Geniuses) - Avg Wealth: {g2['Final_Wealth'].mean():,.2f}")
    
    print(f"Group 1 Survival Rate: {g1['Alive'].mean()*100:.1f}%")
    print(f"Group 2 Survival Rate: {g2['Alive'].mean()*100:.1f}%")
    
    # Plot Boxplot
    data_to_plot = [g1['Final_Wealth'], g2['Final_Wealth']]
    
    plt.figure(figsize=(10, 6))
    
    # Check seaborn
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        # Combine for seaborn
        g1_df = g1.copy()
        g1_df['Group'] = 'High Luck / Low Talent'
        g2_df = g2.copy()
        g2_df['Group'] = 'Low Luck / High Talent'
        combined = pd.concat([g1_df, g2_df])
        
        sns.boxplot(x='Group', y='Final_Wealth', data=combined, showfliers=False)
        sns.stripplot(x='Group', y='Final_Wealth', data=combined, color=".25", alpha=0.5)
        
    except ImportError:
        plt.style.use('ggplot')
        plt.boxplot(data_to_plot, labels=['High Luck / Low Talent', 'Low Luck / High Talent'], showfliers=False)
        
    plt.title('Wealth Outcome Comparison: Luck vs Talent')
    plt.ylabel('Final Wealth')
    
    plt.tight_layout()
    plt.savefig('luck_vs_talent.png')
    print("Saved luck_vs_talent.png")

if __name__ == "__main__":
    analyze_groups()
