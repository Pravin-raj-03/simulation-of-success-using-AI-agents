import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from simulation import Simulation
from config import RANDOM_SEED, POPULATION_SIZE

def run_comparison():
    print("Running Model A: Just Agents (MVP)...")
    sim_a = Simulation(pop_size=POPULATION_SIZE, seed=RANDOM_SEED, enable_interactions=False)
    sim_a.run()
    
    print("\nRunning Model B: With Society (Interactions/Competition/Market)...")
    sim_b = Simulation(pop_size=POPULATION_SIZE, seed=RANDOM_SEED, enable_interactions=True)
    sim_b.run()
    
    # Analyze Final Wealth
    wealth_a = [a.wealth for a in sim_a.agents]
    wealth_b = [a.wealth for a in sim_b.agents]
    
    avg_a = np.mean(wealth_a) if wealth_a else 0
    avg_b = np.mean(wealth_b) if wealth_b else 0
    
    print(f"\n--- Results (Assets per Agent) ---")
    print(f"Model A Avg Wealth: {avg_a:,.0f}")
    print(f"Model B Avg Wealth: {avg_b:,.0f}")
    
    # Plot Distribution
    plt.figure(figsize=(10, 6))
    
    sns_available = False
    try:
        import seaborn as sns
        sns_available = True
    except ImportError:
        pass

    if sns_available:
        sns.set_theme(style="whitegrid")
        # Creating a combined DataFrame
        df_a = pd.DataFrame({'Wealth': wealth_a, 'Model': 'Solo (MVP)'})
        df_b = pd.DataFrame({'Wealth': wealth_b, 'Model': 'Society (Interaction)'})
        df = pd.concat([df_a, df_b])
        
        # Log scale for wealth is barely useful if there are negatives, but assuming pos
        sns.kdeplot(data=df, x='Wealth', hue='Model', fill=True, common_norm=False, log_scale=False)
        plt.title('Wealth Distribution: Solo vs Society')
        
    else:
        plt.style.use('ggplot')
        plt.hist(wealth_a, alpha=0.5, label='Solo (MVP)', bins=30, density=True)
        plt.hist(wealth_b, alpha=0.5, label='Society (Interaction)', bins=30, density=True)
        plt.title('Wealth Distribution: Solo vs Society')
        plt.xlabel('Final Wealth')
        plt.legend()
        
    plt.savefig('comparison_plot.png')
    print("Saved comparison_plot.png")

if __name__ == "__main__":
    run_comparison()
