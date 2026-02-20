import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulation import Simulation
from config import POPULATION_SIZE, RANDOM_SEED

def run_monte_carlo(runs=1000, pop_size=POPULATION_SIZE):
    print(f"--- Starting Monte Carlo Study ({runs} Runs) ---")
    print(f"Goal: Prove statistical stability and reproducibility.")
    
    results = []
    
    # We will track: Survival Rate, Gini Coefficient, Top 1% Wealth Share
    
    for i in tqdm(range(runs), desc="Monte Carlo Progress"):
        # Use a deterministic seed for each run to ensure this specific study is reproducible
        # Run 0 is always Seed 0, etc.
        current_seed = i 
        
        sim = Simulation(pop_size, seed=current_seed, enable_interactions=True)
        # Run silently (no printing inside sim if possible, but our sim prints... we might see a lot of noise)
        # Ideally we'd suppress output, but for now let's just let it run.
        
        df_log = sim.run() 
        
        # Extract Final Metrics
        final_state = df_log.iloc[-1]
        
        # Calculate Top 1% Wealth Share manually from agents (since log only has avg)
        # We need access to agents again.
        agents = sim.agents
        wealths = sorted([a.wealth for a in agents if a.alive], reverse=True)
        total_wealth = sum(wealths)
        
        top_1_percent_count = max(1, int(len(wealths) * 0.01))
        top_1_percent_wealth = sum(wealths[:top_1_percent_count])
        
        share_1pct = top_1_percent_wealth / total_wealth if total_wealth > 0 else 0
        
        results.append({
            'Seed': current_seed,
            'Survival_Rate': final_state['survival_rate'],
            'Gini': final_state['wealth_gini'],
            'Avg_Wealth': final_state['avg_wealth'],
            'Top_1_Percent_Share': share_1pct
        })
        
    df_results = pd.DataFrame(results)
    
    # Analysis
    print("\n--- Monte Carlo Results Summary ---")
    print(df_results.describe())
    
    # Convergence Check
    # Plot how the Mean Gini converges as N increases
    cumulative_gini_mean = df_results['Gini'].expanding().mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_gini_mean, label='Cumulative Mean Gini')
    plt.xlabel('Number of Runs')
    plt.ylabel('Mean Gini Coefficient')
    plt.title('Convergence of Gini Coefficient over 1000 Runs')
    plt.grid(True)
    plt.legend()
    plt.savefig('monte_carlo_convergence.png')
    print("Saved monte_carlo_convergence.png")
    
    # Distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df_results['Gini'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Gini Coefficients')
    plt.xlabel('Gini')
    
    plt.subplot(1, 2, 2)
    plt.hist(df_results['Survival_Rate'], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Survival Rates')
    plt.xlabel('Survival Rate')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_distributions.png')
    print("Saved monte_carlo_distributions.png")
    
    # Save raw data
    df_results.to_csv('monte_carlo_results_1000.csv', index=False)
    print("Saved monte_carlo_results_1000.csv")
    
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1000, help='Number of Monte Carlo runs')
    args = parser.parse_args()
    
    run_monte_carlo(args.runs)
