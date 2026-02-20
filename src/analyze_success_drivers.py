
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import Simulation
from config import RANDOM_SEED, POPULATION_SIZE, MARKOV_TRANSITION_MATRIX

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

import sys

def analyze_success_drivers():
    # Parse Pop Size
    pop_size = 1000
    if len(sys.argv) > 1:
        pop_size = int(sys.argv[1])
        
    print(f"--- 1. Generating Data from Rasch Model Simulation (N={pop_size}) ---")
    sim = Simulation(pop_size=pop_size, seed=42)
    sim.run()
    
    # Extract Agent Data
    data = []
    for a in sim.agents:
        data.append({
            'Talent': a.talent,
            'Preparation': a.adaptability,
            'Luck': a.luck,
            'Wealth': a.wealth,
            'Success': 1 if a.wealth > 10000 else 0 # Threshold for "Success"
        })
    df = pd.DataFrame(data)
    
    print("--- 2. Verifying the S-Curve (Rasch Model) ---")
    # Theoretical Ability = Normal(Talent) + Normal(Prep) + Luck
    # We reconstruct the 'Ability' score manually
    df['Norm_Talent'] = (df['Talent'] - 100) / 15.0
    df['Norm_Prep'] = (df['Preparation'] - 50) / 15.0
    df['Luck_Bias'] = (df['Luck'] - 0.5) * 2.0
    df['Total_Ability'] = df['Norm_Talent'] + df['Norm_Prep'] + (df['Luck_Bias'] * 0.5)
    
    # Plot Success Rate vs Total Ability
    # We bin ability to see the curve
    df['Ability_Bin'] = pd.cut(df['Total_Ability'], bins=20)
    
    # Calculate probability of being "Wealthy" (>Median) per bin
    median_wealth = df['Wealth'].median()
    df['Is_Wealthy'] = (df['Wealth'] > median_wealth).astype(int)
    
    binned = df.groupby('Ability_Bin')['Is_Wealthy'].mean().reset_index()
    # Convert bin intervals to midpoints for plotting
    binned['Ability_Score'] = binned['Ability_Bin'].apply(lambda x: x.mid)
    
    plt.figure(figsize=(10, 6))
    
    # Theoretical Curve
    x = np.linspace(-3, 3, 100)
    y = sigmoid(x) # Difficulty approx 0 relative to centered ability
    plt.plot(x, y, 'r--', label='Theoretical Rasch Curve (Sigmoid)', linewidth=2)
    
    # Empirical Data
    sns.scatterplot(data=binned, x='Ability_Score', y='Is_Wealthy', s=100, color='blue', label='Simulation Data')
    
    plt.title('Validation: Does Success follow the Rasch Law?')
    plt.xlabel('Agent Ability (Talent + Prep + Luck)')
    plt.ylabel('Probability of Being in Top 50% Wealth')
    plt.grid(True)
    plt.legend()
    plt.savefig('rasch_model_validation.png')
    print("Saved rasch_model_validation.png")
    
    print("\n--- 3. Attribution Analysis (Talent vs Prep vs Luck) ---")
    # Linear Regression to find coefficients
    from sklearn.linear_model import LinearRegression
    X = df[['Norm_Talent', 'Norm_Prep', 'Luck_Bias']]
    
    # Handle negative wealth for Log transformation
    # We shift wealth so minimum is positive
    min_wealth = df['Wealth'].min()
    offset = abs(min_wealth) + 100
    y = np.log(df['Wealth'] + offset) # Log(Wealth + large_offset)
    
    reg = LinearRegression().fit(X, y)
    coeffs = reg.coef_
    total_impact = np.sum(np.abs(coeffs))
    
    importance = {
        'Talent': abs(coeffs[0]) / total_impact,
        'Preparation (Hard Work)': abs(coeffs[1]) / total_impact,
        'Luck': abs(coeffs[2]) / total_impact
    }
    
    print(f"Success Driver Attribution:")
    print(f"Talent:      {importance['Talent']*100:.1f}%")
    print(f"Preparation: {importance['Preparation (Hard Work)']*100:.1f}%")
    print(f"Luck:        {importance['Luck']*100:.1f}%")
    
    print("\n--- 4. Archetype Comparison: Unlucky Genius vs Lucky Mediocre ---")
    # Define Archetypes
    # High Talent (>115, +1SD) / Low Luck (<0.3)
    unlucky_geniuses = df[(df['Talent'] > 115) & (df['Luck'] < 0.35)]
    
    # Low Talent (<100, Average/Below) / High Luck (>0.65)
    lucky_mediocre = df[(df['Talent'] < 100) & (df['Luck'] > 0.65)]
    
    print(f"Sample Sizes: Unlucky Geniuses={len(unlucky_geniuses)}, Lucky Mediocre={len(lucky_mediocre)}")
    
    ug_wealth = unlucky_geniuses['Wealth'].mean()
    lm_wealth = lucky_mediocre['Wealth'].mean()
    
    print(f"Average Wealth:")
    print(f"Unlucky Geniuses: ${ug_wealth:,.0f}")
    print(f"Lucky Mediocre:   ${lm_wealth:,.0f}")
    
    if ug_wealth > lm_wealth:
        diff = ((ug_wealth / lm_wealth) - 1) * 100
        print(f"RESULT: Talent Wins (by {diff:.1f}%)")
    else:
        diff = ((lm_wealth / ug_wealth) - 1) * 100
        print(f"RESULT: Luck Wins (by {diff:.1f}%)")

if __name__ == "__main__":
    analyze_success_drivers()
