import pandas as pd
import numpy as np
import sys
from simulation import Simulation
from config import RANDOM_SEED, POPULATION_SIZE

def validate_agent_data(seed=RANDOM_SEED, pop_size=POPULATION_SIZE):
    """
    Runs simulation and validates the generated agent data.
    Returns a dictionary of validation results.
    """
    print(f"--- Validating Agent Data (Seed={seed}, Pop={pop_size}) ---")
    
    # 1. Generate Data
    sim = Simulation(pop_size, seed, enable_interactions=True)
    sim.run()
    
    agents = sim.agents
    data = []
    for a in agents:
        data.append({
            'Talent': a.talent,
            'Strength': a.strength,
            'Luck': a.luck,
            'Adaptability': a.adaptability,
            'Wealth': a.wealth,
            'Alive': a.alive
        })
    
    df = pd.DataFrame(data)
    report = {}
    
    # 2. Integrity Checks
    # Check for NaNs
    nan_counts = df.isnull().sum().sum()
    report['Total NaNs'] = nan_counts
    if nan_counts > 0:
        print(f"[FAIL] Found {nan_counts} missing values in agent data.")
    else:
        print("[PASS] No missing values found.")
        
    # Check Ranges
    # Luck should be 0.0 to 1.0
    luck_in_range = df['Luck'].between(0.0, 1.0).all()
    report['Luck Valid Range'] = luck_in_range
    if not luck_in_range:
        print(f"[FAIL] Luck values out of range [0, 1]. Min: {df['Luck'].min()}, Max: {df['Luck'].max()}")
    else:
        print("[PASS] Luck values within [0.0, 1.0].")

    # Talent should be roughly 0 to 200 (allow some outlier wiggle room but <0 is bad)
    talent_valid = (df['Talent'] >= 0).all()
    report['Talent Non-Negative'] = talent_valid
    if not talent_valid:
         print(f"[FAIL] Negative Talent found. Min: {df['Talent'].min()}")
    else:
        print("[PASS] Talent values valid (non-negative).")

    # 3. Statistical Distributions
    # Wealth should be skewed (Mean > Median often indicates right skew)
    mean_wealth = df['Wealth'].mean()
    median_wealth = df['Wealth'].median()
    report['Mean Wealth'] = mean_wealth
    report['Median Wealth'] = median_wealth
    
    skewness = "Right Skewed (Realistic)" if mean_wealth > median_wealth else "Left Skewed/Normal"
    print(f"[INFO] Wealth Distribution: Mean={mean_wealth:.2f}, Median={median_wealth:.2f} -> {skewness}")
    
    # 4. Correlations
    # We expect some positive correlation between Talent and Wealth, and Luck and Wealth
    corr_talent = df['Talent'].corr(df['Wealth'])
    corr_luck = df['Luck'].corr(df['Wealth'])
    
    report['Corr Talent-Wealth'] = corr_talent
    report['Corr Luck-Wealth'] = corr_luck
    
    print(f"[INFO] Talent-Wealth Correlation: {corr_talent:.4f}")
    print(f"[INFO] Luck-Wealth Correlation: {corr_luck:.4f}")
    
    return report

def validate_aggregate_csv(file_path):
    """
    Validates the structure and content of a simulation results CSV.
    """
    print(f"\n--- Validating Aggregate Log: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return False

    # Required columns
    required_cols = ['step', 'survival_rate', 'avg_wealth', 'wealth_gini']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        print(f"[FAIL] Missing columns: {missing_cols}")
        return False
        
    # Validation Logic
    # Survival rate must be 0.0 to 1.0
    survival_valid = df['survival_rate'].between(0.0, 1.0).all()
    if survival_valid:
        print("[PASS] Survival rates represent valid percentages (0-1).")
    else:
        print("[FAIL] Survival rates out of range.")
        
    # Gini coefficient must be 0.0 to 1.0
    gini_valid = df['wealth_gini'].between(0.0, 1.0).all()
    if gini_valid:
        print("[PASS] Gini coefficients valid (0-1).")
    else:
        print("[FAIL] Gini coefficients out of range.")
        
    return True

if __name__ == "__main__":
    print("Beginning Validation Suite...\n")
    
    # 1. Validate Agent Generation
    validate_agent_data()
    
    # 2. Validate CSV if exists
    # Construct filename based on config seed
    csv_path = f"simulation_results_seed_{RANDOM_SEED}.csv"
    validate_aggregate_csv(csv_path)
    
    print("\nValidation Complete.")
