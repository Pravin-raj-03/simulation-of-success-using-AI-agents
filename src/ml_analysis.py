import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from simulation import Simulation
from config import RANDOM_SEED, POPULATION_SIZE

def run_ml_analysis(seed=RANDOM_SEED, pop_size=POPULATION_SIZE):
    print("Generating training data from simulation...")
    # Enable interactions for full complexity
    sim = Simulation(pop_size, seed, enable_interactions=True)
    sim.run()
    
    # Extract Features and Target
    data = []
    for a in sim.agents:
        # We want to predict Final Wealth based on initial/static attributes
        # Include everyone. Dead agents have wealth 0? Or negative?
        # a.wealth can be negative in our sim.
        
        feature_row = {
            'IQ': a.iq,
            'Strength': a.strength,
            'Luck': a.luck,
            'Adaptability': a.adaptability,
            'Sector_Risk': 1 if a.sector == 'VOLATILE' else 0,
            'Initial_Network_Potential': a.reputation, 
        }
        # If dead, use actual final wealth (which might be low/negative)
        # But we might want to flag 'Alive' as a feature? No, we predict Outcome.
        target = a.wealth
        
        data.append({**feature_row, 'Wealth': target})
        
    df = pd.DataFrame(data)
    print(f"Dataframe shape: {df.shape}")
    print(df.head())
    
    X = df.drop('Wealth', axis=1)
    y = df['Wealth']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest on {len(X_train)} agents...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    score = rf.score(X_test, y_test)
    print(f"Model R^2 Score (Predictability): {score:.4f}")
    
    # Feature Importance (Standard RF)
    importances = rf.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    print("\n--- Feature Importance (Gini Importance) ---")
    for f in range(X.shape[1]):
        print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # SHAP Values (Explainable AI)
    print("\nCalculating SHAP values (may take a moment)...")
    shap_available = False
    try:
        import shap
        shap_available = True
        
        explainer = shap.Explainer(rf, X_train)
        shap_values = explainer(X_test)
        
        # Plot Summary
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("SHAP Feature Importance (Impact on Wealth)")
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        print("Saved shap_summary.png")
        
        # Plot Dependence (Luck vs Wealth interaction)
        # shap.dependence_plot("Luck", shap_values.values, X_test, show=False)
        # plt.savefig('shap_luck_dependence.png')
        
    except ImportError:
        print("SHAP library not found. Skipping detailed explainability plots.")
        # Fallback to standard bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig('rf_importance.png')
        print("Saved rf_importance.png")

if __name__ == "__main__":
    run_ml_analysis()
