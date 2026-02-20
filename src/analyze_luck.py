from config import RANDOM_SEED, POPULATION_SIZE
from simulation import Simulation

def analyze_luck(seed=RANDOM_SEED, pop_size=POPULATION_SIZE):
    # Re-run simulation to get agent-level data (since CSV masked individual agents)
    print("Re-running simulation to extract agent-level data...")
    sim = Simulation(pop_size, seed)
    sim.run()
    
    # Extract Agent Data
    agents = sim.agents
    data = []
    for a in agents:
        data.append({
            'Luck': a.luck,
            'IQ': a.talent,
            'Final_Wealth': a.wealth,
            'Alive': a.alive
        })
    
    df = pd.DataFrame(data)
    
    # Check seaborn
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        plt.style.use('ggplot')

    # Plot Luck vs Wealth
    plt.figure(figsize=(10, 6))
    
    # Color by Alive status
    plt.scatter(df['Luck'], df['Final_Wealth'], alpha=0.5, c=df['IQ'], cmap='viridis')
    plt.colorbar(label='IQ')
    plt.xlabel('Luck Attribute (0.0 - 1.0)')
    plt.ylabel('Final Wealth')
    plt.title('Impact of Luck on Final Wealth (Color=IQ)')
    plt.yscale('symlog') # Log scale because wealth differences are huge
    
    plt.tight_layout()
    plt.savefig('luck_wealth_analysis.png')
    print("Saved luck_wealth_analysis.png")
    
    # specific stats
    correlation = df['Luck'].corr(df['Final_Wealth'])
    print(f"Correlation between Luck and Final Wealth: {correlation:.4f}")
    
    # Compare Top 10% vs Bottom 10% Luck
    top_luck = df[df['Luck'] > 0.8]['Final_Wealth'].mean()
    bottom_luck = df[df['Luck'] < 0.2]['Final_Wealth'].mean()
    print(f"Avg Wealth (High Luck > 0.8): {top_luck:.2f}")
    print(f"Avg Wealth (Low Luck < 0.2): {bottom_luck:.2f}")

if __name__ == "__main__":
    analyze_luck()
