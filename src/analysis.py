import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_results(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # Check for seaborn availability
    sns_available = False
    try:
        import seaborn as sns
        sns_available = True
    except ImportError:
        pass

    if sns_available:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use('ggplot')

    # Plot 1: Wealth and Gini Coefficient
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Average Wealth', color=color)
    ax1.plot(df['step'], df['avg_wealth'], color=color, linewidth=2, label='Avg Wealth')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Wealth Gini (Inequality)', color=color)
    ax2.plot(df['step'], df['wealth_gini'], color=color, linewidth=2, linestyle='--', label='Gini')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Wealth Accumulation and Inequality Over Time')
    fig.tight_layout()
    plt.savefig('wealth_gini_plot.png')
    print("Saved wealth_gini_plot.png")

    # Plot 2: Survival Rate and Energy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:green'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Survival Rate', color=color)
    ax1.plot(df['step'], df['survival_rate'], color=color, linewidth=2, label='Survival Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Average Energy', color=color)
    ax2.plot(df['step'], df['avg_energy'], color=color, linewidth=2, linestyle=':', label='Avg Energy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Survival and Energy Levels')
    fig.tight_layout()
    plt.savefig('survival_energy_plot.png')
    print("Saved survival_energy_plot.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <csv_file>")
    else:
        plot_results(sys.argv[1])
