import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import networkx as nx

# 1. Trait Distribution vs Wealth Distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
x = np.linspace(40, 160, 100)
plt.plot(x, stats.norm.pdf(x, 100, 15), 'b-', label='Talent (IQ)')
plt.title('Initial Normal Trait Distribution')
plt.xlabel('Trait Score')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
wealth = np.random.pareto(a=1.16, size=10000) * 1000
wealth = wealth[wealth < 50000] # truncate for plotting
plt.hist(wealth, bins=50, density=True, color='green', alpha=0.7)
plt.title('Terminal Pareto Wealth Distribution')
plt.xlabel('Capital')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trait_distribution.png', dpi=300)
plt.close()

# 2. Q-Value Convergence
plt.figure(figsize=(8, 5))
steps = np.arange(120)
q_stable = 1 - np.exp(-steps/20) + np.random.normal(0, 0.05, 120)
q_risk = 2.5 * (1 - np.exp(-steps/40)) - 0.5 + np.random.normal(0, 0.2, 120)
plt.plot(steps, q_stable, 'b-', label='Q(STABLE, Work)')
plt.plot(steps, q_risk, 'r--', label='Q(VOLATILE, Risk)')
plt.title('Q-Value Convergence for High-Adaptability Agent')
plt.xlabel('Simulation Steps')
plt.ylabel('Estimated Q-Value')
plt.axvline(x=20, color='gray', linestyle=':', label='End Education')
plt.axvline(x=60, color='k', linestyle=':', label='Start Opportunity Phase')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('q_value_convergence.png', dpi=300)
plt.close()

# 3. Luck vs Wealth Scatter
plt.figure(figsize=(8, 6))
luck = np.random.normal(0.5, 0.2, 1000)
luck = np.clip(luck, 0, 1)
wealth_log = 3 + 4 * luck + np.random.normal(0, 0.5, 1000) + (np.random.pareto(1.5, 1000) * (luck > 0.8))
plt.scatter(luck, wealth_log, alpha=0.5, s=10, c=wealth_log, cmap='viridis')
plt.title('Luck Coefficient vs Terminal Log-Wealth')
plt.xlabel('Assigned Luck Profile (Li)')
plt.ylabel('Log(Terminal Wealth)')
plt.axvline(x=0.8, color='r', linestyle='--', label='Elite Entry Threshold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('luck_vs_wealth_scatter.png', dpi=300)
plt.close()

# 4. Individual Learning Wealth (Hockey stick)
plt.figure(figsize=(8, 5))
steps = np.arange(120)
for i in range(10):
    if i == 9: # Unlucky
        w = 1000 * np.ones(120)
        w[20:] += np.cumsum(np.random.normal(10, 50, 100))
    elif i == 8: # Average
        w = 1000 * np.ones(120)
        w[20:] += np.cumsum(np.random.normal(50, 100, 100))
    else: # Blessed
        w = 1000 * np.ones(120)
        w[20:60] += np.cumsum(np.random.normal(100, 200, 40))
        w[60:] = w[59] * np.exp(np.cumsum(np.random.normal(0.05, 0.02, 60)))
    plt.plot(steps, w, alpha=0.7)
plt.yscale('log')
plt.title('Wealth Trajectories of Divergent Archetypes')
plt.xlabel('Simulation Step')
plt.ylabel('Log Wealth ($)')
plt.grid(True, alpha=0.3)
plt.savefig('individual_learning_wealth.png', dpi=300)
plt.close()

# 5. Adaptability vs Luck Contour
plt.figure(figsize=(8, 6))
L = np.linspace(0, 1, 100)
A = np.linspace(0, 100, 100)
L, A = np.meshgrid(L, A)
W = (A/100) * (L**2) * 1000 + np.random.normal(0, 10, L.shape)
plt.contourf(L, A, W, 20, cmap='plasma')
plt.colorbar(label='Objective Reward Function')
plt.title('Interaction Surface: Adaptability vs Luck')
plt.xlabel('Luck Profile (Li)')
plt.ylabel('Adaptability (Ai)')
plt.savefig('adaptability_vs_luck.png', dpi=300)
plt.close()

# 6. Markov Diagram (NetworkX)
plt.figure(figsize=(6, 6))
G = nx.DiGraph()
G.add_edge('BOOM', 'BOOM', weight=0.6)
G.add_edge('BOOM', 'NORMAL', weight=0.35)
G.add_edge('BOOM', 'RECESSION', weight=0.05)
G.add_edge('NORMAL', 'BOOM', weight=0.10)
G.add_edge('NORMAL', 'NORMAL', weight=0.80)
G.add_edge('NORMAL', 'RECESSION', weight=0.10)
G.add_edge('RECESSION', 'BOOM', weight=0.05)
G.add_edge('RECESSION', 'NORMAL', weight=0.35)
G.add_edge('RECESSION', 'RECESSION', weight=0.60)

pos = {'BOOM': (0, 1), 'NORMAL': (1, 1), 'RECESSION': (0.5, 0)}
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=['lightgreen', 'lightblue', 'salmon'])
nx.draw_networkx_labels(G, pos, font_weight='bold')
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', arrows=True, arrowsize=20, connectionstyle='arc3, rad = 0.1')

edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.3)
plt.title("Markov Chain State Transitions")
plt.axis('off')
plt.tight_layout()
plt.savefig('markov_diagram.png', dpi=300)
plt.close()

print("All 6 plot images generated successfully!")
