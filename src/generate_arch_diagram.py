import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define boxes (x, y, width, height)
    boxes = {
        "Agent": (1, 4, 3, 3),
        "Environment": (8, 4, 3, 3),
        "Engine": (4.5, 1, 3, 2)
    }
    
    colors = {
        "Agent": "#a8dadc", # Light Blue
        "Environment": "#f1faee", # White-ish
        "Engine": "#e63946" # Red-ish
    }
    
    # Draw Boxes
    for name, (x, y, w, h) in boxes.items():
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor='black', facecolor=colors.get(name, 'white'))
        ax.add_patch(rect)
        plt.text(x + w/2, y + h - 0.5, name.upper(), ha='center', fontsize=12, weight='bold')

    # Agent Internals
    plt.text(2.5, 6, "State (S)\n[IQ, Energy, Wealth]", ha='center', fontsize=9)
    plt.text(2.5, 5, "Policy (π)\n[Q-Learning]", ha='center', fontsize=9)
    plt.text(2.5, 4.25, "Action (a)", ha='center', fontsize=9, style='italic')

    # Environment Internals
    plt.text(9.5, 6, "Phases\n[Edu, Career, Opp]", ha='center', fontsize=9)
    plt.text(9.5, 5, "Market State\n[Boom/Recession]", ha='center', fontsize=9)
    plt.text(9.5, 4.25, "Reward (r)", ha='center', fontsize=9, style='italic')

    # Engine Internals
    plt.text(6, 2.2, "Simulation Loop", ha='center', fontsize=10, weight='bold')
    plt.text(6, 1.5, "Clock & Sync\nInteraction Logic", ha='center', fontsize=9)

    # Arrows
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="k")

    # Agent -> Engine (Action)
    a1 = patches.FancyArrowPatch((2.5, 4), (5, 3), connectionstyle="arc3,rad=0.2", **kw)
    ax.add_patch(a1)
    plt.text(3.2, 3.2, "Action", fontsize=10)

    # Engine -> Environment (Apply)
    a2 = patches.FancyArrowPatch((7, 3), (9.5, 4), connectionstyle="arc3,rad=0.2", **kw)
    ax.add_patch(a2)
    plt.text(8.5, 3.2, "Update", fontsize=10)

    # Environment -> Agent (Observation/Reward)
    a3 = patches.FancyArrowPatch((9.5, 7), (2.5, 7), connectionstyle="arc3,rad=-0.1", **kw)
    ax.add_patch(a3)
    plt.text(6, 7.2, "State & Reward", fontsize=10, ha='center')

    plt.title("System Architecture: Agent-Environment Interaction Loop", fontsize=14)
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300)
    print("Saved architecture_diagram.png")

if __name__ == "__main__":
    draw_architecture_diagram()
