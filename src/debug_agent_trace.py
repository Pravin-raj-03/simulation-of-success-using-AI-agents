from simulation import Simulation
import pandas as pd

def trace_agent_life():
    with open("agent_life.txt", "w", encoding="utf-8") as f:
        f.write("--- GENERATING AGENT LIFE STORY ---\n")
        # Run a small simulation
        sim = Simulation(pop_size=1, seed=123, enable_interactions=True)
        
        # Let's inspect our Hero
        hero = sim.agents[0]
        f.write(f"Meet our Agent: {hero.id[:4]}\n")
        f.write(f"Stats: IQ={hero.iq:.0f}, Luck={hero.luck:.2f}, Strength={hero.strength:.0f}\n")
        f.write(f"Starting Wealth: ${hero.wealth}\n")
        f.write("-" * 30 + "\n")
        
        # Run for a few steps (cover Education and Early Career)
        f.write("\n[PHASE 1: EDUCATION]\n")
        for _ in range(5):
            sim.run_step_for_rl() # Just step forward
            latest = hero.history[-1] if hero.history else None
            f.write(f"Age {hero.age}: {latest}\n")
            
        f.write(f"\nStatus check: Wealth=${hero.wealth:.0f}, Energy={hero.energy:.0f}\n")

        # FORCE skip to Career Phase (modify sim internal state)
        sim.current_step = 20 # Skip to Career
        
        f.write("\n[PHASE 2: CAREER START]\n")
        for _ in range(5):
            sim.run_step_for_rl()
            latest = hero.history[-2:] # Get last few events
            events = [f"{e['event']} ({e.get('wealth_delta', e.get('value', 0)):.1f})" for e in latest]
            f.write(f"Age {hero.age}: {', '.join(events)}\n")

        f.write("-" * 30 + "\n")
        f.write(f"FINAL STATUS: Wealth=${hero.wealth:.0f}, Alive={hero.alive}\n")
        print("Done writing to agent_life.txt")

if __name__ == "__main__":
    trace_agent_life()
