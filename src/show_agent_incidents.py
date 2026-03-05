import sys
import os

# Add src to path if needed (assuming running from root)
sys.path.append(os.path.join(os.getcwd(), 'src'))

from simulation import Simulation
import random

def run_life_story():
    print("=" * 60)
    print("  THE CHRONICLES OF AN AGENT: A LIFE IN INCIDENTS  ")
    print("=" * 60)
    
    # Run a simulation with a small population to ensure some interaction/competition
    # but we will focus on one "Protagonist"
    sim = Simulation(pop_size=50, seed=random.randint(0, 10000), enable_interactions=True)
    
    # Let's pick our Protagonist (someone with average talent to see the struggle)
    protagonist = None
    for a in sim.agents:
        if 90 < a.talent < 110:
            protagonist = a
            break
    if not protagonist: protagonist = sim.agents[0]
    
    print(f"\nPROTAGONIST PROFILE:")
    print(f"ID: {protagonist.id[:8]}")
    print(f"Talent: {protagonist.talent:.1f} (IQ Equivalent)")
    print(f"Luck Factor: {protagonist.luck:.2f} (0 to 1)")
    print(f"Starting Wealth: ${protagonist.wealth:.2f}")
    print("-" * 60)
    
    # Run simulation
    sim.run()
    
    # Filter and display major incidents
    print("\n LIFE TIMELINE:")
    print("-" * 60)
    
    last_phase = ""
    for entry in protagonist.history:
        # Infer phase from age (based on config-like durations if not explicit)
        # But we can just print the age and description
        age = entry['age']
        desc = entry.get('description', 'Standard day.')
        delta = entry.get('wealth_delta', 0)
        
        # Determine Phase for better formatting (Approximate based on PHASE_DURATIONS)
        current_phase = ""
        if age < 20: current_phase = "EDUCATION"
        elif age < 50: current_phase = "EARLY/MID CAREER"
        elif age < 70: current_phase = "PEAK OPPORTUNITY"
        else: current_phase = "LATE CAREER / DECLINE"
        
        if current_phase != last_phase:
            print(f"\n--- PHASE: {current_phase} ---")
            last_phase = current_phase
            
        sign = "+" if delta >= 0 else "-"
        delta_str = f"({sign}${abs(delta):.0f})" if delta != 0 else ""
        
        print(f"[Age {age:02d}] {desc} {delta_str}")

    print("-" * 60)
    print(f"FINAL WEALTH: ${protagonist.wealth:.2f}")
    print(f"STATUS: {'RETIRED / ALIVE' if protagonist.alive else 'DECEASED'}")
    print("=" * 60)

if __name__ == "__main__":
    run_life_story()
