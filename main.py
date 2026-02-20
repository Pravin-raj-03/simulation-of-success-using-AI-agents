import argparse
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import RANDOM_SEED, POPULATION_SIZE
from simulation import Simulation

def main():
    parser = argparse.ArgumentParser(description="Simulation of Life by Agents")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--pop_size', type=int, default=POPULATION_SIZE, help='Population size')
    
    args = parser.parse_args()
    
    print(f"Starting Simulation with Seed: {args.seed}, Population: {args.pop_size}")
    
    sim = Simulation(args.pop_size, args.seed)
    df = sim.run()
    
    print("Simulation complete.")
    print(df.tail())
    
    # Save results
    output_file = f"simulation_results_seed_{args.seed}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
