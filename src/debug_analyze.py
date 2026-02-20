import sys
import os
print("Current CWD:", os.getcwd())
print("Sys Path:", sys.path)

try:
    print("Importing config...")
    from config import RANDOM_SEED, POPULATION_SIZE
    print("Config imported. Seed:", RANDOM_SEED)
except Exception as e:
    print("Failed to import config:", e)
    import traceback
    traceback.print_exc()

try:
    print("Importing pandas...")
    import pandas as pd
    print("Pandas imported.")
except Exception as e:
    print("Failed to import pandas:", e)

try:
    print("Importing Simulation...")
    from simulation import Simulation
    print("Simulation imported.")
except Exception as e:
    print("Failed to import Simulation:", e)
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    print("Done.")
