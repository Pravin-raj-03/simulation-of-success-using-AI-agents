try:
    from config import RANDOM_SEED
    print("Config import successful:", RANDOM_SEED)
except Exception as e:
    print("Config import failed:", e)

try:
    from agent import Agent
    print("Agent import successful")
except Exception as e:
    print("Agent import failed:", e)
