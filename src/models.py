import pandas as pd
import numpy as np
import joblib # We will use this to load your trained model

# --- LOAD YOUR TRAINED MODEL (from train.py) ---
# This loads the 'player_pipeline.pkl' file you created.
# We will use this model to predict player 'Overall' scores.
try:
    PLAYER_MODEL = joblib.load('player_pipeline.pkl')
    print("[src/models.py] Player 'Overall' model loaded successfully.")
except FileNotFoundError:
    print("[src/models.py] WARNING: 'player_pipeline.pkl' not found.")
    print("[src/models.py] Using FAKE player model.")
    PLAYER_MODEL = None
except Exception as e:
    print(f"[src/models.py] Error loading model: {e}")
    PLAYER_MODEL = None

# --- SIMULATE PLAYER DATA ---
# This creates a "roster" for our game.
PLAYERS_ON_PITCH = {
    'P1_Adams': {'Position': 'Midfielder', 'Age': 24, 'Pace': 88, 'Shooting': 65, 'Passing': 75, 'Dribbling': 78, 'Defending': 82, 'Physicality': 85},
    'P2_Pulisic': {'Position': 'Forward', 'Age': 25, 'Pace': 92, 'Shooting': 82, 'Passing': 80, 'Dribbling': 88, 'Defending': 45, 'Physicality': 70},
    'P3_Reyna': {'Position': 'Midfielder', 'Age': 21, 'Pace': 85, 'Shooting': 80, 'Passing': 84, 'Dribbling': 86, 'Defending': 40, 'Physicality': 68},
    'P4_Dest': {'Position': 'Defender', 'Age': 23, 'Pace': 91, 'Shooting': 60, 'Passing': 75, 'Dribbling': 82, 'Defending': 78, 'Physicality': 72},
    'P5_Richards': {'Position': 'Defender', 'Age': 24, 'Pace': 82, 'Shooting': 40, 'Passing': 60, 'Dribbling': 65, 'Defending': 85, 'Physicality': 88},
}

PLAYERS_ON_BENCH = {
    'P16_Aaronson': {'Position': 'Midfielder', 'Age': 23, 'Pace': 89, 'Shooting': 74, 'Passing': 78, 'Dribbling': 84, 'Defending': 50, 'Physicality': 70},
    'P17_Weah': {'Position': 'Forward', 'Age': 24, 'Pace': 94, 'Shooting': 78, 'Passing': 72, 'Dribbling': 83, 'Defending': 40, 'Physicality': 75},
    'P18_Robinson': {'Position': 'Defender', 'Age': 26, 'Pace': 93, 'Shooting': 50, 'Passing': 68, 'Dribbling': 75, 'Defending': 80, 'Physicality': 84},
}

ALL_PLAYERS = {**PLAYERS_ON_PITCH, **PLAYERS_ON_BENCH}


def run_stamina_model(match_state):
    """
    (Step 6) Calculates stamina for all players.
    This is now DYNAMIC based on WBGT and Altitude.
    """
    print("[src/models.py] Running DYNAMIC stamina model...")
    minutes = np.arange(0, 91)
    stamina_df = pd.DataFrame(index=pd.RangeIndex(start=0, stop=91, name="Minute"))

    # Get environmental factors from the slider
    wbgt = match_state.get('wbgt', 25)
    altitude = match_state.get('altitude', 0)

    # Create a 'fatigue_factor' from the sliders
    # Higher temp/altitude = faster fatigue
    temp_factor = 1 + (max(0, wbgt - 25) / 10) * 0.5  # 50% faster fatigue at 35C
    alt_factor = 1 + (altitude / 1000) * 0.1          # 10% faster fatigue at 1000m

    for player_id, stats in ALL_PLAYERS.items():
        # Players with better 'Physicality' get tired slower
        phys_factor = 1.0 - (stats['Physicality'] - 75) * 0.01
        base_fatigue = np.random.uniform(0.005, 0.008)

        # Combine all factors
        total_fatigue_rate = base_fatigue * phys_factor * temp_factor * alt_factor

        # Calculate stamina drop
        stamina = 1.0 - (minutes * total_fatigue_rate)
        stamina_df[player_id] = np.clip(stamina, 0.01, 1.0)

    return stamina_df


def run_injury_hazard_model(match_state):
    """
    (Step 7) Calculates injury hazard for all players.
    This is DYNAMIC based on stamina.
    """
    print("[src/models.py] Running DYNAMIC injury hazard model...")

    # Hazard depends on stamina, so we run that first
    stamina_df = run_stamina_model(match_state)

    # Hazard = (base_risk) / (stamina^2)
    # As stamina drops, hazard goes up exponentially
    hazard_df = 0.0001 / (stamina_df ** 2)

    return hazard_df


def run_evpm_model(match_state):
    """
    (Step 8) Calculates Expected Value Per Minute (EVPM) for all players.
    This uses your trained 'player_pipeline.pkl' model!
    """
    print("[src/models.py] Running DYNAMIC EVPM model...")
    evpm_df = pd.DataFrame(index=pd.RangeIndex(start=0, stop=91, name="Minute"))

    # Get stamina
    stamina_df = run_stamina_model(match_state)

    for player_id, stats in ALL_PLAYERS.items():
        if PLAYER_MODEL:
            # 1. Use your trained model to get a base 'Overall' score
            player_stats_df = pd.DataFrame([stats])
            base_overall = PLAYER_MODEL.predict(player_stats_df)[0]
        else:
            # 2. Fallback if model failed to load
            base_overall = np.mean(list(stats.values())[1:])

        # 3. Player value (EVPM) decreases with stamina
        # EVPM = (Base Value) * (Current Stamina)
        evpm = (base_overall / 1000.0) * stamina_df[player_id] # Scale it
        evpm_df[player_id] = evpm

    return evpm_df
