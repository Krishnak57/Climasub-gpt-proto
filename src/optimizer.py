import pandas as pd
import numpy as np

# --- IMPORT OUR NEW DYNAMIC MODELS ---
# This imports from the 'src/models.py' file you just created
try:
    from src.models import (
        run_stamina_model, 
        run_injury_hazard_model, 
        run_evpm_model,
        PLAYERS_ON_PITCH,
        PLAYERS_ON_BENCH
    )
    print("[src/optimizer.py] Successfully imported DYNAMIC models from src/models.py")
except ImportError as e:
    print(f"[src/optimizer.py] FATAL ERROR: Could not import models: {e}")
    # If models.py is missing, we can't run
    exit()


# --- HELPER FUNCTIONS (REAL LOGIC) ---

def my_greedy_optimizer_logic(stamina_df, hazard_df, evpm_df, state):
    """
    --- THIS IS YOUR REAL STEP 10 ---
    It finds the best sub by checking all possibilities.
    """
    print("[src/optimizer.py] Running REAL greedy optimizer (Step 10)...")

    current_minute = state.get('minute', 65)
    alpha = state.get('alpha', 1.0) # Get the risk slider
    minutes_remaining = 90 - current_minute

    best_sub = None
    best_score = -np.inf

    # Get the model data for the REST of the match
    future_evpm = evpm_df.loc[current_minute:]
    future_hazard = hazard_df.loc[current_minute:]

    # Loop 1: Check every player on the pitch
    for out_player_id in PLAYERS_ON_PITCH.keys():

        # Get the current player's future value and risk
        current_player_value = future_evpm[out_player_id].sum()
        current_player_risk = future_hazard[out_player_id].sum()

        # Loop 2: Check every player on the bench
        for in_player_id in PLAYERS_ON_BENCH.keys():

            # Get the sub's future value and risk
            sub_player_value = future_evpm[in_player_id].sum()
            sub_player_risk = future_hazard[in_player_id].sum()

            # --- This is the core logic ---
            # Calculate the change in value and risk
            delta_value = sub_player_value - current_player_value
            delta_risk = sub_player_risk - current_player_risk

            # Calculate the final score using the 'alpha' risk slider
            # score = (change_in_value) - (risk_aversion * change_in_risk)
            score = delta_value - (alpha * delta_risk)

            if score > best_score:
                best_score = score
                best_sub = {
                    'out_player': out_player_id,
                    'in_player': in_player_id,
                    'at_minute': current_minute,
                    'delta_xG': delta_value, # Using EVPM sum as proxy for xG
                    'delta_risk_pct': (delta_risk / current_player_risk) * 100
                }

    if best_sub is None:
         return { 'out_player': 'No Sub', 'in_player': 'No Sub', 'at_minute': current_minute, 'delta_xG': 0, 'delta_risk_pct': 0 }

    return best_sub


def my_pressing_policy_logic(stamina_df, state):
    """
    --- THIS IS YOUR REAL STEP 11 ---
    It decides the pressing level based on team stamina.
    """
    print("[src/optimizer.py] Calculating REAL pressing policy (Step 11)...")

    current_minute = state.get('minute', 65)

    # Get the current stamina for players on the pitch
    player_ids = list(PLAYERS_ON_PITCH.keys())
    current_stamina = stamina_df.loc[current_minute][player_ids]

    # Calculate the team's average stamina
    avg_team_stamina = current_stamina.mean()

    if avg_team_stamina > 0.7:
        return { 'level': 'High', 'reason': f'Avg. stamina is {avg_team_stamina:.0%}'}
    elif avg_team_stamina > 0.5:
        return { 'level': 'Medium', 'reason': f'Avg. stamina is {avg_team_stamina:.0%}'}
    else:
        return { 'level': 'Low', 'reason': f'Avg. stamina is {avg_team_stamina:.0%}'}


# --- THIS IS THE MAIN FUNCTION STREAMLIT WILL CALL ---

def predict_recommendation(match_state):
    """
    This function now runs all the DYNAMIC models
    and passes their data to the REAL optimizer.
    """
    print(f"--- [Optimizer] Running predict_recommendation for minute {match_state.get('minute')} ---")

    # 1. Run your DYNAMIC models (Steps 6, 7, 8)
    stamina_df = run_stamina_model(match_state) 
    hazard_df = run_injury_hazard_model(match_state)
    evpm_df = run_evpm_model(match_state)

    # 2. Combine into one master timeline DataFrame (for charts)
    timelines_list = []
    all_player_ids = list(PLAYERS_ON_PITCH.keys()) + list(PLAYERS_ON_BENCH.keys())

    for player_id in all_player_ids:
        player_df = pd.DataFrame({
            'Minute': stamina_df.index,
            'player_id': player_id,
            'Stamina': stamina_df[player_id],
            'Injury_Hazard': hazard_df[player_id],
            'EVPM': evpm_df[player_id]
        })
        timelines_list.append(player_df)
    timelines_df = pd.concat(timelines_list).reset_index(drop=True)

    # 3. Run your REAL Optimizer (Step 10)
    best_sub = my_greedy_optimizer_logic(stamina_df, hazard_df, evpm_df, match_state)

    # 4. Determine REAL Pressing Policy (Step 11)
    press_policy = my_pressing_policy_logic(stamina_df, match_state)

    # 5. Final API Output
    recommendation = {
        'best_sub': best_sub,
        'press_policy': press_policy
    }

    print("--- [Optimizer] Run complete. Returning results. ---")

    # 6. Return results to Streamlit
    return recommendation, timelines_df