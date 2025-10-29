import pandas as pd
import numpy as np
import os

def generate_synthetic_match_data(num_players=22, num_minutes=90):
    """
    Generates a synthetic dataset for a 90-minute match for 22 players.
    """
    print("Generating synthetic data...")
    
    # 1. Create the basic timeline for the match
    # We need one row for every player for every minute
    minutes = np.arange(1, num_minutes + 1)
    player_ids = np.arange(1, num_players + 1)
    
    # Use np.meshgrid to create all combinations of player and minute
    player_mesh, minute_mesh = np.meshgrid(player_ids, minutes)
    
    # Flatten them into columns
    data = pd.DataFrame({
        'minute': minute_mesh.flatten(),
        'player_id': player_mesh.flatten()
    })
    
    # Sort by minute, then player (this is a realistic data structure)
    data = data.sort_values(by=['minute', 'player_id']).reset_index(drop=True)
    
    num_rows = len(data)
    
    # 2. Add environmental and match-setup features
    # These are mostly constant for the whole match
    data['opponent_strength'] = np.random.uniform(0.5, 1.0, num_rows)
    data['rest_days'] = np.random.randint(2, 6, num_rows)
    data['WBGT'] = np.random.uniform(28, 35) # Simulating a hot day
    data['altitude'] = np.random.uniform(1500, 2500) # Simulating high altitude
    
    # 3. Add per-minute event features (the "actions")
    # We use random numbers here. A real simulation would be more complex,
    # but this is perfect for a prototype.
    
    # Sprints: 0-3 per minute
    data['sprints'] = np.random.poisson(0.5, num_rows) 
    
    # Pressures: 0-5 per minute
    data['pressures'] = np.random.poisson(1.0, num_rows)
    
    # xG_flow: "Expected Goals Flow"
    # A small positive or negative value representing goal-scoring contribution
    data['xG_flow'] = np.random.normal(0.001, 0.005, num_rows)
    
    # 4. Add a (fake) "ground truth" column for our models to predict
    # This is what our models will try to learn.
    
    # Fake "EVPM" (Expected Value Per Minute)
    # Let's say it's driven by sprints and xG_flow
    data['EVPM_ground_truth'] = data['xG_flow'] * 50 + data['sprints'] * 0.01 + np.random.normal(0, 0.01, num_rows)
    
    # Fake "Stamina_Index" (from 1 down to 0)
    # Let's make it decrease over time and faster with more sprints/pressures
    # We'll calculate this per player
    stamina_data = []
    for player in player_ids:
        player_df = data[data['player_id'] == player].copy()
        player_df['cumulative_load'] = (player_df['sprints'] + player_df['pressures']).cumsum()
        
        # Stamina decays with time (minute) and load, faster at high altitude/WBGT
        decay_rate = 0.001 + (player_df['altitude'] / 500000) + (player_df['WBGT'] / 10000)
        player_df['stamina_index'] = 1 - (player_df['minute'] * 0.005) - (player_df['cumulative_load'] * decay_rate)
        
        # Clip at 0 (can't have negative stamina)
        player_df['stamina_index'] = player_df['stamina_index'].clip(lower=0.1)
        stamina_data.append(player_df)
        
    data = pd.concat(stamina_data).sort_values(by=['minute', 'player_id'])

    # Fake "Injury_Event" (0 = no injury, 1 = injury)
    # Very rare event. Let's base it on low stamina and high recent load.
    data['injury_prob'] = (1 - data['stamina_index']) * 0.001 # Base prob
    data['injury_event'] = (np.random.rand(num_rows) < data['injury_prob']).astype(int)
    
    # Ensure at least 1-2 injuries for the model to learn from
    if data['injury_event'].sum() < 2:
        injury_indices = data.sample(2).index
        data.loc[injury_indices, 'injury_event'] = 1
        
    print(f"Generated {num_rows} rows of data.")
    print(data.head())
    
    # 5. Save the CSV to the correct directory
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'sim_match.csv')
    data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")
    
    return data

# --- Run this in your notebook ---
# (You might need to create the 'src' folder first)
# import sys
# sys.path.append('src')
# from data import generate_synthetic_match_data # (if you saved this in src/data.py)

# Or just run the function definition above in your Colab notebook
# and then call it:
synthetic_data = generate_synthetic_match_data()

