import pandas as pd
import numpy as np

def generate_synthetic_data(num_players=500):
    """
    Generates a DataFrame of synthetic player data for training.
    This is the function that train.py is looking for.
    """
    print("--- [src/data.py] Generating synthetic player data... ---")

    positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']

    data = {
        'Position': np.random.choice(positions, num_players),
        'Age': np.random.randint(18, 35, size=num_players),
        'Pace': np.random.randint(60, 99, size=num_players),
        'Shooting': np.random.randint(50, 95, size=num_players),
        'Passing': np.random.randint(55, 95, size=num_players),
        'Dribbling': np.random.randint(60, 99, size=num_players),
        'Defending': np.random.randint(40, 90, size=num_players),
        'Physicality': np.random.randint(50, 95, size=num_players)
    }

    # Create the 'Overall' target variable (what we want to predict)
    df = pd.DataFrame(data)

    # Create a simple formula for 'Overall'
    df['Overall'] = (
        df['Pace'] * 0.15 +
        df['Shooting'] * 0.15 +
        df['Passing'] * 0.2 +
        df['Dribbling'] * 0.2 +
        df['Defending'] * 0.15 +
        df['Physicality'] * 0.15 +
        np.random.normal(0, 2, size=num_players) # add some noise
    ).astype(int)

    print("--- [src/data.py] Data generation complete. ---")
    return df

# This part at the bottom lets you test this file directly
if __name__ == "__main__":
    player_data = generate_synthetic_data()
    print("Test run successful. Data head:")
    print(player_data.head())