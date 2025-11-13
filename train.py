import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Import the data module from the 'src' folder
from src import data 

print("--- Starting Model Training ---")

# 1. Load data from src/data.py
try:
    # --- THIS IS THE FIX ---
    # I am assuming your function is named 'generate_synthetic_data'
    # If your function has a different name, change it here!
    df = data.generate_synthetic_data() 
    print("Synthetic data loaded successfully.")
except Exception as e:
    print(f"Error loading data from src/data.py: {e}")
    print("Please ensure src/data.py has a function named 'generate_synthetic_data()'")
    exit()

# 2. Define features (X) and target (y)
# --- NOTE ---
# Your synthetic data function must create these columns!
FEATURES = ['Position', 'Age', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality']
TARGET = 'Overall'

X = df[FEATURES]
y = df[TARGET]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# 3. Define preprocessing
categorical_features = ['Position']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 4. Create the full model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Train the model
print("Training the model...")
model.fit(X, y)
print("Model training complete.")

# 6. Save the entire pipeline to a file
pipeline_filename = 'player_pipeline.pkl'
joblib.dump(model, pipeline_filename)

print(f"Model pipeline saved successfully as {pipeline_filename}")
print("--- Training Finished ---")