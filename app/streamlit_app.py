import streamlit as st
import pandas as pd
import numpy as np
import time

# --- IMPORTANT ---
# This block helps Python find your 'src' folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORT OUR REAL OPTIMIZER ---
# We are importing the *real* function from your optimizer.py file
from src.optimizer import predict_recommendation
# --- END OF IMPORTS ---


# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ClimaSub GPT",
    page_icon="⚽",
    layout="wide"
)

# --- 2. SIDEBAR (for user controls) ---
st.sidebar.title("Simulation Controls")
st.sidebar.markdown("Adjust scenario parameters and risk tolerance.")

alpha_risk_weight = st.sidebar.slider(
    "Risk Tolerance (Alpha)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,  # Default value
    step=0.25,
    help="How much to penalize injury risk. Higher = more cautious."
)

st.sidebar.header("Environmental Factors")
wbgt = st.sidebar.slider("WBGT (Wet-Bulb Globe Temp, °C)", 20, 40, 34)
altitude = st.sidebar.slider("Altitude (meters)", 0, 3000, 2400)

# --- 3. MAIN PAGE TITLE ---
st.title("ClimaSub-GPT ⚽: Live Substitution Optimizer")
st.markdown(f"Demonstrating recommendations for a high-altitude (**{altitude}m**) and high-heat (**{wbgt}°C**) match.")

# --- 4. LIVE MATCH CONTROLS ---
st.divider()
current_minute = st.slider("Match Minute", 0, 90, 65) # min, max, default

# --- NEW: Initialize session state (our app's 'memory') ---
# This holds our results so the app doesn't forget them
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = None
if 'timelines_df' not in st.session_state:
    st.session_state.timelines_df = None

# --- 5. RUN OPTIMIZER & DISPLAY RECOMMENDATION ---
st.divider()

col1, col2 = st.columns([1, 2])

# --- UPDATED: Column 1 (The Button) ---
with col1:
    st.subheader("Optimal Decision Engine")
    
    # This button will run the optimizer
    if st.button(f"Find Best Action at Minute {current_minute}", type="primary"):
        
        # 1. Create the 'match_state' dictionary
        match_state = {
            'minute': current_minute,
            'wbgt': wbgt,
            'altitude': altitude,
            'alpha': alpha_risk_weight,
        }
        
        # 2. Show a loading spinner
        with st.spinner("Running models and optimizer..."):
            
            # --- THIS IS THE KEY ---
            # We call your REAL optimizer function
            # and save BOTH results into our 'memory' (session_state)
            st.session_state.recommendation, st.session_state.timelines_df = predict_recommendation(match_state)
            # --- END OF KEY PART ---

    # --- UPDATED: Display results AFTER button is clicked ---
    # We check if data exists in our 'memory'
    if st.session_state.recommendation:
        # Get the data from memory
        rec = st.session_state.recommendation['best_sub']
        
        # Display the recommendation
        st.success(f"**Sub: {rec['in_player']} (IN) for {rec['out_player']} (OUT)**")
        
        # Display the metric cards
        col_xg, col_risk = st.columns(2)
        col_xg.metric(
            label="Projected xG Lift (90min)",
            value=f"+{rec['delta_xG']:.2f}",
            help="The extra 'Expected Goals' we get from this sub."
        )
        col_risk.metric(
            label="Injury Risk Change (next 15min)",
            value=f"{rec['delta_risk_pct']:.1f}%",
            delta_color="inverse", # Green for a negative (good) change
            help="The % change in injury probability for the player coming off."
        )
        
        # Display the pressing policy
        st.subheader("Pressing Policy")
        st.info(f"**Set Press: {st.session_state.recommendation['press_policy']['level']}**")
    
    else:
        # Show this message *before* the button is clicked
        st.info("Click the 'Find Best Action' button to run the optimizer.")


# --- UPDATED: Column 2 (The Charts) ---
with col2:
    st.subheader("Player Analytics")
    
    # Check if our 'memory' (session_state) has the chart data
    if st.session_state.timelines_df is not None:
        
        # Get a list of all unique players from the dataframe
        player_list = st.session_state.timelines_df['player_id'].unique()
        
        # --- NEW WIDGET ---
        # Create a dropdown menu to select a player
        selected_player = st.selectbox("Select Player to Analyze", player_list)
        
        if selected_player:
            # Filter the big DataFrame for ONLY the chosen player
            chart_data = st.session_state.timelines_df[
                st.session_state.timelines_df['player_id'] == selected_player
            ]
            
            # Set 'Minute' as the index for charting
            chart_data_indexed = chart_data.set_index('Minute')
            
            # Plot the REAL data from your models
            st.markdown(f"**Stamina Forecast ({selected_player})**")
            st.line_chart(chart_data_indexed[['Stamina']])
            
            st.markdown(f"**Injury Hazard Forecast ({selected_player})**")
            st.line_chart(chart_data_indexed[['Injury_Hazard']])
            
            st.caption("Displaying REAL data from your models.")
    
    else:
        # Show this message *before* the button is clicked
        st.info("Run the optimizer to see player charts.")