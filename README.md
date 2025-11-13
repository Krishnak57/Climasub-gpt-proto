# ClimaSub-GPT: A Live Substitution Optimizer

This project was built for the [Name of your Hackathon] by Krishna Khanal and Bikesh Shrestha.

## 🚀 Live App Demo

You can use the live, deployed version of this app here:
**[https://climasub-gpt-proto.streamlit.app](https://climasub-gpt-proto.streamlit.app)**

## 💡 The Problem
In sports, coaches often rely on "gut-feel" for substitutions. This is unreliable, especially when "invisible threats" like extreme heat (WBGT) and altitude accelerate player fatigue and injury risk.

## Solution
We built a data-driven tool that gives coaches a clear, simple recommendation for the optimal substitution.

Our app uses a two-part system:
1.  **The "Brain":** A `RandomForestRegressor` model (Scikit-learn) trained on a synthetic player dataset to predict performance.
2.  **The "Body":** A dynamic simulation engine (built in Streamlit) that models per-minute stamina and injury risk based on live heat and altitude inputs.

This turns complex data into a single, actionable decision to protect players and win more matches.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data Science:** Pandas, Scikit-learn
* **Backend:** Python