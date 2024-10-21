import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Define forecasting function
def forecast_with_xgboost(df, timesteps, n_forecast_days, model):
    X = []
    y = []

    # Create input-output sequence
    for i in range(len(df) - timesteps - n_forecast_days + 1):
        X.append(df['y'].iloc[i:i + timesteps].values)
        y.append(df['y'].iloc[i + timesteps:i + timesteps + n_forecast_days].values)

    X = np.array(X)
    y = np.array(y)

    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Train the model
    model.fit(X_scaled, y_scaled)

    # Make predictions for n_forecast_days ahead
    last_input = df['y'].iloc[-timesteps:].values.reshape(1, -1)
    last_input_scaled = scaler_X.transform(last_input)
    y_pred_scaled = []

    for _ in range(n_forecast_days):
        pred_scaled = model.predict(last_input_scaled).reshape(1, -1)
        y_pred_scaled.append(pred_scaled[0])
        last_input_scaled = np.roll(last_input_scaled, -1)
        last_input_scaled[0, -1] = pred_scaled[0, 0]

    # Inverse transform the predictions
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled))[0]
    
    return y_pred

# Streamlit app
st.title("Income and Spending Forecasting")

# User input for data (editable DataFrame)
st.write("Input your data (modify as necessary):")
data = {
    "Date": pd.date_range(start="2024-01-01", periods=60),
    "Income": np.random.randint(5000, 15000, 60),
    "Spending": np.random.randint(3000, 10000, 60)
}
df_user = pd.DataFrame(data)
df_user = st.data_editor(df_user)

# Set parameters for forecasting
timesteps = 14
n_forecast_days = 30
model_income = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_spending = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Forecast income
st.write("Income Forecasting")
df_income = pd.DataFrame({"y": df_user["Income"]})
income_forecast = forecast_with_xgboost(df_income, timesteps, 1, model_income)
st.write(f"Income prediction for next day: {income_forecast[0]:.2f}")

# Forecast spending
st.write("Spending Forecasting")
df_spending = pd.DataFrame({"y": df_user["Spending"]})
spending_forecast = forecast_with_xgboost(df_spending, timesteps, 1, model_spending)
st.write(f"Spending prediction for next day: {spending_forecast[0]:.2f}")

# Calculate balance
balance = income_forecast[0] - spending_forecast[0]
st.metric(label="Balance for next day", value=f"{balance:.2f}")

# Plot actual vs predicted for income and spending
st.write("Plot of Actual vs Predicted (Income and Spending)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_user["Date"], df_user["Income"], label="Actual Income", color='blue')
ax.plot(df_user["Date"], df_user["Spending"], label="Actual Spending", color='red')

# Add predictions to the plot
ax.scatter(df_user["Date"].iloc[-1] + pd.Timedelta(days=1), income_forecast[0], label="Predicted Income", color='green', marker='x')
ax.scatter(df_user["Date"].iloc[-1] + pd.Timedelta(days=1), spending_forecast[0], label="Predicted Spending", color='orange', marker='x')

ax.set_xlabel("Date")
ax.set_ylabel("Amount")
ax.legend()
st.pyplot(fig)
