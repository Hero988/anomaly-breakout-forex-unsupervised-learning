import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load environment variables for MT5 login credentials
load_dotenv()
login = int(os.getenv('MT5_LOGIN'))  # Replace with your login ID
password = os.getenv('MT5_PASSWORD')  # Replace with your password
server = os.getenv('MT5_SERVER')  # Replace with your server name

# Initialize MetaTrader 5 connection
if not mt5.initialize(login=login, password=password, server=server):
    print("Failed to initialize MT5, error code:", mt5.last_error())
    quit()

def train_anomaly_model(df, output_dir):
    """
    Train an anomaly detection model (Isolation Forest) for breakout detection.

    Parameters:
        df (DataFrame): Historical data for training.
        output_dir (str): Directory to save the trained model and scaler.
    """
    # Assume the target breakout signal is identified by 'close' column or similar
    if 'close' not in df.columns:
        raise ValueError("The dataframe must contain a 'close' column for breakout detection.")

    # Compute percentage change and breakout signal
    df['pct_change'] = df['close'].pct_change()

    # Focus on upward breakouts: filtering positive changes
    df['breakout_signal'] = (df['pct_change'] > df['pct_change'].quantile(0.95)).astype(int)

    # Prepare features for training (excluding the breakout_signal)
    features = df[['pct_change']].fillna(0)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(features_scaled)

    # Save model and scaler
    model_path = os.path.join(output_dir, "isolation_forest_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model and scaler saved in {output_dir}")

def evaluate_anomaly_model(df, output_dir, model_path, scaler_path, pair_folder):
    """
    Evaluate the trained anomaly detection model on new data and simulate trading.

    Parameters:
        df (DataFrame): New data for evaluation.
        output_dir (str): Directory to save evaluation results.
        model_path (str): Path to the trained model.
        scaler_path (str): Path to the trained scaler.
        pair_folder (str): Identifier for the data being evaluated.
    """
    # Load model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Adjust lot size for JPY pairs
    lot_size = 1.0  # Default lot size
    if "JPY" in pair_folder:
        lot_size = 0.1  # Adjust to a smaller, more reasonable lot size for JPY pairs

    # Initialize variables for trading simulation
    equity = 10000  # Starting equity
    max_equity = equity  # For drawdown calculation
    position = None  # Track if we are in a position
    stop_loss = 100  # Stop loss in units
    take_profit = 200  # Take profit in units
    equity_history = []
    drawdown_history = []

    previous_close = None  # To calculate pct_change

    # Simulate trading row by row
    for i, row in df.iterrows():
        if previous_close is not None:
            # Calculate pct_change for the current row
            pct_change = (row['close'] - previous_close) / previous_close
        else:
            pct_change = 0  # No change for the first row
        previous_close = row['close']  # Update previous_close for the next iteration

        # Create a DataFrame for the current row to retain feature names
        current_features = pd.DataFrame({'pct_change': [pct_change]})

        # Scale features and predict anomaly for the current row
        features_scaled = scaler.transform(current_features)
        anomaly_score = model.decision_function(features_scaled)
        is_anomaly = model.predict(features_scaled)
        upward_breakout = (is_anomaly == -1) and (pct_change > 0)

        # Handle trading logic
        if upward_breakout and position is None:
            # Open a buy position
            position = {
                'entry_price': row['close'],
                'stop_loss': row['close'] - stop_loss * mt5.symbol_info(pair_folder).point,
                'take_profit': row['close'] + take_profit * mt5.symbol_info(pair_folder).point
            }

        if position is not None:
            # Check for stop loss or take profit
            if row['low'] <= position['stop_loss']:
                # Stop loss hit
                equity -= stop_loss * lot_size
                position = None
            elif row['high'] >= position['take_profit']:
                # Take profit hit
                equity += take_profit * lot_size
                position = None

        # Track equity and drawdown
        equity_history.append(equity)
        max_equity = max(max_equity, equity)
        drawdown = (max_equity - equity) / max_equity
        drawdown_history.append(drawdown)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Equity plot
    axs[0].plot(equity_history, label="Equity", color='blue')
    axs[0].set_title(f"Equity Balance Over Time ({pair_folder})")
    axs[0].set_ylabel("Equity Balance")
    axs[0].legend()
    axs[0].grid()

    # Drawdown plot
    axs[1].plot([d * max_equity for d in drawdown_history], label="Drawdown", color='red', linestyle='--')
    axs[1].set_title(f"Drawdown Over Time ({pair_folder})")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Drawdown Value")
    axs[1].legend()
    axs[1].grid()

    # Save the combined plot
    plot_file = os.path.join(output_dir, f"{pair_folder}_equity_drawdown_plot.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"Equity and drawdown plot saved in {plot_file}")

def main_function(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the main output directory exists

    for pair_folder in os.listdir(data_dir):
        pair_path = os.path.join(data_dir, pair_folder)
        if not os.path.isdir(pair_path):
            continue

        # Create a subfolder inside output_dir for the pair_folder
        pair_output_dir = os.path.join(output_dir, pair_folder)
        os.makedirs(pair_output_dir, exist_ok=True)

        # Load Data
        five_years_data_path = os.path.join(pair_path, f"{pair_folder}_5_years.csv")
        one_year_data_path = os.path.join(pair_path, f"{pair_folder}_2024_present.csv")
        if not os.path.exists(five_years_data_path) or not os.path.exists(one_year_data_path):
            print(f"Data files missing for {pair_folder}")
            continue

        df_five_years = pd.read_csv(five_years_data_path)
        df_1_year = pd.read_csv(one_year_data_path)

        # Train the model using five years of data
        train_anomaly_model(df_five_years, pair_output_dir)

        # Evaluate the model using one year of data
        model_path = os.path.join(pair_output_dir, "isolation_forest_model.pkl")
        scaler_path = os.path.join(pair_output_dir, "scaler.pkl")
        evaluate_anomaly_model(df_1_year, pair_output_dir, model_path, scaler_path, pair_folder)

        print(f"Completed anomaly detection and trading simulation for {pair_folder}")

# Set directories
data_directory = "forex_data_pair_per_folder"
output_directory = "forex_result"
main_function(data_directory, output_directory)
