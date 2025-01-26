import sys
import requests
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


log_file = open("fmp5.log", "w", buffering=1)  
sys.stdout = log_file
sys.stderr = log_file

API_KEY = "7d249bca36be144c39b2e02124d0baaf"
SYMBOL = "GME"
INITIAL_BALANCE = 10000
TRANSACTION_FEE = 2.5
STOP_LOSS_PCT = 0.07

def fetch_fail_to_deliver(symbol, api_key):
    """Fetch FTD (Fail to Deliver) data with pagination."""
    fail_df = []
    page = 0
    while True:
        url = f"https://financialmodelingprep.com/api/v4/fail_to_deliver?symbol={symbol}&page={page}&apikey={api_key}"
        resp = requests.get(url).json()
        if not resp:
            break
        fail_df.extend(resp)
        page += 1
    
    df = pd.DataFrame(fail_df)
    df['date'] = pd.to_datetime(df['date'])
    return df.dropna(subset=['date']).sort_values('date')

def fetch_historical_prices(symbol, api_key):
    """Fetch daily OHLCV data from the Historical Price Full API."""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    data = requests.get(url).json().get('historical', [])
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

def compute_rsi(prices, window=14):
    """
    Computes RSI (Relative Strength Index) from a Pandas Series of prices,
    without external libraries.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Exponential Moving Average for smoothing
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_features(df):
    """Enhance feature engineering with FTD data, price changes, and RSI."""
    df['fail_ratio'] = df['fail_qty'] / df['volume'].replace(0, 1)
    df['price_change'] = df['close'].pct_change()
    df['ma7'] = df['close'].rolling(7).mean()
    df['ma21'] = df['close'].rolling(21).mean()
    df['rsi'] = compute_rsi(df['close'], window=14)
    df['volatility'] = df['close'].rolling(21).std()

    # Lagged features for fail_ratio
    for lag in [1, 3, 5]:
        df[f'fail_ratio_lag{lag}'] = df['fail_ratio'].shift(lag)

    # Simple trend indicator
    df['trend'] = np.where(df['ma7'] > df['ma21'], 1, -1)

    return df.dropna()

def prepare_data(ftd_df, price_df):
    """Merge the FTD dataframe with price data and generate features."""
    merged = pd.merge(
        price_df, 
        ftd_df[['date','quantity']], 
        on='date', 
        how='left'
    )
    merged.rename(columns={'quantity': 'fail_qty'}, inplace=True)
    merged['fail_qty'] = merged['fail_qty'].fillna(0)
    
    merged = calculate_features(merged)
    return merged

class ShortSqueezeEnv(gym.Env):
    """
    A custom Gymnasium environment for trading GME (or any symbol)
    with fail-to-deliver data + risk management (stop loss).
    """

    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # We'll collect 14 numeric features + 1 for balance ratio => total 15
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(15,), 
            dtype=np.float32
        )

        self.stop_loss = STOP_LOSS_PCT
        self.transaction_fee = TRANSACTION_FEE

        # Initialize state
        self.current_step = 0
        self.balance = INITIAL_BALANCE
        self.shares = 0
        self.entry_price = 0

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = INITIAL_BALANCE
        self.shares = 0
        self.entry_price = 0

        observation = self._next_observation()
        return observation, {}

    def _next_observation(self):
        """Constructs the observation vector of features."""
        row = self.df.loc[self.current_step]
        
        features = [
            row['fail_qty'],
            row['fail_ratio'],
            row['volume'],
            row['open'],      
            row['close'],
            row['ma7'],
            row['ma21'],
            row['rsi'],
            row['volatility'],
            row['price_change'],
            row['trend'],
            row['fail_ratio_lag1'],
            row['fail_ratio_lag3'],
            row['fail_ratio_lag5']
        ]
        # ratio of current balance to the initial balance
        balance_ratio = [self.balance / INITIAL_BALANCE]

        obs = np.array(features + balance_ratio, dtype=np.float32)
        return obs

    def _execute_stop_loss(self, current_price):
        """If stop loss is triggered, close the position and penalize the agent."""
        if self.shares > 0 and current_price < self.entry_price * (1 - self.stop_loss):
            self.balance += self.shares * current_price - self.transaction_fee
            self.shares = 0
            return True
        return False

    def step(self, action):
        """
        Executes the action in the environment,
        returns (obs, reward, done, truncated, info)
        """
        if self.current_step >= self.max_steps:
            obs = self._next_observation()
            return obs, 0.0, True, False, {}

        current_price = self.df.loc[self.current_step, 'close']
        next_price = self.df.loc[self.current_step + 1, 'close']

        # Check stop loss first
        stop_triggered = self._execute_stop_loss(current_price)
        reward = -10.0 if stop_triggered else 0.0

        # Execute new action if stop wasn’t triggered
        if action == 1 and not stop_triggered:  # Buy/Long
            max_shares = int((self.balance - self.transaction_fee) // current_price)
            if max_shares > 0:
                self.shares += max_shares
                self.balance -= (max_shares * current_price + self.transaction_fee)
                self.entry_price = current_price

        elif action == 2 and not stop_triggered:  # Sell if we have shares
            if self.shares > 0:
                self.balance += (self.shares * current_price - self.transaction_fee)
                self.shares = 0

        # Calculate reward based on next day's portfolio value
        portfolio_value = self.balance + self.shares * next_price
        reward += (portfolio_value - INITIAL_BALANCE) / 100.0

        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        truncated = False

        obs = self._next_observation()
        return obs, float(reward), done, truncated, {}

def main():
    print("Fetching Fail to Deliver data...")
    ftd_df = fetch_fail_to_deliver(SYMBOL, API_KEY)

    print("Fetching historical price data...")
    price_df = fetch_historical_prices(SYMBOL, API_KEY)

    print("Merging datasets and computing features...")
    merged_df = prepare_data(ftd_df, price_df)
    if merged_df.empty:
        print("No data available after merging. Exiting.")
        return

    print("Setting up the ShortSqueezeEnv...")
    env = ShortSqueezeEnv(merged_df)
    check_env(env)  

    print("Training the PPO model...")
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )
    model.learn(total_timesteps=25000)

    print("Evaluating the trained model...")
    obs, _ = env.reset()
    step_count = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        print(f"Step {step_count} | Action: {action} | "
              f"Balance: ${env.balance:.2f} | Shares: {env.shares} | Reward: {reward:.2f}")

        if done or truncated:
            break

    print(f"Final Balance: ${env.balance:.2f}\nDone.")

if __name__ == "__main__":
    main()
