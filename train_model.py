import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from sklearn.ensemble import RandomForestClassifier
import json
import os
import threading
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute timeframe
SHORT_MA_PERIOD = 5
LONG_MA_PERIOD = 20
LOTS = 0.1
TRADE_LOG_PATH = 'trade_log.json'

# Initialize MT5 connection with parameters
def initialize_mt5(account, password, server):
    if not mt5.initialize():
        logging.error("Failed to initialize MT5")
        raise Exception("MT5 initialization failed")

    if not mt5.login(int(account), password, server):
        logging.error("Failed to login to MT5 account")
        raise Exception("MT5 login failed")
    else:
        logging.info(f"Successfully logged in to account {account}")

        # Check terminal trade permissions
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logging.error("Failed to get terminal info")
            raise Exception("Failed to get terminal info")
        else:
            logging.info(f"Terminal Info - Trade Allowed: {terminal_info.trade_allowed}, Trade API Disabled: {terminal_info.tradeapi_disabled}")
            if not terminal_info.trade_allowed or terminal_info.tradeapi_disabled:
                logging.error("Trading is not allowed by the MetaTrader 5 terminal. Please enable 'Algo Trading' in the terminal.")
                raise Exception("Trading not allowed by MetaTrader 5 terminal")

        # Retrieve symbol information to get volume constraints
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            logging.error(f"Symbol {SYMBOL} not found.")
            raise Exception(f"Symbol {SYMBOL} not found.")

        global MIN_VOLUME, MAX_VOLUME, VOLUME_STEP
        MIN_VOLUME = symbol_info.volume_min
        MAX_VOLUME = symbol_info.volume_max
        VOLUME_STEP = symbol_info.volume_step
        logging.info(f"Volume constraints - Min: {MIN_VOLUME}, Max: {MAX_VOLUME}, Step: {VOLUME_STEP}")

# Get historical data
def get_data(symbol, timeframe, periods):
    # Ensure the symbol is selected in MetaTrader5
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Symbol {symbol} not found or could not be selected.")
        raise Exception(f"Symbol {symbol} not found or could not be selected.")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, periods)
    if rates is None or len(rates) == 0:
        logging.warning(f"No data received for symbol {symbol}")
        raise ValueError("Failed to retrieve historical data")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Calculate moving averages
def calculate_ma(data, period):
    return data['close'].rolling(window=period).mean()

# New function to calculate technical indicators
def calculate_indicators(data):
    # RSI calculation
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # ATR calculation
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['atr'] = true_range.rolling(14).mean()

    # Momentum
    data['momentum'] = data['close'] - data['close'].shift(10)

    return data

# Add data preparation for AI model
def prepare_features(data):
    data['short_ma'] = calculate_ma(data, SHORT_MA_PERIOD)
    data['long_ma'] = calculate_ma(data, LONG_MA_PERIOD)
    data['ma_diff'] = data['short_ma'] - data['long_ma']
    data = calculate_indicators(data)
    data['target'] = data['close'].shift(-1) > data['close']
    data = data.dropna()
    return data[['ma_diff', 'rsi', 'atr', 'momentum']], data['target']

# Modify train_model to accept symbol
def train_model(symbol):
    data = get_data(symbol, TIMEFRAME, 1000)  # Use the selected symbol
    X, y = prepare_features(data)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logging.info("Model trained.")
    return model

# Update predict_signal function
def predict_signal(model, latest_data):
    prediction = model.predict(latest_data)
    proba = model.predict_proba(latest_data)[0]
    
    # Get the actual values for validation
    rsi = latest_data['rsi'].iloc[0]
    momentum = latest_data['momentum'].iloc[0]
    ma_diff = latest_data['ma_diff'].iloc[0]

    # Enhanced trading rules
    if prediction[0]:  # Model predicts up
        if rsi < 30 and momentum > 0 and ma_diff > 0:
            return "buy"
    else:  # Model predicts down
        if rsi > 70 and momentum < 0 and ma_diff < 0:
            return "sell"
    
    return "hold"  # Default to hold if conditions aren't met

# Modify the log_trade function to handle profit later
def log_trade(action, symbol, volume, price, profit=0):
    trade = {
        "action": action,
        "symbol": symbol,
        "volume": volume,
        "price": price,
        "profit": profit,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, 'w') as f:
            json.dump([trade], f, indent=4)
    else:
        with open(TRADE_LOG_PATH, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(trade)
            f.seek(0)
            json.dump(data, f, indent=4)

# Update place_trade function to include position sizing
def place_trade(action, symbol):
    if action == "hold":
        return

    # Get account info for position sizing
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to get account info")
        return

    # Calculate position size based on risk percentage (1% risk per trade)
    risk_percent = 0.01
    account_balance = account_info.balance
    pip_value = mt5.symbol_info(symbol).point * 10
    risk_amount = account_balance * risk_percent
    
    # Calculate position size based on 50 pip stop loss
    position_size = risk_amount / (50 * pip_value)
    
    # Adjust to symbol's volume constraints
    position_size = max(MIN_VOLUME, min(position_size, MAX_VOLUME))
    position_size = round(round(position_size / VOLUME_STEP) * VOLUME_STEP, 2)
    
    global LOTS
    LOTS = position_size

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get tick info for symbol {symbol}")
        return

    price = tick.ask if action == "buy" else tick.bid
    if price == 0:
        logging.error(f"Invalid price for action {action}")
        return

    # Pre-trade validation
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Symbol {symbol} not selected")
        return

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to retrieve symbol info for {symbol}")
        return

    point = symbol_info.point
    if point == 0:
        logging.error(f"Invalid point size for symbol {symbol}")
        return

    # Validate and adjust LOTS (using the already declared global LOTS)
    LOTS = max(MIN_VOLUME, min(LOTS, MAX_VOLUME))
    LOTS = round(round(LOTS / VOLUME_STEP) * VOLUME_STEP, 2)

    # Define SL and TP distances in points (pip_distance * point size)
    pip_distance = 50
    if action == "buy":
        sl = price - pip_distance * point
        tp = price + pip_distance * point
    else:
        sl = price + pip_distance * point
        tp = price - pip_distance * point

    # Simplify trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOTS,
        "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "Python MT5 Bot",
        # 'type_time': mt5.ORDER_TIME_GTC,  # Optional
        # 'type_filling': mt5.ORDER_FILLING_IOC,  # Set below based on allowed modes
    }

    # Check allowed filling modes
    filling_mode = symbol_info.filling_mode
    if filling_mode & mt5.ORDER_FILLING_FOK:
        request['type_filling'] = mt5.ORDER_FILLING_FOK
    elif filling_mode & mt5.ORDER_FILLING_IOC:
        request['type_filling'] = mt5.ORDER_FILLING_IOC
    else:
        logging.error("No acceptable filling mode available for this symbol.")
        return

    logging.info(f"Placing trade: {request}")

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade failed, retcode: {result.retcode}")
        logging.error(f"Trade result: {result}")
        last_error = mt5.last_error()
        logging.error(f"MT5 last error: {last_error}")
        handle_trade_error(result.retcode)
    else:
        logging.info(f"Trade successful: {result}")
        # Remove the invalid profit access
        # profit = result.profit
        log_trade(action, symbol, LOTS, price)
        # Note: Profit will be logged when the trade is closed

# Function to update trade profit when a trade is closed
def update_trade_profit():
    while True:
        try:
            closed_trades = mt5.history_deals_get()
            if closed_trades is not None:
                for deal in closed_trades:
                    # Extract necessary details
                    order_ticket = deal.order
                    profit = deal.profit
                    symbol = deal.symbol
                    action = "buy" if deal.type == mt5.TRADE_ACTION_DEAL and deal.type_filling == mt5.ORDER_TYPE_BUY else "sell"

                    # Find the corresponding trade in the log
                    with open(TRADE_LOG_PATH, 'r+') as f:
                        try:
                            trades = json.load(f)
                        except json.JSONDecodeError:
                            trades = []
                        for trade in trades:
                            if trade["action"] == action and trade["symbol"] == symbol and trade["profit"] == 0:
                                trade["profit"] = profit
                                break
                        f.seek(0)
                        json.dump(trades, f, indent=4)
        except Exception as e:
            logging.error(f"Error updating trade profit: {e}")
        time.sleep(60)  # Check every minute

# Start updating trade profits in a separate thread
def start_profit_updater():
    threading.Thread(target=update_trade_profit, daemon=True).start()

# Handle specific trade errors
def handle_trade_error(retcode):
    error_messages = {
        mt5.TRADE_RETCODE_REQUOTE: "Requote",
        mt5.TRADE_RETCODE_REJECT: "Request rejected",
        mt5.TRADE_RETCODE_CANCEL: "Request canceled by trader",
        mt5.TRADE_RETCODE_PLACED: "Order placed",
        mt5.TRADE_RETCODE_DONE: "Deal executed",
        mt5.TRADE_RETCODE_DONE_PARTIAL: "Partial deal executed",
        mt5.TRADE_RETCODE_ERROR: "Generic error",
        mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
        mt5.TRADE_RETCODE_INVALID: "Invalid request",
        mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
        mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
        mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
        mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
        mt5.TRADE_RETCODE_NO_MONEY: "Not enough money",
        mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
        mt5.TRADE_RETCODE_PRICE_OFF: "No quotes",
        mt5.TRADE_RETCODE_INVALID_FILL: "Invalid order filling type",
        mt5.TRADE_RETCODE_INVALID_ORDER: "Invalid order",
        # Add more retcode mappings as needed
    }
    message = error_messages.get(retcode, "Unknown error")
    logging.error(f"Trade error {retcode}: {message}")

# Modify trading_bot to accept symbol
def trading_bot(model, symbol):
    logging.info("Starting trading bot...")
    start_profit_updater()
    while True:
        try:
            data = get_data(symbol, TIMEFRAME, LONG_MA_PERIOD + 1)
            features, _ = prepare_features(data)
            latest_features = features.iloc[-1:].reset_index(drop=True)
            action = predict_signal(model, latest_features)

            place_trade(action, symbol)

        except Exception as e:
            logging.error(f"Error in trading logic: {e}")

        time.sleep(60)  # Wait for a minute before checking again

# Run the bot
if __name__ == "__main__":
    try:
        account = 190331763  # Replace with your account number
        password = "Karthik.413"     # Replace with your password
        server = "Exness-MT5Trial14"    # Replace with your broker's server name
        initialize_mt5(account, password, server)
        model = train_model(SYMBOL)  # Train the model and keep it in memory
        trading_bot(model, SYMBOL)     # Pass the trained model to the trading bot
    except KeyboardInterrupt:
        logging.info("Stopping trading bot...")
    except Exception as e:
        logging.critical(f"Critical error: {e}")
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")
