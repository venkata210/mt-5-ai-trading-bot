import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from sklearn.ensemble import RandomForestClassifier
import json
import os
import threading
import numpy as np
import getpass
from contextlib import contextmanager
import fcntl  # Only works on Unix, so fallback for Windows below

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Exceptions
class MT5Error(Exception):
    """Base class for MetaTrader 5 related errors."""
    pass

class MT5ConnectionError(MT5Error):
    """Raised for errors during MT5 initialization or login."""
    pass

class MT5SymbolError(MT5Error, LookupError):
    """Raised for errors related to symbol not found or selection issues."""
    pass

class MT5TradeError(MT5Error):
    """Raised for errors during trade execution if not handled by retcodes."""
    pass

# Constants
DEFAULT_SYMBOL = "EURUSD"  # Default symbol
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute timeframe
SHORT_MA_PERIOD = 5
LONG_MA_PERIOD = 20
TRADE_LOG_PATH = 'trade_log.json'

# Use environment variables for credentials
ACCOUNT = os.getenv('MT5_ACCOUNT')
PASSWORD = os.getenv('MT5_PASSWORD')
SERVER = os.getenv('MT5_SERVER')

# File lock context manager for cross-platform
@contextmanager
def file_lock(fp):
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)

# Initialize MT5 connection with parameters
def initialize_mt5(account=None, password=None, server=None, symbol=DEFAULT_SYMBOL):  # Added symbol argument with default
    account = account or ACCOUNT
    password = password or PASSWORD
    server = server or SERVER
    if not account or not password or not server:
        # Not using logging.critical here as it might be too strong for a library function.
        # Raising an error is sufficient.
        raise ValueError("MT5 credentials (account, password, server) are required.")
    if not mt5.initialize():
        # logging.error("Failed to initialize MT5. Last error: %s", mt5.last_error()) # mt5.last_error() might be useful
        raise MT5ConnectionError(f"Failed to initialize MT5. Last error: {mt5.last_error()}")
    
    try: # Attempt login as int first, then try str if that fails (though account is usually int)
        login_account = int(account)
    except ValueError:
        raise ValueError("Account ID must be an integer.")

    if not mt5.login(login_account, password, server):
        # logging.error("Failed to login to MT5 account %s. Last error: %s", account, mt5.last_error())
        raise MT5ConnectionError(f"Failed to login to MT5 account {account}. Last error: {mt5.last_error()}")
    
    logging.info(f"Successfully logged in to account {account}")
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        # logging.error("Failed to get terminal info. Last error: %s", mt5.last_error())
        raise MT5ConnectionError(f"Failed to get terminal info. Last error: {mt5.last_error()}")
    if not terminal_info.trade_allowed or terminal_info.tradeapi_disabled:
        # logging.error("Trading is not allowed by the MetaTrader 5 terminal. Please enable 'Algo Trading'.")
        raise MT5ConnectionError("Trading is not allowed by the MetaTrader 5 terminal. Please enable 'Algo Trading'.")
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        # logging.error(f"Symbol {symbol} not found. Last error: %s", mt5.last_error())
        raise MT5SymbolError(f"Symbol {symbol} not found. Last error: {mt5.last_error()}")
    
    global MIN_VOLUME, MAX_VOLUME, VOLUME_STEP
    MIN_VOLUME = symbol_info.volume_min
        MAX_VOLUME = symbol_info.volume_max
        VOLUME_STEP = symbol_info.volume_step
        logging.info(f"Volume constraints - Min: {MIN_VOLUME}, Max: {MAX_VOLUME}, Step: {VOLUME_STEP}")

# Get historical data
def get_data(symbol, timeframe, periods):
    # Ensure the symbol is selected in MetaTrader5
    if not mt5.symbol_select(symbol, True):
        # logging.error(f"Symbol {symbol} not found or could not be selected. Last error: %s", mt5.last_error())
        raise MT5SymbolError(f"Symbol {symbol} not found or could not be selected. Last error: {mt5.last_error()}")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, periods)
    if rates is None or len(rates) == 0:
        # logging.warning will remain, as this is a valid operational state (no data yet) but needs handling.
        raise ValueError(f"No data received for symbol {symbol} for the specified periods.")

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

    # Initialize rsi column
    data['rsi'] = np.nan

    # Case 1: loss == 0 and gain == 0 (RSI = 50)
    condition_neutral = (loss == 0) & (gain == 0)
    data.loc[condition_neutral, 'rsi'] = 50.0

    # Case 2: loss == 0 and gain > 0 (RSI = 100)
    condition_overbought = (loss == 0) & (gain > 0)
    data.loc[condition_overbought, 'rsi'] = 100.0

    # Case 3: loss > 0 (normal calculation)
    # This also covers gain == 0 and loss > 0, resulting in RSI = 0.
    condition_normal = loss > 0
    rs_normal = gain[condition_normal] / loss[condition_normal]
    data.loc[condition_normal, 'rsi'] = 100.0 - (100.0 / (1.0 + rs_normal))
    
    # Fill initial NaNs from rolling mean with a neutral 50, or forward fill
    # data['rsi'] = data['rsi'].fillna(50) # Option 1: fill with 50
    data['rsi'] = data['rsi'].ffill() # Option 2: forward fill, then backfill for any leading NaNs
    data['rsi'] = data['rsi'].bfill()


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
def log_trade(action, symbol, volume, price, order_ticket, profit=0):  # Added order_ticket
    trade = {
        "action": action,
        "symbol": symbol,
        "volume": volume,
        "price": price,
        "profit": profit,
        "timestamp": pd.Timestamp.now().isoformat(),
        "order_ticket": order_ticket  # Store order_ticket
    }
    # Thread-safe file write
    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, 'w+') as f:
            with file_lock(f):
                json.dump([trade], f, indent=4)
    else:
        with open(TRADE_LOG_PATH, 'r+') as f:
            with file_lock(f):
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(trade)
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()

# Update place_trade function to include position sizing
def place_trade(action, symbol):
    if action == "hold":
        return
    try:
        # Get account info for position sizing
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info for position sizing.") # No traceback needed, it's an expected check
            # Consider raising MT5ConnectionError if this implies a deeper issue
            return

        # Calculate position size based on risk percentage (1% risk per trade)
    risk_percent = 0.01
    account_balance = account_info.balance
    pip_value = mt5.symbol_info(symbol).point * 10
    risk_amount = account_balance * risk_percent
    
    # Calculate position size based on 50 pip stop loss
    position_size = risk_amount / (50 * pip_value)
    
    # Adjust to symbol's volume constraints
    # Re-fetch symbol_info for robustness in getting volume constraints
    symbol_info_local = mt5.symbol_info(symbol)
    if not symbol_info_local:
        logging.error(f"Failed to get symbol_info for {symbol} in place_trade for volume constraints")
        return

    min_vol = symbol_info_local.volume_min
    max_vol = symbol_info_local.volume_max
    vol_step = symbol_info_local.volume_step

    # Ensure pip_value is valid, otherwise default or log error
    if pip_value <= 0:
        logging.error(f"Invalid pip_value {pip_value} for symbol {symbol}. Cannot calculate position size.")
        return # Or handle with a default/fallback position size if appropriate

    trade_volume = risk_amount / (50 * pip_value) # Initial calculation based on risk
    
    # Adjust to symbol's volume constraints using local variables
    trade_volume = max(min_vol, min(trade_volume, max_vol))
    trade_volume = round(round(trade_volume / vol_step) * vol_step, 2) # Ensure precision

    # Ensure trade_volume is not zero or negative after adjustments
    if trade_volume <= 0:
        logging.error(f"Calculated trade volume {trade_volume} is invalid for symbol {symbol}.")
        return

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
        "volume": trade_volume, # Use the local, processed trade_volume
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
        # These logs are already specific and helpful. handle_trade_error also logs.
        # No need for exc_info=True here as it's not an exception path.
        logging.error(f"Trade failed for {symbol}, retcode: {result.retcode}, result: {result}")
        last_mt5_error = mt5.last_error()
        logging.error(f"MT5 last error for failed trade: {last_mt5_error}")
        handle_trade_error(result.retcode) # This function logs the meaning of the retcode
        # Optionally, raise MT5TradeError here if you want to propagate it
        # raise MT5TradeError(f"Trade failed for {symbol} with retcode {result.retcode}. MT5 Error: {last_mt5_error}")
    else:
        logging.info(f"Trade successful for {symbol}: {result}")
        order_ticket = result.order  # Get order ticket
        # Remove the invalid profit access
        # profit = result.profit
        log_trade(action, symbol, trade_volume, price, order_ticket) # Pass local trade_volume
        # Note: Profit will be logged when the trade is closed
    except MT5Error as e_mt5: # Catch specific MT5 errors if they occur unexpectedly
        logging.exception(f"MT5 error during trade placement for {symbol}:")
        # Re-raise or handle as appropriate for the bot's logic
        raise
    except Exception: # Catch any other unexpected errors
        logging.exception(f"Unexpected error during trade placement for {symbol}:")
        # Re-raise or handle
        raise


# Function to update trade profit when a trade is closed
def update_trade_profit():
    while True:
        try:
            # Potentially, mt5.history_deals_get() could raise an MT5 error if connection is lost.
            closed_trades = mt5.history_deals_get() 
            if closed_trades is not None:
                trades_from_log = []
                try:
                    with open(TRADE_LOG_PATH, 'r+') as f: # Changed to r+ to allow reading then writing
                        with file_lock(f):
                            trades_from_log = json.load(f)
                except FileNotFoundError:
                    logging.info(f"Trade log file {TRADE_LOG_PATH} not found. Will be created.")
                    trades_from_log = [] # Ensure it's a list
                except json.JSONDecodeError:
                    logging.exception(f"Error decoding JSON from {TRADE_LOG_PATH}. File might be corrupted.")
                    # Decide on recovery strategy: backup, rename, or skip update cycle
                    time.sleep(60) # Wait before retrying to avoid rapid logging
                    continue 
                except IOError:
                    logging.exception(f"IOError when accessing {TRADE_LOG_PATH}.")
                    time.sleep(60)
                    continue

                needs_update = False
                for deal in closed_trades:
                    deal_order_ticket = deal.order
                    deal_profit = deal.profit
                    for trade_index, trade in enumerate(trades_from_log):
                        if trade.get("order_ticket") == deal_order_ticket and trade.get("profit", 0) == 0:
                            trades_from_log[trade_index]["profit"] = deal_profit
                            logging.info(f"Updated profit for order ticket {deal_order_ticket} to {deal_profit}")
                            needs_update = True
                            break 

                if needs_update:
                    try:
                        # Overwriting the file completely. Consider safer approaches for critical data (e.g., temp file then rename)
                        with open(TRADE_LOG_PATH, 'w') as f: # Changed to 'w' for overwrite
                            with file_lock(f):
                                json.dump(trades_from_log, f, indent=4)
                    except IOError:
                        logging.exception(f"IOError when writing updated trades to {TRADE_LOG_PATH}.")
            
        except MT5Error: # Catch MT5 related errors from history_deals_get()
            logging.exception("MT5 error during update_trade_profit:")
        except Exception: # Catch any other unexpected errors
            logging.exception("Unexpected error in update_trade_profit:")
        time.sleep(60)  # Check every minute

# Start updating trade profits in a separate thread
def start_profit_updater():
    if not getattr(start_profit_updater, "started", False):
        threading.Thread(target=update_trade_profit, daemon=True).start()
        start_profit_updater.started = True

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

        except MT5Error as e_mt5: # Catch specific MT5 errors that might propagate
            logging.exception(f"MT5 error in trading_bot main loop for {symbol}:")
            # Depending on error, might need to attempt re-initialization or stop.
            # For now, log and continue loop, assuming transient issue.
        except ValueError as e_val: # E.g., from get_data if no data, or prepare_features
            logging.exception(f"ValueError in trading_bot main loop for {symbol}:")
        except Exception: # Catch any other unexpected errors
            logging.exception(f"Unexpected error in trading_bot main loop for {symbol}:")

        time.sleep(60)  # Wait for a minute before checking again

# Run the bot
if __name__ == "__main__":
    try:
        # Use environment variables or prompt for credentials
        account = os.getenv('MT5_ACCOUNT') or input('Enter MT5 account: ')
        password = os.getenv('MT5_PASSWORD') or getpass.getpass('Enter MT5 password: ')
        server = os.getenv('MT5_SERVER') or input('Enter MT5 server: ')
        initialize_mt5(account, password, server, DEFAULT_SYMBOL)
        model = train_model(DEFAULT_SYMBOL)
        trading_bot(model, DEFAULT_SYMBOL)
    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user (KeyboardInterrupt).")
    except MT5ConnectionError:
        logging.exception("Failed to connect or login to MT5 in main execution:")
    except MT5SymbolError:
        logging.exception("Symbol related error in main execution:")
    except ValueError as ve: # Catch ValueErrors from initialize_mt5 (e.g. bad account ID) or get_data
        logging.exception(f"Configuration or data error in main execution: {ve}")
    except Exception: # Catch-all for any other unhandled exceptions
        logging.exception("Critical unhandled error in main execution:")
    finally:
        logging.info("Shutting down MT5 connection...")
        if mt5.shutdown():
            logging.info("MT5 connection shut down successfully.")
        else:
            # This might indicate an issue, but often shutdown is called when connection is already lost.
            # logging.warning("MT5 shutdown reported failure, possibly already disconnected. Last error: %s", mt5.last_error())
            logging.warning("MT5 shutdown reported failure. Check MT5 terminal logs if issues persist.")
        logging.info("MT5 connection closed")
