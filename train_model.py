import MetaTrader5 as mt5
import pandas as pd
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute timeframe
SHORT_MA_PERIOD = 5
LONG_MA_PERIOD = 20
LOTS = 0.1

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MT5")
        raise Exception("MT5 initialization failed")
    
    account = 190331763  # Replace with your account number
    password = "Karthik.413"  # Replace with your password
    server = "Exness-MT5Trial14"  # Replace with your broker's server name

    if not mt5.login(account, password, server):
        logging.error("Failed to login to MT5 account")
        raise Exception("MT5 login failed")
    else:
        logging.info(f"Successfully logged in to account {account}")

# Get historical data
def get_data(symbol, timeframe, periods):
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

# Place a trade
def place_trade(action):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOTS,
        "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(SYMBOL).ask if action == "buy" else mt5.symbol_info_tick(SYMBOL).bid,
        "deviation": 20,
        "magic": 123456,
        "comment": "Python MT5 Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade failed, retcode: {result.retcode}")
    else:
        logging.info(f"Trade successful: {result}")

# Main trading logic
def trading_bot():
    logging.info("Starting trading bot...")
    while True:
        try:
            data = get_data(SYMBOL, TIMEFRAME, LONG_MA_PERIOD + 1)
            data['short_ma'] = calculate_ma(data, SHORT_MA_PERIOD)
            data['long_ma'] = calculate_ma(data, LONG_MA_PERIOD)

            if len(data) < LONG_MA_PERIOD:
                logging.info("Not enough data for analysis, waiting 60 seconds...")
                time.sleep(60)
                continue

            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]

            # Buy signal
            if prev_row['short_ma'] <= prev_row['long_ma'] and last_row['short_ma'] > last_row['long_ma']:
                place_trade("buy")

            # Sell signal
            elif prev_row['short_ma'] >= prev_row['long_ma'] and last_row['short_ma'] < last_row['long_ma']:
                place_trade("sell")

        except Exception as e:
            logging.error(f"Error in trading logic: {e}")

        time.sleep(60)  # Wait for a minute before checking again

# Run the bot
if __name__ == "__main__":
    try:
        initialize_mt5()
        trading_bot()
    except KeyboardInterrupt:
        logging.info("Stopping trading bot...")
    except Exception as e:
        logging.critical(f"Critical error: {e}")
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")
