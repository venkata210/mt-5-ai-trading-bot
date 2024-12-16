from flask import Flask, render_template, request, redirect, url_for, session
import MetaTrader5 as mt5
import threading
from train_model import train_model, trading_bot, initialize_mt5, get_data  # Import get_data
import json
import os
import pandas as pd
import logging  # Ensure logging is imported

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

# Ensure 'static' folder is correctly set up for serving static files
app.static_folder = 'static'

# Path to the trade log file
TRADE_LOG_PATH = 'trade_log.json'

def get_trade_summary():
    if not os.path.exists(TRADE_LOG_PATH):
        return {
            "total_profit": 0,
            "total_trades": 0,
            "trade_actions": {},
            "profits_over_time": [],
            "trade_dates": []
        }
    
    with open(TRADE_LOG_PATH, 'r') as f:
        try:
            trades = json.load(f)
        except json.JSONDecodeError:
            trades = []
    
    total_profit = sum(trade['profit'] for trade in trades)
    total_trades = len(trades)
    
    trade_actions = {}
    for trade in trades:
        action = trade['action']
        trade_actions[action] = trade_actions.get(action, 0) + 1
    
    # Prepare data for profit over time
    profits_over_time = [trade['profit'] for trade in trades]
    trade_dates = [pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M') for trade in trades]
    
    return {
        "total_profit": total_profit,
        "total_trades": total_trades,
        "trade_actions": trade_actions,
        "profits_over_time": profits_over_time,
        "trade_dates": trade_dates
    }

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        account = request.form['account']
        password = request.form['password']
        server = request.form['server']

        # Initialize MT5 connection with user credentials
        if not mt5.initialize():
            return "Failed to initialize MT5"

        if not mt5.login(int(account), password, server):
            return "Failed to login to MT5 account"
        else:
            session['account'] = account
            session['password'] = password
            session['server'] = server
            return redirect(url_for('select_currency'))

    return render_template('login.html')

@app.route('/select_currency', methods=['GET', 'POST'])
def select_currency():
    if request.method == 'POST':
        symbol = request.form['symbol']
        session['symbol'] = symbol

        # Retrieve credentials from the session
        account = session.get('account')
        password = session.get('password')
        server = session.get('server')

        # Start the trading bot in a separate thread, passing credentials
        threading.Thread(
            target=start_trading_bot,
            args=(symbol, account, password, server)
        ).start()

        return redirect(url_for('dashboard'))

    return render_template('select_currency.html')

def start_trading_bot(symbol, account, password, server):
    if symbol and account and password and server:
        initialize_mt5(account, password, server)
        model = train_model(symbol)  # Modify train_model to accept symbol
        trading_bot(model, symbol)   # Modify trading_bot to accept symbol

@app.route('/dashboard')
def dashboard():
    symbol = session.get('symbol')
    if not symbol:
        return redirect(url_for('select_currency'))
    
    try:
        # Get account info
        account_info = mt5.account_info()
        account_balance = account_info.balance if account_info else 0
        account_equity = account_info.equity if account_info else 0
        
        # Get trading status
        is_trading = True  # You'll need to implement actual trading status check
        
        # Calculate active time
        start_time = session.get('start_time', pd.Timestamp.now())
        active_time = str(pd.Timestamp.now() - pd.Timestamp(start_time)).split('.')[0]
        
        # Get historical data
        data = get_data(symbol, mt5.TIMEFRAME_M1, 100)
        dates = data['time'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        closes = data['close'].tolist()
        
        # Get trade summary
        trade_summary = get_trade_summary()
        
        # Calculate win rate
        profitable_trades = len([t for t in trade_summary.get('trades', []) if t['profit'] > 0])
        total_trades = trade_summary['total_trades']
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Get recent trades (last 10)
        recent_trades = trade_summary.get('trades', [])[-10:]
        
        return render_template(
            'dashboard.html',
            symbol=symbol,
            is_trading=is_trading,
            account_balance=account_balance,
            account_equity=account_equity,
            active_time=active_time,
            dates=dates,
            closes=closes,
            total_profit=trade_summary["total_profit"],
            total_trades=trade_summary["total_trades"],
            win_rate=win_rate,
            trade_actions=trade_summary["trade_actions"],
            recent_trades=recent_trades
        )
        
    except Exception as e:
        logging.error(f"Error in dashboard: {e}")
        return render_template('dashboard.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)