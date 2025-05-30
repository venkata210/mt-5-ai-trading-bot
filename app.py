from flask import Flask, render_template, request, redirect, url_for, session
import MetaTrader5 as mt5
import threading
from train_model import train_model, trading_bot, initialize_mt5, get_data  # Import get_data
import json
import os
import pandas as pd
import logging  # Ensure logging is imported
from flask_wtf import CSRFProtect
from wtforms import Form, StringField, PasswordField, validators

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(32))
CSRFProtect(app)

# Ensure 'static' folder is correctly set up for serving static files
app.static_folder = 'static'

# Global bot status tracking
BOT_STATUS = {}
BOT_STATUS_LOCK = threading.Lock()

# Path to the trade log file
TRADE_LOG_PATH = 'trade_log.json'

# WTForms for input validation
class LoginForm(Form):
    account = StringField('Account', [validators.DataRequired(), validators.Regexp(r'^\d+$', message="Account must be numeric")])
    password = PasswordField('Password', [validators.DataRequired()])
    server = StringField('Server', [validators.DataRequired()])

class SymbolForm(Form):
    symbol = StringField('Symbol', [validators.DataRequired(), validators.Length(min=3, max=10)])

def get_trade_summary():
    if not os.path.exists(TRADE_LOG_PATH):
        return {
            "total_profit": 0,
            "total_trades": 0,
            "trade_actions": {},
            "profits_over_time": [],
            "trade_dates": [],
            "trades": []
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
        "trade_dates": trade_dates,
        "trades": trades
    }

@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        account = form.account.data
        password = form.password.data
        server = form.server.data

        try:
            # Initialize MT5 connection with user credentials
            if not mt5.initialize():
                raise Exception("Failed to initialize MT5")

            if not mt5.login(int(account), password, server):
                raise Exception("Failed to login to MT5 account")
            else:
                session.clear()
                session['account'] = account
                session['server'] = server
                session['authenticated'] = True
                return redirect(url_for('select_currency'))
        except Exception: # Catching general Exception as MT5 specific exceptions are not well-documented for granular handling here
            logging.exception("Login error:") # Automatically includes traceback
            return render_template('login.html', form=form, error="Login failed. Please check your credentials and MT5 connection.")

    return render_template('login.html', form=form)

@app.route('/select_currency', methods=['GET', 'POST'])
def select_currency():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    form = SymbolForm(request.form)
    if request.method == 'POST' and form.validate():
        symbol = form.symbol.data
        session['symbol'] = symbol

        with BOT_STATUS_LOCK:
            if BOT_STATUS.get(symbol) == 'running':
                # Using flash requires importing it: from flask import flash
                # For now, just log and redirect, or pass a message to template
                logging.info(f"Bot for {symbol} is already running.")
                # flash(f"A trading bot for {symbol} is already running.", "info") # Requires import flash
                return redirect(url_for('dashboard'))

        # Check session['bot_thread'] to prevent this specific session from starting multiple bots
        # if not session.get('bot_thread_started_for_symbol', {}).get(symbol): # More granular session tracking
        if 'bot_thread' not in session: # Simpler session check, assumes one bot per session overall
            # Retrieve credentials from the session
            account = session.get('account')
            server = session.get('server')
            password = request.form.get('password') or ''  # Not stored in session for security
            session['start_time'] = str(pd.Timestamp.now())

            # Start the trading bot in a separate thread, passing credentials
            thread = threading.Thread(
                target=start_trading_bot,
                args=(symbol, account, password, server),
                daemon=True
            )
            thread.start()
            session['bot_thread'] = True

        return redirect(url_for('dashboard'))

    return render_template('select_currency.html', form=form)

def start_trading_bot(symbol, account, password, server):
    with BOT_STATUS_LOCK:
        BOT_STATUS[symbol] = 'running'
    try:
        if symbol and account and password and server:
            initialize_mt5(account, password, server, symbol)
            model = train_model(symbol)
            trading_bot(model, symbol)  # This is an infinite loop
    except Exception: # Catching general Exception as this wraps multiple complex calls from train_model
        logging.exception(f"Trading bot error for {symbol}:") # Automatically includes traceback
        with BOT_STATUS_LOCK:
            BOT_STATUS[symbol] = 'error'
    finally:
        # This block will be reached if trading_bot exits (e.g., unhandled exception inside it,
        # or if it's redesigned to stop) or if initialize_mt5/train_model fails.
        with BOT_STATUS_LOCK:
            current_status = BOT_STATUS.get(symbol)
            if current_status == 'running': # Only mark as stopped if it was 'running' and didn't hit the 'error' state
                BOT_STATUS[symbol] = 'stopped'
                logging.info(f"Bot for {symbol} stopped or completed.")
            elif current_status == 'error':
                logging.info(f"Bot for {symbol} remains in 'error' state.")
            else: # e.g. None or 'stopped' already
                logging.info(f"Bot for {symbol} was already {current_status} or finished initialization in error.")


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
        with BOT_STATUS_LOCK:
            bot_status_for_symbol = BOT_STATUS.get(symbol) # Get status for the current symbol
        
        is_trading = bot_status_for_symbol == 'running'
        # You can pass bot_status_for_symbol to the template to show 'error', 'stopped' etc.
        # For example: 'dashboard.html', ..., bot_status=bot_status_for_symbol
        
        # Calculate active time
        start_time = session.get('start_time')
        if start_time:
            active_time = str(pd.Timestamp.now() - pd.Timestamp(start_time)).split('.')[0]
        else:
            active_time = '0:00:00'
        
        # Get historical data
        data = get_data(symbol, mt5.TIMEFRAME_M1, 100)
        dates = data['time'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        closes = data['close'].tolist()
        
        # Get trade summary
        trade_summary = get_trade_summary()
        trades = trade_summary.get('trades', [])
        
        # Calculate win rate
        profitable_trades = len([t for t in trades if t['profit'] > 0])
        total_trades = trade_summary['total_trades']
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Get recent trades (last 10)
        recent_trades = trades[-10:]
        
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
        
    except IOError as e_io: # Specific for file issues in get_trade_summary
        logging.exception("Dashboard error - file I/O issue:")
        return render_template('dashboard.html', symbol=symbol, error="Error loading trade summary. Please check logs.")
    except ValueError as e_val: # Specific for data conversion issues or MT5 data issues
        logging.exception("Dashboard error - data issue:")
        return render_template('dashboard.html', symbol=symbol, error="Error processing data for dashboard. Please check logs.")
    except Exception: # Catch-all for other unexpected errors (e.g. MT5 connection issues)
        logging.exception("Error in dashboard:") # Automatically includes traceback
        return render_template('dashboard.html', symbol=symbol, error="An unexpected error occurred on the dashboard. Please check logs.")

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode)