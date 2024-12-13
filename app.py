from flask import Flask, render_template, request, redirect, url_for, session
import MetaTrader5 as mt5
import threading
from train_model import train_model, trading_bot, initialize_mt5, get_data  # Import get_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

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
    # Retrieve the selected symbol from the session
    symbol = session.get('symbol')
    if not symbol:
        return redirect(url_for('select_currency'))
    
    try:
        # Fetch historical data for the graph
        data = get_data(symbol, mt5.TIMEFRAME_M1, 100)  # Fetch last 100 data points
        dates = data['time'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        closes = data['close'].tolist()
    except Exception as e:
        logging.error(f"Error fetching data for dashboard: {e}")
        dates = []
        closes = []
    
    return render_template('dashboard.html', dates=dates, closes=closes)

if __name__ == '__main__':
    app.run(debug=True)