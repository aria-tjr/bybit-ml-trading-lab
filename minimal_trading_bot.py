#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced AI-Driven Automated Futures Trading Script for BYBIT Exchange
Version: 2.0
Date: March 8, 2025

This script implements a sophisticated AI-driven trading strategy for cryptocurrency
futures trading on the BYBIT exchange. It uses TensorFlow for predictive modeling,
technical indicators for market analysis, and includes comprehensive risk management.

DISCLAIMER: This script is for educational purposes only. Trading cryptocurrencies
involves substantial risk and may not be suitable for everyone. Past performance is
not indicative of future results. Use this script at your own risk.
"""

# ------------------------------------------------------------------------------
# Imports and Configuration
# ------------------------------------------------------------------------------
import os
import sys
import time
import json
import logging
import datetime
import traceback
import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import threading
import asyncio
from typing import Dict, List, Tuple, Union, Optional

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure matplotlib for non-interactive environments
matplotlib.use('Agg')

# Fix for Keras imports - use flexible imports to handle different versions
try:
    # Try standalone keras imports first (for keras 3.x)
    from keras.models import Sequential, load_model, save_model
    from keras.layers import Dense, LSTM, Dropout, GRU
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    KERAS_SOURCE = "standalone"
except ImportError:
    # Fall back to tensorflow.keras (for TF 2.x with built-in keras)
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    KERAS_SOURCE = "tensorflow"

import talib
import tensorflow as tf

# Optional imports with graceful degradation
try:
    import telegram
    from telegram.ext import ApplicationBuilder, CommandHandler

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    from pybit.unified_trading import HTTP, WebSocket

    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False

# Print library versions for debugging
print(f"Using {'standalone' if KERAS_SOURCE == 'standalone' else 'TensorFlow'} Keras")
print(f"Using TensorFlow {tf.__version__}")
print(f"Using Pandas {pd.__version__}")
print(f"Using NumPy {np.__version__}")
print(f"Using Matplotlib {matplotlib.__version__}")

# ------------------------------------------------------------------------------
# Constants and Configuration
# ------------------------------------------------------------------------------

# API Credentials — set via environment variables; never commit real keys.
import os
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Trading parameters
SYMBOL = "BTCUSDT"
LEVERAGE = 3
TIMEFRAMES = ["1", "5", "15", "60"]
MAX_RISK_PER_TRADE = 0.01  # Maximum 1% risk per trade
TRADING_MODE = "simulation"  # "simulation" by default, change to "real" for actual trading

# Additional config parameters
DEBUG_MODE = True  # Enable debug output and simplified models for testing
TELEGRAM_ENABLED = True  # Enable Telegram when credentials are properly set
PERFORMANCE_TRACKING = True  # Track and log performance metrics

# API endpoints
BYBIT_REST_API_URL = "https://api.bybit.com"
BYBIT_WEBSOCKET_PUBLIC_URL = "wss://stream.bybit.com/v5/public"
BYBIT_WEBSOCKET_PRIVATE_URL = "wss://stream.bybit.com/v5/private"


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
def setup_logging(log_dir="logs", debug_mode=False):
    """Configure logging with rotating file handler and console output."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create timestamped log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")

    # Configure log level based on debug mode
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Configure logging format and handlers
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Logging system initialized")
    return log_file


# ------------------------------------------------------------------------------
# Telegram Integration
# ------------------------------------------------------------------------------
class TelegramNotifier:
    """Handles all Telegram notifications and commands."""

    def __init__(self, token: str, chat_id: str):
        """Initialize Telegram bot with token and chat ID."""
        self.token = token
        self.chat_id = chat_id
        self.enabled = False
        self.bot = None
        self.app = None
        self.thread = None
        self.trading_system = None  # Will be set later

        if not TELEGRAM_AVAILABLE:
            logging.warning("Telegram libraries not available. Notifications disabled.")
            return

        if not token or not chat_id:
            logging.warning("Telegram token or chat ID not provided. Notifications disabled.")
            return

        try:
            # Initialize the bot
            self.bot = telegram.Bot(token=token)

            # Initialize the application with the event loop
            self.app = ApplicationBuilder().token(token).build()

            # Register command handlers
            self.app.add_handler(CommandHandler("status", self._cmd_status))
            self.app.add_handler(CommandHandler("balance", self._cmd_balance))
            self.app.add_handler(CommandHandler("positions", self._cmd_positions))
            self.app.add_handler(CommandHandler("performance", self._cmd_performance))

            # Disable thread for now to avoid async issues
            self.enabled = True
            logging.info("Telegram notifications initialized")
            print("✅ Telegram notifications initialized")
        except Exception as e:
            logging.error(f"Error initializing Telegram: {str(e)}")
            print(f"⚠️ Error initializing Telegram: {str(e)}")

    def send_message(self, message: str) -> bool:
        """Send a text message to the specified chat ID."""
        if not self.enabled or not self.bot:
            return False

        try:
            # Create a new event loop for this async operation
            loop = asyncio.new_event_loop()

            async def _send():
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                    return True
                except Exception as e:
                    logging.error(f"Failed to send Telegram message: {str(e)}")
                    return False

            # Run in the new event loop
            result = loop.run_until_complete(_send())
            loop.close()

            if result:
                logging.info(f"Telegram notification sent: {message[:50]}...")

            return result

        except Exception as e:
            logging.error(f"Failed to send Telegram message: {str(e)}")
            return False

    def send_chart(self, chart_path: str, caption: str = "") -> bool:
        """Send a chart image to the specified chat ID with improved connection handling."""
        if not self.enabled or not self.bot:
            return False

        if not os.path.exists(chart_path):
            logging.error(f"Chart file not found: {chart_path}")
            return False

        try:
            # Create a new event loop for this async operation
            loop = asyncio.new_event_loop()

            async def _send_chart():
                try:
                    # Use explicit timeout settings
                    client_session = telegram.request.HTTPXRequest(
                        connection_pool_size=8,  # Increased from default
                        read_timeout=30,  # Increased timeout
                        write_timeout=30,  # Increased timeout
                        connect_timeout=30  # Increased timeout
                    )

                    # Create a temporary bot with these settings
                    temp_bot = telegram.Bot(token=self.token, request=client_session)

                    with open(chart_path, 'rb') as photo:
                        await temp_bot.send_photo(
                            chat_id=self.chat_id,
                            photo=photo,
                            caption=caption
                        )
                    return True
                except Exception as e:
                    logging.error(f"Failed to send chart via Telegram: {str(e)}")
                    return False

            # Run in the new event loop with a timeout
            result = loop.run_until_complete(asyncio.wait_for(_send_chart(), timeout=45))
            loop.close()

            if result:
                logging.info(f"Chart sent via Telegram: {chart_path}")

            return result

        except Exception as e:
            logging.error(f"Failed to send chart via Telegram: {str(e)}")
            return False

    def send_trade_alert(self, action: str, symbol: str, price: float, quantity: float,
                         confidence: float, risk: float) -> bool:
        """Send a well-formatted trade alert."""
        emoji = "🟢" if action.lower() == "buy" else "🔴"
        message = f"{emoji} <b>TRADE EXECUTED: {action.upper()}</b>\n\n" \
                  f"<b>Symbol:</b> {symbol}\n" \
                  f"<b>Price:</b> ${price:.2f}\n" \
                  f"<b>Quantity:</b> {quantity:.6f}\n" \
                  f"<b>Confidence:</b> {confidence:.2%}\n" \
                  f"<b>Risk Level:</b> {risk:.2%}\n" \
                  f"<b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return self.send_message(message)

    def send_error_alert(self, error_type: str, error_message: str, critical: bool = False) -> bool:
        """Send an error alert."""
        alert = "🚨 <b>CRITICAL ERROR</b> 🚨" if critical else "⚠️ <b>ERROR</b>"
        message = f"{alert}\n\n" \
                  f"<b>Type:</b> {error_type}\n" \
                  f"<b>Message:</b> {error_message}\n" \
                  f"<b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return self.send_message(message)

    def send_performance_report(self, data: Dict) -> bool:
        """Send a performance report with key metrics."""
        message = f"📊 <b>PERFORMANCE REPORT</b> 📊\n\n" \
                  f"<b>Period:</b> {data['period']}\n" \
                  f"<b>Net P&L:</b> ${data['net_pnl']:.2f} ({data['pnl_percent']:.2%})\n" \
                  f"<b>Win Rate:</b> {data['win_rate']:.2%}\n" \
                  f"<b>Sharpe Ratio:</b> {data['sharpe_ratio']:.2f}\n" \
                  f"<b>Max Drawdown:</b> {data['max_drawdown']:.2%}\n" \
                  f"<b>AI Accuracy:</b> {data['ai_accuracy']:.2%}\n\n" \
                  f"<b>Total Trades:</b> {data['total_trades']}\n" \
                  f"<b>Winning Trades:</b> {data['winning_trades']}\n" \
                  f"<b>Losing Trades:</b> {data['losing_trades']}\n"
        return self.send_message(message)

    # Command handlers
    async def _cmd_status(self, update, context):
        """Handle /status command - shows system status."""
        if not self.trading_system:
            await update.message.reply_text("Trading system reference not set.")
            return

        status = self.trading_system.get_status()
        message = f"🤖 <b>SYSTEM STATUS</b>\n\n" \
                  f"<b>Running since:</b> {status['running_since']}\n" \
                  f"<b>Trading enabled:</b> {status['trading_enabled']}\n" \
                  f"<b>Current symbol:</b> {status['symbol']}\n" \
                  f"<b>Current leverage:</b> {status['leverage']}x\n" \
                  f"<b>AI model status:</b> {status['ai_status']}\n" \
                  f"<b>Last prediction:</b> {status['last_prediction']}\n" \
                  f"<b>Websocket status:</b> {status['websocket_status']}\n"
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_balance(self, update, context):
        """Handle /balance command - shows account balance."""
        if not self.trading_system:
            await update.message.reply_text("Trading system reference not set.")
            return

        balance = self.trading_system.get_account_balance()
        message = f"💰 <b>ACCOUNT BALANCE</b>\n\n" \
                  f"<b>USDT Balance:</b> ${balance['usdt_balance']:.2f}\n" \
                  f"<b>Unrealized P&L:</b> ${balance['unrealized_pnl']:.2f}\n" \
                  f"<b>Available Balance:</b> ${balance['available_balance']:.2f}\n" \
                  f"<b>Used Margin:</b> ${balance['used_margin']:.2f}\n"
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_positions(self, update, context):
        """Handle /positions command - shows open positions."""
        if not self.trading_system:
            await update.message.reply_text("Trading system reference not set.")
            return

        positions = self.trading_system.get_open_positions()
        if not positions:
            await update.message.reply_text("No open positions.")
            return

        message = "📈 <b>OPEN POSITIONS</b>\n\n"
        for pos in positions:
            side_emoji = "🟢" if pos['side'] == "Buy" else "🔴"
            message += f"{side_emoji} <b>{pos['symbol']}</b>\n" \
                       f"Side: {pos['side']}\n" \
                       f"Size: {pos['size']}\n" \
                       f"Entry Price: ${pos['entry_price']:.2f}\n" \
                       f"Current Price: ${pos['current_price']:.2f}\n" \
                       f"Unrealized P&L: ${pos['unrealized_pnl']:.2f} ({pos['pnl_percent']:.2%})\n" \
                       f"Leverage: {pos['leverage']}x\n\n"
        await update.message.reply_text(message, parse_mode='HTML')

    async def _cmd_performance(self, update, context):
        """Handle /performance command - shows trading performance."""
        if not self.trading_system:
            await update.message.reply_text("Trading system reference not set.")
            return

        perf = self.trading_system.get_performance_metrics()
        await update.message.reply_text(
            f"📊 <b>PERFORMANCE REPORT</b> 📊\n\n"
            f"<b>Period:</b> {perf['period']}\n"
            f"<b>Net P&L:</b> ${perf['net_pnl']:.2f} ({perf['pnl_percent']:.2%})\n"
            f"<b>Win Rate:</b> {perf['win_rate']:.2%}\n"
            f"<b>Total Trades:</b> {perf['total_trades']}\n",
            parse_mode='HTML'
        )

        # Also send the performance chart if available
        chart_path = self.trading_system.generate_performance_chart()
        if chart_path:
            try:
                with open(chart_path, 'rb') as photo:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=photo,
                        caption="Performance Chart"
                    )
            except Exception as e:
                await update.message.reply_text(f"Error sending chart: {str(e)}")


# ------------------------------------------------------------------------------
# AI Model Setup, Training, and Optimization
# ------------------------------------------------------------------------------
class PredictiveModel:
    """AI model for price prediction using TensorFlow/Keras."""

    def __init__(self, symbol: str, timeframes: List[str], model_dir: str = "models", debug_mode: bool = False):
        """Initialize the AI predictive model for market analysis."""
        self.symbol = symbol
        self.timeframes = timeframes
        self.model_dir = model_dir
        self.debug_mode = debug_mode

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Model parameters
        self.look_back = 60  # How many previous time points to use for prediction
        self.features = 20  # Number of features after feature engineering
        self.target_steps = 5  # How many steps ahead to predict

        # Dictionary to store models for each timeframe
        self.models = {}

        # Dictionary to store scalers for feature normalization
        self.scalers = {}

        # Model performance tracking
        self.prediction_history = []
        self.accuracy_metrics = {tf: {"rmse": [], "mae": [], "r2": []} for tf in timeframes}

        # Model training parameters
        self.batch_size = 32
        self.epochs = 50
        self.validation_split = 0.2

        # Track when models were last updated
        self.last_training = {tf: None for tf in timeframes}

        # Initialize or load models for each timeframe
        for tf in self.timeframes:
            self._init_model_for_timeframe(tf)

    def _init_model_for_timeframe(self, timeframe: str):
        """Initialize or load model for a specific timeframe."""
        model_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_model.h5")

        try:
            if os.path.exists(model_path):
                logging.info(f"Loading existing model for {self.symbol} {timeframe}m timeframe")
                try:
                    self.models[timeframe] = load_model(model_path)
                    print(f"✅ Successfully loaded model for {timeframe}m timeframe")
                except Exception as e:
                    logging.error(f"Error loading model for {timeframe}m timeframe: {str(e)}")
                    print(f"⚠️ Failed to load model for {timeframe}m timeframe. Creating new model.")
                    self.models[timeframe] = self._build_model()
            else:
                logging.info(f"Initializing new model for {self.symbol} {timeframe}m timeframe")
                self.models[timeframe] = self._build_model()
                print(f"🆕 Created new model for {timeframe}m timeframe")
        except Exception as e:
            logging.error(f"Error initializing model for {timeframe}m timeframe: {str(e)}")
            print(f"⚠️ Error initializing model: {str(e)}")
            # Fall back to simplified model
            self.models[timeframe] = self._build_simple_model()
            print(f"⚠️ Using fallback simple model for {timeframe}m timeframe")

    def _build_model(self) -> Sequential:
        """Build and compile the LSTM model architecture."""
        try:
            # Use simplified model in debug mode
            if self.debug_mode:
                return self._build_simple_model()

            # Regular LSTM model for production
            model = Sequential()

            # First LSTM layer
            model.add(LSTM(128,
                           return_sequences=True,
                           input_shape=(self.look_back, self.features)))
            model.add(Dropout(0.2))

            # Second LSTM layer
            model.add(LSTM(64, return_sequences=False))
            model.add(Dropout(0.2))

            # Dense output layers
            model.add(Dense(32, activation='relu'))
            model.add(Dense(self.target_steps))  # Output layer

            # Compile model with Adam optimizer
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            return model

        except Exception as e:
            logging.error(f"Error building LSTM model: {str(e)}")
            return self._build_simple_model()

    def _build_simple_model(self) -> Sequential:
        """Build a simplified model for testing or fallback."""
        print("ℹ️ Using simplified model architecture for debug mode")

        try:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(self.look_back, self.features)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(self.target_steps)
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error'
            )

            return model
        except Exception as e:
            logging.error(f"Error building simplified model: {str(e)}")

            # Last resort - ultra simple model
            print("⚠️ Using ultra-simplified model as last resort")
            simple_model = Sequential([
                Dense(32, activation='relu', input_shape=(self.features,)),
                Dense(self.target_steps)
            ])

            simple_model.compile(
                optimizer='adam',
                loss='mean_squared_error'
            )

            return simple_model

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess market data for AI model:
        1. Calculate technical indicators
        2. Normalize features
        3. Create sequences for LSTM input
        """
        # Skip preprocessing if not enough data
        if data is None or data.empty or len(data) < self.look_back + 10:
            # Return dummy data
            dummy_X = np.random.random((10, self.look_back, self.features))
            dummy_y = np.random.random((10, self.target_steps))
            return dummy_X, dummy_y

        # Extract price and volume data
        df = data.copy()

        try:
            # Calculate technical indicators using TA-Lib
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                df['close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle

            # EMA - Different periods
            df['ema_9'] = talib.EMA(df['close'].values, timeperiod=9)
            df['ema_21'] = talib.EMA(df['close'].values, timeperiod=21)
            df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
            df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)

            # Price normalized by EMAs - Distance from moving averages
            df['price_ema_9_ratio'] = df['close'] / df['ema_9']
            df['price_ema_21_ratio'] = df['close'] / df['ema_21']

            # Volume indicators
            df['volume_ema'] = talib.EMA(df['volume'].values, timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_ema']

            # Add momentum indicators
            df['mom'] = talib.MOM(df['close'].values, timeperiod=10)
            df['willr'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)

            # Add Ichimoku Cloud components
            df['tenkan_sen'] = talib.SMA(
                (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2,
                timeperiod=9
            )
            df['kijun_sen'] = talib.SMA(
                (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2,
                timeperiod=26
            )

            # Fibonacci-based price levels (using recent swing high/low)
            window = 14
            df['swing_high'] = df['high'].rolling(window).max()
            df['swing_low'] = df['low'].rolling(window).min()
            df['fib_range'] = df['swing_high'] - df['swing_low']
            df['fib_0.236'] = df['swing_low'] + 0.236 * df['fib_range']
            df['fib_0.382'] = df['swing_low'] + 0.382 * df['fib_range']
            df['fib_0.618'] = df['swing_low'] + 0.618 * df['fib_range']

            # Calculate price changes (percentage)
            df['pct_change'] = df['close'].pct_change()
            df['pct_change_abs'] = df['pct_change'].abs()

            # Add volatility indicators
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['natr'] = talib.NATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)

            # Create target variable (future price changes for multiple steps ahead)
            for i in range(1, self.target_steps + 1):
                df[f'future_return_{i}'] = df['close'].pct_change(i).shift(-i)

            # Drop rows with NaN values that result from calculation of indicators
            df.dropna(inplace=True)

            # Select features (technical indicators) and target
            feature_cols = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'price_ema_9_ratio', 'price_ema_21_ratio',
                'volume_ratio', 'mom', 'willr',
                'pct_change', 'pct_change_abs', 'atr', 'natr',
                'tenkan_sen', 'kijun_sen',
                'fib_0.236', 'fib_0.382', 'fib_0.618'
            ]

            # Ensure we have exactly the number of features we specified
            if len(feature_cols) != self.features:
                logging.warning(f"Expected {self.features} features, got {len(feature_cols)}. Adjusting feature count.")
                self.features = len(feature_cols)

            # Normalize features (simple min-max scaling)
            X = df[feature_cols].values
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_scaled = (X - X_min) / (X_max - X_min + 1e-10)  # Add epsilon to avoid division by zero

            # Target variables (future returns)
            y = df[[f'future_return_{i}' for i in range(1, self.target_steps + 1)]].values

            # Create sequences for LSTM
            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - self.look_back):
                X_seq.append(X_scaled[i:i + self.look_back])
                y_seq.append(y[i + self.look_back - 1])

            return np.array(X_seq), np.array(y_seq)

        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            if self.debug_mode:
                print(f"⚠️ Error preprocessing data: {str(e)}")
                traceback.print_exc()

            # Return dummy data in case of error
            dummy_X = np.random.random((10, self.look_back, self.features))
            dummy_y = np.random.random((10, self.target_steps))

            return dummy_X, dummy_y

    def predict(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Make predictions using the trained model for the given timeframe."""
        logging.info(f"Making prediction for {self.symbol} on {timeframe}m timeframe")

        try:
            # Handle empty or insufficient data
            if data is None or data.empty or len(data) < self.look_back + 10:
                logging.warning(f"Insufficient data for prediction on {timeframe}m timeframe")
                return {
                    "status": "error",
                    "timeframe": timeframe,
                    "message": "Insufficient data for prediction",
                    "action": "hold",  # Default to hold on errors
                    "confidence": 0.5,
                    "signal_strength": 0.0,
                    "timestamp": datetime.datetime.now().isoformat()
                }

            # Debug fallback mode - generate synthetic prediction if model isn't available
            if self.debug_mode or timeframe not in self.models:
                logging.info(f"Using simplified prediction for {timeframe}m timeframe (debug mode)")

                # Calculate simple momentum indicators
                last_close = data['close'].iloc[-1]
                prev_close = data['close'].iloc[-2]
                pct_change = (last_close - prev_close) / prev_close

                # Generate a simple directional signal based on recent price movement
                signal_strength = np.tanh(pct_change * 100)  # Scale and bound to -1 to 1

                # Add some randomness for testing
                confidence = 0.5 + abs(signal_strength) * 0.3 + np.random.uniform(0, 0.2)
                confidence = min(confidence, 0.95)  # Cap at 0.95

                # Create synthetic prediction
                y_pred = [pct_change * (i + 1) for i in range(self.target_steps)]

                # Determine action
                if signal_strength > 0.01:
                    action = "buy"
                elif signal_strength < -0.01:
                    action = "sell"
                else:
                    action = "hold"

                print(f"🔄 Debug prediction on {timeframe}m: {action.upper()} (signal: {signal_strength:.4f})")

                return {
                    "status": "success",
                    "timeframe": timeframe,
                    "prediction": y_pred,
                    "signal_strength": float(signal_strength),
                    "confidence": float(confidence),
                    "action": action,
                    "is_debug": True,
                    "timestamp": datetime.datetime.now().isoformat()
                }

            # Regular prediction using trained model
            # Preprocess data
            X, _ = self.preprocess_data(data)

            # Take the most recent sequence for prediction
            X_pred = X[-1].reshape(1, self.look_back, self.features)

            # Make prediction
            y_pred = self.models[timeframe].predict(X_pred, verbose=0)[0]

            # Calculate prediction confidence based on historical accuracy
            # In a real system, this would be based on backtest performance
            if len(self.accuracy_metrics[timeframe]["rmse"]) > 0:
                # Use inverse of recent RMSE to estimate confidence
                recent_rmse = np.mean(self.accuracy_metrics[timeframe]["rmse"][-10:])
                # Scale to a reasonable confidence value (0.5-0.95)
                confidence = max(0.5, min(0.95, 1.0 - recent_rmse))
            else:
                confidence = 0.75  # Default when no accuracy metrics available

            # Calculate signal strength (-1 to 1, where -1 is strong sell, 1 is strong buy)
            # Weight predictions by their time horizon (closer predictions matter more)
            weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])[:len(y_pred)]
            signal_strength = np.average(y_pred, weights=weights)

            # Adjust confidence based on market volatility
            if 'atr' in data.columns:
                volatility = data['atr'].iloc[-1] / data['close'].iloc[-1]
                # Lower confidence in highly volatile markets
                if volatility > 0.02:  # 2% ATR/price ratio is high
                    confidence *= 0.9

            # Determine action based on signal strength and confidence
            threshold = 0.005  # Base threshold
            if confidence > 0.8:
                # More confident predictions can use a lower threshold
                threshold *= 0.8

            if signal_strength > threshold:
                action = "buy"
            elif signal_strength < -threshold:
                action = "sell"
            else:
                action = "hold"

            # Store prediction for tracking
            self.prediction_history.append({
                "timeframe": timeframe,
                "timestamp": datetime.datetime.now().isoformat(),
                "prediction": y_pred.tolist(),
                "signal_strength": float(signal_strength),
                "confidence": float(confidence),
                "action": action
            })

            # Keep history manageable
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            return {
                "status": "success",
                "timeframe": timeframe,
                "prediction": y_pred.tolist(),
                "signal_strength": float(signal_strength),
                "confidence": float(confidence),
                "action": action,
                "timestamp": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Prediction error for {timeframe}m timeframe: {str(e)}")
            print(f"⚠️ Prediction error for {timeframe}m timeframe: {str(e)}")

            # Return fallback prediction
            return {
                "status": "error",
                "timeframe": timeframe,
                "message": str(e),
                "action": "hold",  # Default to hold on errors
                "confidence": 0.5,
                "signal_strength": 0.0,
                "timestamp": datetime.datetime.now().isoformat()
            }

    def multi_timeframe_prediction(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Perform predictions across multiple timeframes and aggregate the results.

        Args:
            data_dict: Dictionary mapping timeframes to their respective DataFrames

        Returns:
            Dictionary with aggregated prediction results
        """
        predictions = {}
        confidence_weighted_signals = []
        total_confidence = 0

        # Get predictions for each timeframe
        for tf in self.timeframes:
            if tf in data_dict and not data_dict[tf].empty:
                pred = self.predict(data_dict[tf], tf)
                predictions[tf] = pred

                if pred["status"] == "success":
                    # Weight signal by confidence and timeframe importance
                    # Higher timeframes get more weight
                    tf_weight = float(tf) / 10  # Example: 1min=0.1, 60min=6.0
                    weighted_confidence = pred["confidence"] * tf_weight
                    confidence_weighted_signals.append(
                        pred["signal_strength"] * weighted_confidence
                    )
                    total_confidence += weighted_confidence
            else:
                logging.warning(f"No data available for {tf}m timeframe prediction")

        # Calculate aggregated signal
        if total_confidence > 0:
            aggregated_signal = sum(confidence_weighted_signals) / total_confidence
        else:
            aggregated_signal = 0

        # Determine overall action
        if aggregated_signal > 0.01:
            overall_action = "buy"
            overall_confidence = min(abs(aggregated_signal) * 5, 0.95)  # Cap confidence at 95%
        elif aggregated_signal < -0.01:
            overall_action = "sell"
            overall_confidence = min(abs(aggregated_signal) * 5, 0.95)
        else:
            overall_action = "hold"
            overall_confidence = 0.5

        return {
            "individual_predictions": predictions,
            "aggregated_signal": float(aggregated_signal),
            "overall_action": overall_action,
            "overall_confidence": float(overall_confidence),
            "timestamp": datetime.datetime.now().isoformat()
        }


# ------------------------------------------------------------------------------
# Historical and Real-time Market Data Fetching
# ------------------------------------------------------------------------------
class MarketDataManager:
    """Manages fetching, processing, and storing market data."""

    def __init__(self, api_key: str, api_secret: str, symbol: str, timeframes: List[str], api_url: str = None):
        """Initialize the market data manager."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.timeframes = timeframes
        self.api_url = api_url or "https://api.bybit.com"

        # Initialize BYBIT REST API client with version compatibility handling
        self.bybit_client = None
        if PYBIT_AVAILABLE:
            try:
                # Try new style constructor (newer pybit versions)
                self.bybit_client = HTTP(
                    testnet=False,  # Use mainnet
                    api_key=api_key,
                    api_secret=api_secret
                )
                print("Using standard pybit constructor")
            except TypeError:
                # Fall back to old style constructor
                self.bybit_client = HTTP(
                    endpoint=self.api_url,
                    api_key=api_key,
                    api_secret=api_secret
                )
                print("Using legacy pybit constructor")
        else:
            print("Using fallback HTTP requests for API calls")

        # Dictionary to store DataFrames for each timeframe
        self.data = {tf: pd.DataFrame() for tf in timeframes}

        # Dictionary for real-time price data
        self.latest_prices = {
            "bid": 0.0,
            "ask": 0.0,
            "last": 0.0,
            "timestamp": None
        }

        # Dictionary to store order book data
        self.order_book = {
            "bids": [],
            "asks": [],
            "timestamp": None
        }

        # Dictionary for tracking data update status
        self.last_update = {tf: None for tf in timeframes}

    def fetch_historical_klines(self, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical kline (candlestick) data for a specific timeframe.

        Args:
            timeframe: Time interval in minutes (e.g., "1", "5", "15", "60")
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        interval_map = {
            "1": "1",
            "5": "5",
            "15": "15",
            "60": "60",
            "240": "240",
            "D": "D"
        }

        if timeframe not in interval_map:
            logging.error(f"Unsupported timeframe: {timeframe}")
            return pd.DataFrame()

        interval = interval_map[timeframe]

        try:
            if self.bybit_client:
                # Use pybit client if available
                response = self.bybit_client.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval=interval,
                    limit=limit
                )

                if response["retCode"] != 0:
                    logging.error(f"Failed to fetch klines: {response['retMsg']}")
                    return pd.DataFrame()

                # Extract kline data
                klines = response["result"]["list"]
            else:
                # Fallback to direct API requests
                url = f"{self.api_url}/v5/market/kline"
                params = {
                    "category": "linear",
                    "symbol": self.symbol,
                    "interval": interval,
                    "limit": limit
                }

                response = requests.get(url, params=params)
                data = response.json()

                if data["retCode"] != 0:
                    logging.error(f"Failed to fetch klines: {data['retMsg']}")
                    return pd.DataFrame()

                klines = data["result"]["list"]

            if not klines:
                logging.warning(f"No kline data received for {self.symbol} {timeframe}m")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            # Convert types
            numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])

            # Convert timestamp from milliseconds to datetime - fixed for pandas warning
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")

            # Sort by timestamp (oldest first)
            df = df.sort_values("timestamp")

            # Set timestamp as index
            df.set_index("timestamp", inplace=True)

            logging.info(f"Fetched {len(df)} klines for {self.symbol} {timeframe}m")
            self.last_update[timeframe] = datetime.datetime.now()

            # Store in data dictionary
            self.data[timeframe] = df

            return df

        except Exception as e:
            logging.error(f"Error fetching historical klines: {str(e)}")
            return pd.DataFrame()

    def fetch_latest_ticker(self) -> Dict:
        """Fetch latest ticker data for the trading pair."""
        try:
            if self.bybit_client:
                # Use pybit client if available
                response = self.bybit_client.get_tickers(
                    category="linear",
                    symbol=self.symbol
                )

                if response["retCode"] != 0:
                    logging.error(f"Failed to fetch ticker: {response['retMsg']}")
                    return {}

                ticker = response["result"]["list"][0]
            else:
                # Fallback to direct API requests
                url = f"{self.api_url}/v5/market/tickers"
                params = {
                    "category": "linear",
                    "symbol": self.symbol
                }

                response = requests.get(url, params=params)
                data = response.json()

                if data["retCode"] != 0:
                    logging.error(f"Failed to fetch ticker: {data['retMsg']}")
                    return {}

                ticker = data["result"]["list"][0]

            # Update latest prices
            self.latest_prices = {
                "bid": float(ticker["bid1Price"]),
                "ask": float(ticker["ask1Price"]),
                "last": float(ticker["lastPrice"]),
                "timestamp": datetime.datetime.now()
            }

            return self.latest_prices

        except Exception as e:
            logging.error(f"Error fetching latest ticker: {str(e)}")
            return {}

    def update_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        """Update data for all timeframes."""
        for tf in self.timeframes:
            self.fetch_historical_klines(tf)

        return self.data

    def get_market_data(self, timeframe: str) -> pd.DataFrame:
        """Get the latest market data for a specific timeframe."""
        # Check if we need to update the data
        current_time = datetime.datetime.now()
        if (self.last_update[timeframe] is None or
                (current_time - self.last_update[timeframe]).total_seconds() > int(timeframe) * 60):
            self.fetch_historical_klines(timeframe)

        return self.data[timeframe]

    def get_all_market_data(self) -> Dict[str, pd.DataFrame]:
        """Get the latest market data for all timeframes."""
        return self.data

    def get_latest_price(self) -> float:
        """Get the latest mid-price."""
        if self.latest_prices["timestamp"] is None:
            self.fetch_latest_ticker()

        if self.latest_prices["bid"] == 0 or self.latest_prices["ask"] == 0:
            return float(self.latest_prices["last"])

        return (self.latest_prices["bid"] + self.latest_prices["ask"]) / 2

    def get_bid_ask_spread(self) -> float:
        """Get the current bid-ask spread."""
        if self.latest_prices["timestamp"] is None:
            self.fetch_latest_ticker()

        return self.latest_prices["ask"] - self.latest_prices["bid"]


# ------------------------------------------------------------------------------
# Technical Indicator Computation
# ------------------------------------------------------------------------------
class TechnicalAnalysis:
    """Computes and analyzes technical indicators for market analysis."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        # Skip if DataFrame is empty or has insufficient data
        if df is None or df.empty or len(df) < 30:
            return df

        df_copy = df.copy()

        # Ensure we have the necessary columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df_copy.columns for col in required_cols):
            logging.error("Missing required columns for technical analysis")
            return df_copy

        try:
            # RSI
            df_copy['rsi'] = talib.RSI(df_copy['close'].values, timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df_copy['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df_copy['macd'] = macd
            df_copy['macd_signal'] = macd_signal
            df_copy['macd_hist'] = macd_hist

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                df_copy['close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            df_copy['bb_upper'] = upper
            df_copy['bb_middle'] = middle
            df_copy['bb_lower'] = lower
            df_copy['bb_width'] = (upper - lower) / middle

            # EMAs
            df_copy['ema_9'] = talib.EMA(df_copy['close'].values, timeperiod=9)
            df_copy['ema_21'] = talib.EMA(df_copy['close'].values, timeperiod=21)
            df_copy['ema_50'] = talib.EMA(df_copy['close'].values, timeperiod=50)
            df_copy['ema_200'] = talib.EMA(df_copy['close'].values, timeperiod=200)

            # Stochastic
            slowk, slowd = talib.STOCH(
                df_copy['high'].values,
                df_copy['low'].values,
                df_copy['close'].values,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            df_copy['stoch_k'] = slowk
            df_copy['stoch_d'] = slowd

            # Average True Range (ATR)
            df_copy['atr'] = talib.ATR(
                df_copy['high'].values,
                df_copy['low'].values,
                df_copy['close'].values,
                timeperiod=14
            )

            # On-Balance Volume (OBV)
            df_copy['obv'] = talib.OBV(df_copy['close'].values, df_copy['volume'].values)

            # Momentum
            df_copy['mom'] = talib.MOM(df_copy['close'].values, timeperiod=10)

            # Rate of Change (ROC)
            df_copy['roc'] = talib.ROC(df_copy['close'].values, timeperiod=10)

            # Williams %R
            df_copy['willr'] = talib.WILLR(
                df_copy['high'].values,
                df_copy['low'].values,
                df_copy['close'].values,
                timeperiod=14
            )

            # Commodity Channel Index (CCI)
            df_copy['cci'] = talib.CCI(
                df_copy['high'].values,
                df_copy['low'].values,
                df_copy['close'].values,
                timeperiod=14
            )

            return df_copy

        except Exception as e:
            logging.error(f"Error calculating technical indicators: {str(e)}")
            return df_copy


# ------------------------------------------------------------------------------
# Trading Strategy Implementation
# ------------------------------------------------------------------------------
class TradingStrategy:
    """Implements AI-driven futures trading strategies."""

    def __init__(self, symbol: str, timeframes: List[str], max_risk_per_trade: float = 0.05):
        """Initialize the trading strategy."""
        self.symbol = symbol
        self.timeframes = timeframes
        self.max_risk_per_trade = max_risk_per_trade

        # Initialize technical analysis
        self.ta = TechnicalAnalysis()

        # Strategy parameters
        self.entry_threshold = 0.6  # Confidence threshold for entry
        self.exit_threshold = 0.4  # Confidence threshold for exit

        # Strategy state
        self.current_position = None
        self.current_position_data = {}

        # Strategy performance tracking
        self.trades = []
        self.performance_metrics = {
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }

        print(f"Trading strategy initialized for {symbol}")

    def analyze_market(self, data_dict: Dict[str, pd.DataFrame],
                       ai_prediction: Dict) -> Dict:
        """
        Analyze market conditions using technical indicators and AI predictions.

        Args:
            data_dict: Dictionary mapping timeframes to their respective DataFrames
            ai_prediction: Dictionary with AI model predictions

        Returns:
            Dictionary with market analysis results
        """
        analysis_results = {}

        # Analyze each timeframe
        for tf in self.timeframes:
            if tf in data_dict and not data_dict[tf].empty:
                try:
                    # Calculate technical indicators
                    df_with_indicators = self.ta.calculate_indicators(data_dict[tf])

                    # Analyze trend
                    trend_analysis = self.analyze_trend(df_with_indicators)

                    # Store analysis results for this timeframe
                    analysis_results[tf] = {
                        "trend_analysis": trend_analysis,
                        "last_close": float(df_with_indicators['close'].iloc[-1]),
                        "last_volume": float(df_with_indicators['volume'].iloc[-1]),
                        "last_timestamp": df_with_indicators.index[-1].isoformat()
                    }
                except Exception as e:
                    logging.error(f"Error analyzing {tf}m timeframe: {str(e)}")

        # Combine technical analysis with AI predictions
        combined_analysis = {
            "timeframe_analysis": analysis_results,
            "ai_prediction": ai_prediction,
            "timestamp": datetime.datetime.now().isoformat()
        }

        return combined_analysis

    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the current market trend using technical indicators.

        Args:
            df: DataFrame with technical indicators

        Returns:
            Dictionary with trend analysis results
        """
        # Handle empty dataframe
        if df is None or df.empty:
            return {
                "trend_direction": "neutral",
                "trend_strength": 0,
                "trend_score": 0,
                "ema_trend": 0,
                "rsi_signal": 0,
                "macd_signal": 0,
                "bb_signal": 0,
                "volume_signal": 0
            }

        # Get the most recent data point
        latest = df.iloc[-1]

        # Analyze trend using EMAs
        ema_trend = 0
        if all(col in df.columns for col in ['ema_50', 'ema_9', 'ema_21']):
            if latest['close'] > latest['ema_50'] and latest['ema_9'] > latest['ema_21']:
                ema_trend = 1  # Bullish
            elif latest['close'] < latest['ema_50'] and latest['ema_9'] < latest['ema_21']:
                ema_trend = -1  # Bearish

        # Analyze RSI
        rsi_signal = 0
        if 'rsi' in df.columns:
            if latest['rsi'] > 70:
                rsi_signal = -1  # Overbought
            elif latest['rsi'] < 30:
                rsi_signal = 1  # Oversold

        # Analyze MACD
        macd_signal = 0
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
                macd_signal = 1  # Bullish
            elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
                macd_signal = -1  # Bearish

        # Analyze Bollinger Bands
        bb_signal = 0
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            if latest['close'] > latest['bb_upper']:
                bb_signal = -1  # Overbought
            elif latest['close'] < latest['bb_lower']:
                bb_signal = 1  # Oversold

        # Volume analysis
        volume_signal = 0
        if 'volume' in df.columns:
            vol_avg = df['volume'].rolling(20).mean().iloc[-1]
            if latest['volume'] > vol_avg * 1.5:
                # Significant volume increase
                if latest['close'] > latest['open']:
                    volume_signal = 1  # Bullish volume
                else:
                    volume_signal = -1  # Bearish volume

        # Calculate overall trend score
        signals = [ema_trend, rsi_signal, macd_signal, bb_signal, volume_signal]
        valid_signals = [s for s in signals if s != 0]
        trend_score = sum(valid_signals) / len(valid_signals) if valid_signals else 0

        # Determine trend strength
        trend_strength = abs(trend_score)

        # Determine trend direction
        if trend_score > 0.3:
            trend_direction = "bullish"
        elif trend_score < -0.3:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        return {
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "trend_score": float(trend_score),
            "ema_trend": ema_trend,
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "bb_signal": bb_signal,
            "volume_signal": volume_signal
        }

    def generate_trade_signals(self, market_analysis: Dict) -> Dict:
        """
        Generate trade signals based on market analysis and AI predictions.

        Args:
            market_analysis: Dictionary with market analysis results

        Returns:
            Dictionary with trade signals
        """
        ai_prediction = market_analysis["ai_prediction"]
        timeframe_analysis = market_analysis["timeframe_analysis"]

        # Extract overall AI prediction
        ai_action = ai_prediction.get("overall_action", "hold")
        ai_confidence = ai_prediction.get("overall_confidence", 0.5)

        # Initialize signals
        signals = {
            "entry_signal": None,
            "exit_signal": None,
            "position_sizing": 0.0,
            "take_profit_pct": 0.0,
            "stop_loss_pct": 0.0,
            "confidence": 0.0,
            "reasoning": []
        }

        # Check if we have a current position
        if self.current_position:
            # Handle exit conditions
            if ai_action == "sell" and ai_confidence > self.exit_threshold and self.current_position == "long":
                signals["exit_signal"] = "close_long"
                signals["confidence"] = ai_confidence
                signals["reasoning"].append(f"AI predicts sell with {ai_confidence:.2%} confidence")

            elif ai_action == "buy" and ai_confidence > self.exit_threshold and self.current_position == "short":
                signals["exit_signal"] = "close_short"
                signals["confidence"] = ai_confidence
                signals["reasoning"].append(f"AI predicts buy with {ai_confidence:.2%} confidence")

        else:
            # No current position, evaluate entry conditions
            if ai_action == "buy" and ai_confidence > self.entry_threshold:
                signals["entry_signal"] = "long"
                signals["confidence"] = ai_confidence
                signals["reasoning"].append(f"AI predicts buy with {ai_confidence:.2%} confidence")

                # Calculate dynamic position sizing based on confidence
                signals["position_sizing"] = min(ai_confidence * self.max_risk_per_trade, self.max_risk_per_trade)

                # Set dynamic take profit and stop loss
                signals["take_profit_pct"] = 0.03  # 3% take profit
                signals["stop_loss_pct"] = 0.01  # 1% stop loss

            elif ai_action == "sell" and ai_confidence > self.entry_threshold:
                signals["entry_signal"] = "short"
                signals["confidence"] = ai_confidence
                signals["reasoning"].append(f"AI predicts sell with {ai_confidence:.2%} confidence")

                # Calculate dynamic position sizing based on confidence
                signals["position_sizing"] = min(ai_confidence * self.max_risk_per_trade, self.max_risk_per_trade)

                # Set dynamic take profit and stop loss
                signals["take_profit_pct"] = 0.03  # 3% take profit
                signals["stop_loss_pct"] = 0.01  # 1% stop loss

        return signals


# ------------------------------------------------------------------------------
# Trade Execution and Order Management
# ------------------------------------------------------------------------------
class TradeExecutor:
    """Handles the execution of trades on the BYBIT exchange."""

    def __init__(self, api_key: str, api_secret: str, symbol: str,
                 leverage: int = 3, trading_mode: str = "real", api_url: str = None):
        """Initialize the trade executor."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.leverage = leverage
        self.trading_mode = trading_mode
        self.api_url = api_url or "https://api.bybit.com"

        # Initialize BYBIT client with compatibility handling
        self.bybit_client = None
        if PYBIT_AVAILABLE and api_key and api_secret:
            try:
                # Try new style constructor (newer pybit versions)
                self.bybit_client = HTTP(
                    testnet=False,  # Use mainnet
                    api_key=api_key,
                    api_secret=api_secret
                )
                print("Using standard pybit constructor")
            except TypeError:
                # Fall back to old style constructor
                self.bybit_client = HTTP(
                    endpoint=self.api_url,
                    api_key=api_key,
                    api_secret=api_secret
                )
                print("Using legacy pybit constructor")
        else:
            print("No pybit client available or missing API credentials - using simulation mode only")

        # Set initial leverage if in real trading mode
        if self.trading_mode == "real" and self.bybit_client:
            self.set_leverage(leverage)

        # Order execution status tracking
        self.last_order_result = None
        self.open_positions = []
        self.order_history = []

        print(f"Trade executor initialized for {symbol} in {trading_mode} mode")

    def set_leverage(self, leverage: int) -> Dict:
        """Set leverage for the trading pair."""
        if self.trading_mode != "real" or not self.bybit_client:
            return {"success": True, "message": "Leverage setting simulated"}

        try:
            response = self.bybit_client.set_leverage(
                category="linear",
                symbol=self.symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )

            if response["retCode"] == 0:
                self.leverage = leverage
                logging.info(f"Leverage set to {leverage}x for {self.symbol}")
                return {"success": True, "message": f"Leverage set to {leverage}x"}
            else:
                logging.error(f"Failed to set leverage: {response['retMsg']}")
                return {"success": False, "message": f"Failed to set leverage: {response['retMsg']}"}

        except Exception as e:
            logging.error(f"Error setting leverage: {str(e)}")
            return {"success": False, "message": f"Error setting leverage: {str(e)}"}

    def get_account_balance(self) -> Dict:
        """Get the current account balance."""
        if self.trading_mode != "real" or not self.bybit_client:
            # Return simulated balance for testing
            return {
                "usdt_balance": 10000.0,
                "unrealized_pnl": 0.0,
                "available_balance": 10000.0,
                "used_margin": 0.0
            }

        try:
            # Replace this in your account balance request
            response = self.bybit_client.get_wallet_balance(accountType="UNIFIED", coin="USDT")


            if response["retCode"] != 0:
                logging.error(f"Failed to get account balance: {response['retMsg']}")
                return {
                    "usdt_balance": 0.0,
                    "unrealized_pnl": 0.0,
                    "available_balance": 0.0,
                    "used_margin": 0.0
                }

            # Extract USDT balance
            coin_data = response["result"]["list"][0]["coin"]
            usdt_data = next((item for item in coin_data if item["coin"] == "USDT"), None)

            if not usdt_data:
                logging.error("USDT balance not found in account")
                return {
                    "usdt_balance": 0.0,
                    "unrealized_pnl": 0.0,
                    "available_balance": 0.0,
                    "used_margin": 0.0
                }

            return {
                "usdt_balance": float(usdt_data["walletBalance"]),
                "unrealized_pnl": float(usdt_data["unrealisedPnl"]),
                "available_balance": float(usdt_data["availableBalance"]),
                "used_margin": float(usdt_data["usedMargin"])
            }

        except Exception as e:
            logging.error(f"Error getting account balance: {str(e)}")
            return {
                "usdt_balance": 0.0,
                "unrealized_pnl": 0.0,
                "available_balance": 0.0,
                "used_margin": 0.0
            }

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        if self.trading_mode != "real" or not self.bybit_client:
            return self.open_positions  # Return simulated positions

        try:
            response = self.bybit_client.get_positions(
                category="linear",
                symbol=self.symbol
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to get positions: {response['retMsg']}")
                return []

            positions = response["result"]["list"]
            result = []

            for pos in positions:
                if float(pos["size"]) > 0:
                    result.append({
                        "symbol": pos["symbol"],
                        "side": "Buy" if pos["side"] == "Buy" else "Sell",
                        "size": float(pos["size"]),
                        "entry_price": float(pos["entryPrice"]),
                        "current_price": float(pos["markPrice"]),
                        "unrealized_pnl": float(pos["unrealisedPnl"]),
                        "pnl_percent": float(pos["unrealisedPnl"]) / (float(pos["positionValue"]) or 1),
                        "leverage": float(pos["leverage"])
                    })

            # Update local cache
            self.open_positions = result
            return result

        except Exception as e:
            logging.error(f"Error getting open positions: {str(e)}")
            return []

    def place_market_order(self, side: str, quantity: float,
                           stop_loss_price: float = None,
                           take_profit_price: float = None) -> Dict:
        """
        Place a market order with optional stop loss and take profit.

        Args:
            side: Order side ("Buy" or "Sell")
            quantity: Order quantity
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price

        Returns:
            Dictionary with order result
        """
        if self.trading_mode != "real" or not self.bybit_client:
            logging.info(f"SIMULATION: {side} market order for {quantity} {self.symbol}")

            # Add to simulated positions in simulation mode
            if side == "Buy":
                self.open_positions.append({
                    "symbol": self.symbol,
                    "side": "Buy",
                    "size": quantity,
                    "entry_price": 0,  # Would need current price here
                    "current_price": 0,
                    "unrealized_pnl": 0,
                    "pnl_percent": 0,
                    "leverage": self.leverage
                })

            return {
                "success": True,
                "message": f"SIMULATION: {side} order placed successfully",
                "order_id": f"sim-{int(time.time())}",
                "stop_loss_order_id": f"sim-sl-{int(time.time())}" if stop_loss_price else None,
                "take_profit_order_id": f"sim-tp-{int(time.time())}" if take_profit_price else None
            }

        try:
            # Round quantity to appropriate precision
            quantity = round(quantity, 6)

            logging.info(f"Placing {side} market order for {quantity} {self.symbol}")
            response = self.bybit_client.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(quantity),
                reduceOnly=False,
                closeOnTrigger=False
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to place market order: {response['retMsg']}")
                return {
                    "success": False,
                    "message": f"Failed to place market order: {response['retMsg']}",
                    "order_id": None
                }

            order_id = response["result"]["orderId"]
            logging.info(f"Market order placed successfully. Order ID: {order_id}")

            # Place stop loss if provided
            sl_order_id = None
            if stop_loss_price:
                sl_side = "Sell" if side == "Buy" else "Buy"
                sl_response = self.bybit_client.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side=sl_side,
                    orderType="Market",
                    triggerPrice=str(stop_loss_price),
                    qty=str(quantity),
                    triggerBy="LastPrice",
                    reduceOnly=True,
                    closeOnTrigger=True,
                    timeInForce="GoodTillCancel",
                    positionIdx=0
                )

                if sl_response["retCode"] == 0:
                    sl_order_id = sl_response["result"]["orderId"]
                    logging.info(f"Stop loss order placed at {stop_loss_price}. Order ID: {sl_order_id}")
                else:
                    logging.error(f"Failed to place stop loss order: {sl_response['retMsg']}")

            # Place take profit if provided
            tp_order_id = None
            if take_profit_price:
                tp_side = "Sell" if side == "Buy" else "Buy"
                tp_response = self.bybit_client.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side=tp_side,
                    orderType="Market",
                    triggerPrice=str(take_profit_price),
                    qty=str(quantity),
                    triggerBy="LastPrice",
                    reduceOnly=True,
                    closeOnTrigger=True,
                    timeInForce="GoodTillCancel",
                    positionIdx=0
                )

                if tp_response["retCode"] == 0:
                    tp_order_id = tp_response["result"]["orderId"]
                    logging.info(f"Take profit order placed at {take_profit_price}. Order ID: {tp_order_id}")
                else:
                    logging.error(f"Failed to place take profit order: {tp_response['retMsg']}")

            self.last_order_result = {
                "success": True,
                "message": "Market order placed successfully",
                "order_id": order_id,
                "stop_loss_order_id": sl_order_id,
                "take_profit_order_id": tp_order_id
            }

            return self.last_order_result

        except Exception as e:
            error_message = f"Error placing market order: {str(e)}"
            logging.error(error_message)

            self.last_order_result = {
                "success": False,
                "message": error_message,
                "order_id": None
            }

            return self.last_order_result


# ------------------------------------------------------------------------------
# Visualization and Charting
# ------------------------------------------------------------------------------
class Visualizer:
    """Creates and manages visualizations and reports."""

    def __init__(self, symbol: str, chart_dir: str = "charts"):
        """Initialize the visualizer."""
        self.symbol = symbol
        self.chart_dir = chart_dir

        # Create chart directory if it doesn't exist
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)

    def generate_price_chart(self, data: pd.DataFrame,
                             indicators: bool = True,
                             trades: List[Dict] = None) -> str:
        """
        Generate a price chart with optional indicators and trade markers.

        Args:
            data: DataFrame with OHLCV data
            indicators: Whether to include technical indicators
            trades: List of trade data for marking entries and exits

        Returns:
            Path to the generated chart image
        """
        # Validate input data
        if data is None or data.empty:
            logging.error("Cannot generate chart: Empty dataframe provided")
            return ""

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_cols):
            logging.error("Cannot generate chart: Missing required columns")
            return ""

        try:
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
            fig.suptitle(f"{self.symbol} - {data.index[-1].strftime('%Y-%m-%d %H:%M')}", fontsize=16)

            # Plot price candlesticks - using a different approach than the original code
            ax_price = axs[0]

            # Create arrays for easier plotting
            dates = np.arange(len(data))
            opens = data['open'].values
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values

            # Color logic for candlesticks
            colors = np.where(closes > opens, 'green', 'red')

            # Plot candlesticks
            for i in range(len(data)):
                # Plot body
                ax_price.plot([dates[i], dates[i]], [opens[i], closes[i]],
                              color=colors[i], linewidth=5, solid_capstyle='butt')
                # Plot wicks
                ax_price.plot([dates[i], dates[i]], [lows[i], highs[i]],
                              color=colors[i], linewidth=1)

            # Plot indicators if requested
            if indicators:
                # Calculate indicators if they don't exist
                if 'ema_9' not in data.columns:
                    data.loc[:, 'ema_9'] = talib.EMA(data['close'].values, timeperiod=9)

                if 'ema_21' not in data.columns:
                    data.loc[:, 'ema_21'] = talib.EMA(data['close'].values, timeperiod=21)

                if 'ema_50' not in data.columns:
                    data.loc[:, 'ema_50'] = talib.EMA(data['close'].values, timeperiod=50)

                # Plot EMAs
                ax_price.plot(dates, data['ema_9'].values, color='blue', linestyle='-', linewidth=1, label='EMA 9')
                ax_price.plot(dates, data['ema_21'].values, color='orange', linestyle='-', linewidth=1, label='EMA 21')
                ax_price.plot(dates, data['ema_50'].values, color='purple', linestyle='-', linewidth=1.5,
                              label='EMA 50')

            # Set price chart properties
            ax_price.set_ylabel('Price')
            ax_price.legend(loc='upper left')
            ax_price.grid(True, alpha=0.3)

            # Plot volume
            ax_volume = axs[1]
            volume_colors = np.where(data['close'].values >= data['open'].values, 'green', 'red')
            ax_volume.bar(dates, data['volume'].values, color=volume_colors, alpha=0.7)
            ax_volume.set_ylabel('Volume')
            ax_volume.grid(True, alpha=0.3)

            # Plot RSI
            ax_rsi = axs[2]
            if 'rsi' not in data.columns:
                data.loc[:, 'rsi'] = talib.RSI(data['close'].values, timeperiod=14)

            ax_rsi.plot(dates, data['rsi'].values, color='purple', linewidth=1.5)
            ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3)

            # Set common x-axis ticks - use a sample of dates for readability
            tick_locations = np.linspace(0, len(data) - 1, 10, dtype=int)
            for ax in axs:
                ax.set_xticks(tick_locations)
                ax.set_xticklabels([data.index[i].strftime('%m-%d %H:%M') for i in tick_locations], rotation=45)

            # Adjust layout
            plt.tight_layout()

            # Save to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(self.chart_dir, f"{self.symbol}_chart_{timestamp}.png")
            plt.savefig(chart_path, dpi=150)
            plt.close(fig)

            logging.info(f"Chart saved to {chart_path}")
            return chart_path

        except Exception as e:
            logging.error(f"Error generating price chart: {str(e)}")
            traceback.print_exc()
            return ""

    def generate_performance_chart(self, trades: List[Dict], account_value: List[Dict]) -> str:
        """
        Generate a performance chart showing account equity and trade results.

        Args:
            trades: List of completed trades
            account_value: List of account value snapshots

        Returns:
            Path to the generated chart image
        """
        if not trades or not account_value:
            logging.warning("Cannot generate performance chart: No trade or account data")
            return ""

        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f"{self.symbol} - Trading Performance", fontsize=16)

            # Parse data for plotting
            dates = [datetime.datetime.fromisoformat(point['timestamp']) for point in account_value]
            equity = [point['equity'] for point in account_value]

            # Plot equity curve
            ax1.plot(dates, equity, 'b-', linewidth=2)
            ax1.set_ylabel('Account Value (USDT)')
            ax1.grid(True, alpha=0.3)

            # Format x-axis dates
            plt.gcf().autofmt_xdate()

            # Adjust layout
            plt.tight_layout()

            # Save to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(self.chart_dir, f"{self.symbol}_performance_{timestamp}.png")
            plt.savefig(chart_path, dpi=150)
            plt.close(fig)

            logging.info(f"Performance chart saved to {chart_path}")
            return chart_path

        except Exception as e:
            logging.error(f"Error generating performance chart: {str(e)}")
            traceback.print_exc()
            return ""


# ------------------------------------------------------------------------------
# Main Trading System
# ------------------------------------------------------------------------------
class TradingSystem:
    """Main trading system that integrates all components."""

    def __init__(self):
        """Initialize the trading system with configuration."""
        # Initialize logging
        self.log_file = setup_logging(debug_mode=DEBUG_MODE)

        # Set configuration directly (simplified for easy startup)
        self.config = {
            "SYMBOL": SYMBOL,
            "TIMEFRAMES": TIMEFRAMES,
            "LEVERAGE": LEVERAGE,
            "MAX_RISK_PER_TRADE": MAX_RISK_PER_TRADE,
            "TRADING_MODE": TRADING_MODE,
            "BYBIT_API_KEY": BYBIT_API_KEY,
            "BYBIT_API_SECRET": BYBIT_API_SECRET,
            "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
            "TELEGRAM_ENABLED": TELEGRAM_ENABLED,
            "DEBUG_MODE": DEBUG_MODE,
            "BYBIT_REST_API_URL": BYBIT_REST_API_URL
        }

        # Initialize the Bybit client
        from pybit.unified_trading import HTTP
        self.bybit_client = HTTP(
            testnet=False,
            api_key=self.config["BYBIT_API_KEY"],
            api_secret=self.config["BYBIT_API_SECRET"]
        )

        # Initialize Telegram notifications (if enabled)
        if self.config["TELEGRAM_ENABLED"]:
            try:
                self.telegram = TelegramNotifier(
                    self.config["TELEGRAM_BOT_TOKEN"],
                    self.config["TELEGRAM_CHAT_ID"]
                )
                self.telegram.trading_system = self  # Set reference to self
                logging.info("Telegram notifications initialized")
                print("✅ Telegram notifications initialized")
                self.telegram.send_message("🤖 Trading Bot starting up...")
            except Exception as e:
                logging.error(f"Failed to initialize Telegram: {str(e)}")
                print(f"⚠️ Failed to initialize Telegram: {str(e)}")
                self.telegram = None
        else:
            logging.info("Telegram notifications disabled")
            print("ℹ️ Telegram notifications disabled")
            self.telegram = None

        # Initialize components
        self.initialize_components()

        # System status tracking
        self.running = False
        self.start_time = None
        self.last_data_update = None
        self.last_prediction = None
        self.last_trade_signal = None
        self.last_trade_execution = None
        self.trading_enabled = True  # Flag to enable/disable trading

        # Performance tracking
        self.trades = []
        self.predictions = []
        self.account_history = []

        logging.info(f"Trading system initialized for {self.config['SYMBOL']}")
        print(f"✅ Trading system fully initialized for {self.config['SYMBOL']}")

    def initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize market data manager
            self.market_data = MarketDataManager(
                api_key=self.config.get("BYBIT_API_KEY", ""),
                api_secret=self.config.get("BYBIT_API_SECRET", ""),
                symbol=self.config["SYMBOL"],
                timeframes=self.config["TIMEFRAMES"],
                api_url=self.config.get("BYBIT_REST_API_URL")
            )
            print(f"✅ Market data manager initialized for {self.config['SYMBOL']}")
        except Exception as e:
            logging.error(f"Error initializing market data manager: {str(e)}")
            print(f"⚠️ Error initializing market data manager: {str(e)}")
            traceback.print_exc()
            raise

        try:
            # Initialize AI predictive model
            self.ai_model = PredictiveModel(
                symbol=self.config["SYMBOL"],
                timeframes=self.config["TIMEFRAMES"],
                debug_mode=self.config.get("DEBUG_MODE", False)
            )
            print(f"✅ AI model initialized")
        except Exception as e:
            logging.error(f"Error initializing AI model: {str(e)}")
            print(f"⚠️ Error initializing AI model: {str(e)}")
            traceback.print_exc()
            raise

        try:
            # Initialize trading strategy
            self.strategy = TradingStrategy(
                symbol=self.config["SYMBOL"],
                timeframes=self.config["TIMEFRAMES"],
                max_risk_per_trade=self.config["MAX_RISK_PER_TRADE"]
            )
            print(f"✅ Trading strategy initialized")
        except Exception as e:
            logging.error(f"Error initializing trading strategy: {str(e)}")
            print(f"⚠️ Error initializing trading strategy: {str(e)}")
            traceback.print_exc()
            raise

        try:
            # Initialize trade executor
            self.trade_executor = TradeExecutor(
                api_key=self.config.get("BYBIT_API_KEY", ""),
                api_secret=self.config.get("BYBIT_API_SECRET", ""),
                symbol=self.config["SYMBOL"],
                leverage=self.config["LEVERAGE"],
                trading_mode=self.config["TRADING_MODE"],
                api_url=self.config.get("BYBIT_REST_API_URL")
            )
            print(f"✅ Trade executor initialized")
        except Exception as e:
            logging.error(f"Error initializing trade executor: {str(e)}")
            print(f"⚠️ Error initializing trade executor: {str(e)}")
            traceback.print_exc()
            raise

        try:
            # Initialize visualizer
            self.visualizer = Visualizer(
                symbol=self.config["SYMBOL"]
            )
            print(f"✅ Visualizer initialized")
        except Exception as e:
            logging.error(f"Error initializing visualizer: {str(e)}")
            print(f"⚠️ Error initializing visualizer: {str(e)}")
            traceback.print_exc()
            raise

    def start(self):
        """Start the trading system."""
        if self.running:
            logging.warning("Trading system is already running")
            print("⚠️ Trading system is already running")
            return

        try:
            logging.info("Starting trading system")
            print("🚀 Starting trading system...")
            self.running = True
            self.start_time = datetime.datetime.now()

            # Initial data fetch
            logging.info("Fetching initial market data")
            print("📊 Fetching initial market data...")
            self.market_data.update_all_timeframes()
            self.last_data_update = datetime.datetime.now()

            # Send startup notification
            if self.telegram and self.telegram.enabled:
                account_balance = self.trade_executor.get_account_balance()
                startup_message = (
                    f"🚀 <b>Trading Bot Started</b>\n\n"
                    f"<b>Symbol:</b> {self.config['SYMBOL']}\n"
                    f"<b>Mode:</b> {'REAL TRADING' if self.config['TRADING_MODE'] == 'real' else 'Simulation Mode'}\n"
                    f"<b>Balance:</b> ${account_balance['usdt_balance']:.2f} USDT\n"
                    f"<b>Leverage:</b> {self.config['LEVERAGE']}x\n"
                    f"<b>Max Risk:</b> {self.config['MAX_RISK_PER_TRADE'] * 100:.1f}% per trade\n"
                    f"<b>Start Time:</b> {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.telegram.send_message(startup_message)

            logging.info("Trading system started successfully")
            print("✅ Trading system started successfully")

        except Exception as e:
            self.running = False
            logging.error(f"Error starting trading system: {str(e)}")
            print(f"❌ Error starting trading system: {str(e)}")
            traceback.print_exc()

    def stop(self):
        """Stop the trading system."""
        if not self.running:
            logging.warning("Trading system is not running")
            print("⚠️ Trading system is not running")
            return

        try:
            logging.info("Stopping trading system")
            print("🛑 Stopping trading system...")
            self.running = False

            # Send shutdown notification
            if self.telegram and self.telegram.enabled:
                runtime = datetime.datetime.now() - self.start_time if self.start_time else datetime.timedelta(0)
                shutdown_message = (
                    f"🛑 <b>Trading Bot Stopped</b>\n\n"
                    f"<b>Runtime:</b> {runtime.total_seconds() / 3600:.1f} hours\n"
                    f"<b>Stop Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.telegram.send_message(shutdown_message)

            logging.info("Trading system stopped successfully")
            print("✅ Trading system stopped successfully")

        except Exception as e:
            logging.error(f"Error stopping trading system: {str(e)}")
            print(f"❌ Error stopping trading system: {str(e)}")
            traceback.print_exc()

    def update(self):
        """Update data, generate predictions and trade signals."""
        if not self.running:
            return {
                "status": "stopped",
                "message": "Trading system is not running"
            }

        try:
            # Update market data
            print("\n📊 Updating market data...")
            data = self.market_data.update_all_timeframes()
            self.last_data_update = datetime.datetime.now()

            # Generate AI predictions
            print("🧠 Generating AI predictions...")
            ai_prediction = self.ai_model.multi_timeframe_prediction(data)
            self.last_prediction = ai_prediction

            # Analyze market and generate signals
            print("📈 Analyzing market conditions...")
            market_analysis = self.strategy.analyze_market(data, ai_prediction)
            trade_signals = self.strategy.generate_trade_signals(market_analysis)
            self.last_trade_signal = trade_signals

            # Print signal information
            signal_type = "ENTRY" if trade_signals["entry_signal"] else "EXIT" if trade_signals[
                "exit_signal"] else "HOLD"
            signal_details = trade_signals["entry_signal"] or trade_signals["exit_signal"] or "hold"
            confidence = trade_signals["confidence"]

            print(f"\n📊 TRADE SIGNAL: {signal_type} - {signal_details.upper()} with {confidence:.2%} confidence")

            if len(trade_signals["reasoning"]) > 0:
                print("📝 Reasoning:")
                for reason in trade_signals["reasoning"]:
                    print(f"  - {reason}")

            # Execute trade if appropriate and trading is enabled
            if self.trading_enabled and (trade_signals["entry_signal"] or trade_signals["exit_signal"]):
                if self.config["TRADING_MODE"] == "real":
                    print("💰 Executing trade...")
                    self._execute_trade(trade_signals)
                else:
                    print("🔹 SIMULATION MODE: No real trade executed")

                    # Log simulated trades for tracking
                    self._log_simulated_trade(trade_signals)

            # Generate chart for the main timeframe (if data is available)
            main_timeframe = "15"  # Default to 15-minute timeframe for charts
            if main_timeframe in data and not data[main_timeframe].empty:
                print("📊 Generating chart...")
                chart_path = self.visualizer.generate_price_chart(data[main_timeframe], True)
                if chart_path and self.telegram and self.telegram.enabled:
                    self.telegram.send_chart(
                        chart_path,
                        f"{self.config['SYMBOL']} - {signal_type} {signal_details.upper()} ({confidence:.2%})"
                    )

            # Track account value for performance monitoring
            self._update_account_history()

            return {
                "status": "success",
                "market_data": data,
                "ai_prediction": ai_prediction,
                "trade_signals": trade_signals
            }

        except Exception as e:
            error_msg = f"Error updating trading system: {str(e)}"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            traceback.print_exc()

            # Send error alert if configured
            if self.telegram and self.telegram.enabled:
                self.telegram.send_error_alert("Update Error", str(e))

            return {
                "status": "error",
                "message": error_msg
            }

    def _execute_trade(self, trade_signals):
        """Execute a trade based on signals."""
        try:
            # Get current market price
            current_price = self.market_data.get_latest_price()

            if not current_price:
                logging.error("Cannot execute trade: Failed to get current price")
                return False

            # Calculate position size based on account balance and risk
            account_balance = self.trade_executor.get_account_balance()
            available_balance = account_balance["available_balance"]
            position_value = available_balance * trade_signals["position_sizing"]

            # Entry signals
            if trade_signals["entry_signal"] == "long":
                # Calculate quantity based on leverage
                quantity = (position_value * self.config["LEVERAGE"]) / current_price

                # Calculate stop loss and take profit prices
                stop_loss_price = current_price * (1 - trade_signals["stop_loss_pct"])
                take_profit_price = current_price * (1 + trade_signals["take_profit_pct"])

                # Execute order
                order_result = self.trade_executor.place_market_order(
                    side="Buy",
                    quantity=quantity,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )

                # Log trade
                if order_result["success"]:
                    trade_data = {
                        "type": "entry",
                        "side": "buy",
                        "price": current_price,
                        "quantity": quantity,
                        "value": position_value,
                        "stop_loss": stop_loss_price,
                        "take_profit": take_profit_price,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "order_id": order_result["order_id"],
                        "confidence": trade_signals["confidence"]
                    }
                    self.trades.append(trade_data)

                    # Send notification
                    if self.telegram and self.telegram.enabled:
                        self.telegram.send_trade_alert(
                            action="buy",
                            symbol=self.config["SYMBOL"],
                            price=current_price,
                            quantity=quantity,
                            confidence=trade_signals["confidence"],
                            risk=trade_signals["position_sizing"]
                        )

                    logging.info(f"Executed LONG entry at ${current_price:.2f}")
                    return True
                else:
                    logging.error(f"Failed to execute LONG entry: {order_result['message']}")
                    return False

            elif trade_signals["entry_signal"] == "short":
                # Calculate quantity based on leverage
                quantity = (position_value * self.config["LEVERAGE"]) / current_price

                # Calculate stop loss and take profit prices
                stop_loss_price = current_price * (1 + trade_signals["stop_loss_pct"])
                take_profit_price = current_price * (1 - trade_signals["take_profit_pct"])

                # Execute order
                order_result = self.trade_executor.place_market_order(
                    side="Sell",
                    quantity=quantity,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )

                # Log trade
                if order_result["success"]:
                    trade_data = {
                        "type": "entry",
                        "side": "sell",
                        "price": current_price,
                        "quantity": quantity,
                        "value": position_value,
                        "stop_loss": stop_loss_price,
                        "take_profit": take_profit_price,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "order_id": order_result["order_id"],
                        "confidence": trade_signals["confidence"]
                    }
                    self.trades.append(trade_data)

                    # Send notification
                    if self.telegram and self.telegram.enabled:
                        self.telegram.send_trade_alert(
                            action="sell",
                            symbol=self.config["SYMBOL"],
                            price=current_price,
                            quantity=quantity,
                            confidence=trade_signals["confidence"],
                            risk=trade_signals["position_sizing"]
                        )

                    logging.info(f"Executed SHORT entry at ${current_price:.2f}")
                    return True
                else:
                    logging.error(f"Failed to execute SHORT entry: {order_result['message']}")
                    return False

            # Exit signals
            elif trade_signals["exit_signal"] == "close_long":
                # Find the matching open long position
                open_positions = self.trade_executor.get_open_positions()
                long_position = next((pos for pos in open_positions if pos["side"] == "Buy"), None)

                if not long_position:
                    logging.warning("Exit signal received but no long position found")
                    return False

                # Execute close order
                order_result = self.trade_executor.place_market_order(
                    side="Sell",
                    quantity=long_position["size"],
                    reduceOnly=True
                )

                # Log trade
                if order_result["success"]:
                    # Find the entry trade
                    entry_trade = next(
                        (t for t in reversed(self.trades) if t["type"] == "entry" and t["side"] == "buy"), None)

                    if entry_trade:
                        # Calculate profit/loss
                        entry_price = entry_trade["price"]
                        exit_price = current_price
                        quantity = long_position["size"]
                        pnl = (exit_price - entry_price) * quantity
                        pnl_percent = (exit_price - entry_price) / entry_price

                        trade_data = {
                            "type": "exit",
                            "side": "sell",
                            "price": current_price,
                            "quantity": quantity,
                            "pnl": pnl,
                            "pnl_percent": pnl_percent,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "order_id": order_result["order_id"]
                        }
                        self.trades.append(trade_data)

                    logging.info(f"Executed LONG exit at ${current_price:.2f}")
                    return True
                else:
                    logging.error(f"Failed to execute LONG exit: {order_result['message']}")
                    return False

            elif trade_signals["exit_signal"] == "close_short":
                # Find the matching open short position
                open_positions = self.trade_executor.get_open_positions()
                short_position = next((pos for pos in open_positions if pos["side"] == "Sell"), None)

                if not short_position:
                    logging.warning("Exit signal received but no short position found")
                    return False

                # Execute close order
                order_result = self.trade_executor.place_market_order(
                    side="Buy",
                    quantity=short_position["size"],
                    reduceOnly=True
                )

                # Log trade
                if order_result["success"]:
                    # Find the entry trade
                    entry_trade = next(
                        (t for t in reversed(self.trades) if t["type"] == "entry" and t["side"] == "sell"), None)

                    if entry_trade:
                        # Calculate profit/loss
                        entry_price = entry_trade["price"]
                        exit_price = current_price
                        quantity = short_position["size"]
                        pnl = (entry_price - exit_price) * quantity
                        pnl_percent = (entry_price - exit_price) / entry_price

                        trade_data = {
                            "type": "exit",
                            "side": "buy",
                            "price": current_price,
                            "quantity": quantity,
                            "pnl": pnl,
                            "pnl_percent": pnl_percent,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "order_id": order_result["order_id"]
                        }
                        self.trades.append(trade_data)

                    logging.info(f"Executed SHORT exit at ${current_price:.2f}")
                    return True
                else:
                    logging.error(f"Failed to execute SHORT exit: {order_result['message']}")
                    return False

            return False

        except Exception as e:
            logging.error(f"Error executing trade: {str(e)}")
            traceback.print_exc()
            return False


    def _log_simulated_trade(self, trade_signals):
        """Log simulated trades for tracking in simulation mode."""
        try:
            # Get current market price
            current_price = self.market_data.get_latest_price()

            if not current_price:
                logging.error("Cannot log simulated trade: Failed to get current price")
                return

            # Calculate position size based on account balance and risk
            position_value = 10000 * trade_signals["position_sizing"]  # Assume $10K account in simulation

            # Entry signals
            if trade_signals["entry_signal"] == "long":
                # Calculate quantity based on leverage
                quantity = (position_value * self.config["LEVERAGE"]) / current_price

                # Calculate stop loss and take profit prices
                stop_loss_price = current_price * (1 - trade_signals["stop_loss_pct"])
                take_profit_price = current_price * (1 + trade_signals["take_profit_pct"])

                # Log simulated trade
                trade_data = {
                    "type": "entry",
                    "side": "buy",
                    "price": current_price,
                    "quantity": quantity,
                    "value": position_value,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "order_id": f"sim-{int(time.time())}",
                    "confidence": trade_signals["confidence"],
                    "simulated": True
                }
                self.trades.append(trade_data)
                logging.info(f"Simulated LONG entry at ${current_price:.2f}")

            elif trade_signals["entry_signal"] == "short":
                # Calculate quantity based on leverage
                quantity = (position_value * self.config["LEVERAGE"]) / current_price

                # Calculate stop loss and take profit prices
                stop_loss_price = current_price * (1 + trade_signals["stop_loss_pct"])
                take_profit_price = current_price * (1 - trade_signals["take_profit_pct"])

                # Log simulated trade
                trade_data = {
                    "type": "entry",
                    "side": "sell",
                    "price": current_price,
                    "quantity": quantity,
                    "value": position_value,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "order_id": f"sim-{int(time.time())}",
                    "confidence": trade_signals["confidence"],
                    "simulated": True
                }
                self.trades.append(trade_data)
                logging.info(f"Simulated SHORT entry at ${current_price:.2f}")

            # Exit signals
            elif trade_signals["exit_signal"] == "close_long":
                # Find the matching open long position (simulated)
                entry_trade = next((t for t in reversed(self.trades)
                                    if t["type"] == "entry" and t["side"] == "buy" and t.get("simulated", False)), None)

                if not entry_trade:
                    logging.warning("Exit signal received but no simulated long position found")
                    return

                # Calculate profit/loss
                entry_price = entry_trade["price"]
                exit_price = current_price
                quantity = entry_trade["quantity"]
                pnl = (exit_price - entry_price) * quantity
                pnl_percent = (exit_price - entry_price) / entry_price

                trade_data = {
                    "type": "exit",
                    "side": "sell",
                    "price": current_price,
                    "quantity": quantity,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "order_id": f"sim-{int(time.time())}",
                    "simulated": True
                }
                self.trades.append(trade_data)
                logging.info(f"Simulated LONG exit at ${current_price:.2f}, P&L: {pnl_percent:.2%}")

            elif trade_signals["exit_signal"] == "close_short":
                # Find the matching open short position (simulated)
                entry_trade = next((t for t in reversed(self.trades)
                                    if t["type"] == "entry" and t["side"] == "sell" and t.get("simulated", False)),
                                   None)

                if not entry_trade:
                    logging.warning("Exit signal received but no simulated short position found")
                    return

                # Calculate profit/loss
                entry_price = entry_trade["price"]
                exit_price = current_price
                quantity = entry_trade["quantity"]
                pnl = (entry_price - exit_price) * quantity
                pnl_percent = (entry_price - exit_price) / entry_price

                trade_data = {
                    "type": "exit",
                    "side": "buy",
                    "price": current_price,
                    "quantity": quantity,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "order_id": f"sim-{int(time.time())}",
                    "simulated": True
                }
                self.trades.append(trade_data)
                logging.info(f"Simulated SHORT exit at ${current_price:.2f}, P&L: {pnl_percent:.2%}")

        except Exception as e:
            logging.error(f"Error logging simulated trade: {str(e)}")

    def _update_account_history(self):
        """Update account history for performance tracking."""
        try:
            if self.config["TRADING_MODE"] == "real":
                # Get actual account balance in real trading mode
                balance = self.trade_executor.get_account_balance()
                equity = balance["usdt_balance"] + balance["unrealized_pnl"]
            else:
                # Calculate simulated equity in simulation mode
                initial_equity = 10000.0  # Assume $10K starting balance

                # Calculate P&L from completed trades
                completed_pnl = sum(t.get("pnl", 0) for t in self.trades if t["type"] == "exit")

                # Calculate unrealized P&L from open positions
                unrealized_pnl = 0
                current_price = self.market_data.get_latest_price()

                if current_price:
                    # Find open long positions
                    open_longs = [t for t in self.trades if t["type"] == "entry" and t["side"] == "buy"
                                  and not any(ex["type"] == "exit" and ex["side"] == "sell" for ex in self.trades)]

                    # Find open short positions
                    open_shorts = [t for t in self.trades if t["type"] == "entry" and t["side"] == "sell"
                                   and not any(ex["type"] == "exit" and ex["side"] == "buy" for ex in self.trades)]

                    # Calculate unrealized P&L for longs
                    for position in open_longs:
                        entry_price = position["price"]
                        quantity = position["quantity"]
                        unrealized_pnl += (current_price - entry_price) * quantity

                    # Calculate unrealized P&L for shorts
                    for position in open_shorts:
                        entry_price = position["price"]
                        quantity = position["quantity"]
                        unrealized_pnl += (entry_price - current_price) * quantity

                equity = initial_equity + completed_pnl + unrealized_pnl

            # Add to account history
            self.account_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "equity": equity,
                "unrealized_pnl": balance["unrealized_pnl"] if self.config["TRADING_MODE"] == "real" else unrealized_pnl
            })

            # Limit history length
            if len(self.account_history) > 1000:
                self.account_history = self.account_history[-1000:]

        except Exception as e:
            logging.error(f"Error updating account history: {str(e)}")

    def get_status(self):
        """Get the current system status."""
        return {
            "running": self.running,
            "running_since": self.start_time.isoformat() if self.start_time else None,
            "trading_enabled": self.trading_enabled,
            "symbol": self.config["SYMBOL"],
            "leverage": self.config["LEVERAGE"],
            "trading_mode": self.config["TRADING_MODE"],
            "ai_status": "healthy" if hasattr(self, 'ai_model') else "not initialized",
            "last_prediction": self.last_prediction["overall_action"] if self.last_prediction else None,
            "last_data_update": self.last_data_update.isoformat() if self.last_data_update else None,
            "websocket_status": "connected"  # Placeholder for websocket status
        }

    def get_account_balance(self) -> Dict:
        """Get the current account balance from Bybit (even in simulation mode)."""
        try:
            if self.bybit_client:
                response = self.bybit_client.get_wallet_balance(
                    accountType="UNIFIED",  # Only use UNIFIED, not CONTRACT
                    coin="USDT"
                )

                # Debug log the response
                logging.debug(f"Wallet balance response: {response}")

                if response["retCode"] != 0:
                    logging.error(f"Failed to get account balance: {response['retMsg']}")
                    return self._get_simulated_balance()

                # Extract USDT balance
                result_data = response["result"]["list"][0]
                coin_data = result_data["coin"]
                usdt_data = next((item for item in coin_data if item["coin"] == "USDT"), None)

                if not usdt_data:
                    logging.error("USDT balance not found in account")
                    return self._get_simulated_balance()

                # Handle empty strings with safe conversion
                def safe_float(value, default=0.0):
                    if value == '' or value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                # Get available balance from proper fields
                available_balance = safe_float(result_data.get("totalAvailableBalance"))

                return {
                    "usdt_balance": safe_float(usdt_data.get("walletBalance", 0.0)),
                    "unrealized_pnl": safe_float(usdt_data.get("unrealisedPnl", 0.0)),
                    "available_balance": available_balance,
                    "used_margin": safe_float(usdt_data.get("totalOrderIM", 0.0))
                }
            else:
                logging.warning("No Bybit client available - using simulated balance")
                return self._get_simulated_balance()

        except Exception as e:
            logging.error(f"Error getting account balance: {str(e)}")
            return self._get_simulated_balance()

    def _get_simulated_balance(self) -> Dict:
        """Return simulated balance values."""
        return {
            "usdt_balance": 10000.0,
            "unrealized_pnl": 0.0,
            "available_balance": 10000.0,
            "used_margin": 0.0
        }

    def get_open_positions(self):
        """Get all open positions."""
        if hasattr(self, 'trade_executor'):
            return self.trade_executor.get_open_positions()
        return []

    def get_performance_metrics(self):
        """Get the current performance metrics."""
        try:
            # Calculate performance metrics
            completed_trades = [t for t in self.trades if t["type"] == "exit"]
            winning_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]

            total_trades = len(completed_trades)
            winning_count = len(winning_trades)

            # Avoid division by zero
            win_rate = winning_count / total_trades if total_trades > 0 else 0

            # Calculate P&L
            total_pnl = sum(t.get("pnl", 0) for t in completed_trades)

            # Calculate max drawdown
            if self.account_history:
                equity_values = [point["equity"] for point in self.account_history]
                max_drawdown = 0
                peak = equity_values[0]

                for value in equity_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0

            # Calculate Sharpe ratio (simplified)
            if len(completed_trades) > 1:
                returns = [t.get("pnl_percent", 0) for t in completed_trades]
                avg_return = sum(returns) / len(returns)
                std_dev = np.std(returns) if len(returns) > 1 else 1
                sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
            else:
                sharpe_ratio = 0

            # Estimate AI accuracy
            if self.predictions:
                # Compare predictions to actual price movements
                correct_predictions = 0
                total_predictions = 0

                for pred in self.predictions:
                    if "action" in pred and "actual_movement" in pred:
                        total_predictions += 1
                        if (pred["action"] == "buy" and pred["actual_movement"] > 0) or \
                                (pred["action"] == "sell" and pred["actual_movement"] < 0) or \
                                (pred["action"] == "hold" and abs(pred["actual_movement"]) < 0.005):
                            correct_predictions += 1

                ai_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.5
            else:
                ai_accuracy = 0.5  # Default accuracy

            # Initial equity (fixed for simulation, or first account history point)
            initial_equity = 10000.0
            if self.account_history:
                initial_equity = self.account_history[0]["equity"]

            # Current equity
            current_equity = self.get_account_balance()["usdt_balance"]
            if self.account_history:
                current_equity = self.account_history[-1]["equity"]

            # Calculate percent return
            pnl_percent = (current_equity - initial_equity) / initial_equity if initial_equity > 0 else 0

            # Time period
            if self.start_time:
                now = datetime.datetime.now()
                days = (now - self.start_time).days
                hours = (now - self.start_time).seconds // 3600
                period = f"{days}d {hours}h"
            else:
                period = "Unknown"

            return {
                "period": period,
                "net_pnl": total_pnl,
                "pnl_percent": pnl_percent,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "winning_trades": winning_count,
                "losing_trades": total_trades - winning_count,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "ai_accuracy": ai_accuracy
            }

        except Exception as e:
            logging.error(f"Error calculating performance metrics: {str(e)}")
            return {
                "period": "Unknown",
                "net_pnl": 0.0,
                "pnl_percent": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "ai_accuracy": 0.0
            }

    def generate_performance_chart(self):
        """Generate a performance chart."""
        if not self.account_history or len(self.account_history) < 2:
            logging.warning("Cannot generate performance chart: Insufficient history data")
            return ""

        return self.visualizer.generate_performance_chart(self.trades, self.account_history)

    def run_simulation(self, duration_minutes=60, update_interval_seconds=60):
        """Run a trading simulation for the specified duration."""
        print(f"\n▶️ RUNNING SIMULATION MODE for {duration_minutes} minutes")
        print(f"Updates every {update_interval_seconds} seconds")
        print("-" * 60)

        if not self.running:
            self.start()

        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(minutes=duration_minutes)

        try:
            update_count = 0
            while datetime.datetime.now() < end_time and self.running:
                # Current time info
                current_time = datetime.datetime.now()
                elapsed = (current_time - start_time).total_seconds() / 60
                remaining = (end_time - current_time).total_seconds() / 60

                print(
                    f"\n[{current_time.strftime('%H:%M:%S')}] Simulation: {elapsed:.1f} min elapsed, {remaining:.1f} min remaining")

                # Update data and get signals
                update_count += 1
                print(f"Update #{update_count}")

                # Perform update
                update_result = self.update()

                if update_result["status"] == "success":
                    # Display latest price
                    price = self.market_data.get_latest_price()
                    if price:
                        print(f"Current {self.config['SYMBOL']} price: ${price:,.2f}")

                    # Display performance metrics periodically
                    if update_count % 5 == 0:
                        metrics = self.get_performance_metrics()
                        print(f"\nPerformance metrics:")
                        print(f"Total trades: {metrics['total_trades']}")
                        print(f"Win rate: {metrics['win_rate']:.2%}")
                        print(f"Net P&L: ${metrics['net_pnl']:.2f} ({metrics['pnl_percent']:.2%})")

                # Wait before next update
                time.sleep(update_interval_seconds)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")

        finally:
            print("\nSimulation completed")
            self.stop()


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def print_colored(text, color=None):
    """Print colored text to console."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'end': '\033[0m'
    }

    if color and color in colors:
        print(f"{colors[color]}{text}{colors['end']}")
    else:
        print(text)


def display_market_summary(trading_system):
    """Display current market summary in the console."""
    try:
        # Get latest price data
        symbol = trading_system.config["SYMBOL"]
        last_price = trading_system.market_data.get_latest_price()

        # Get account info
        balance = trading_system.get_account_balance()

        # Print summary
        print_colored("\n▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MARKET SUMMARY ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓", "cyan")
        print_colored(f"Current {symbol} Price: ${last_price:.2f}", "bold")
        print_colored(f"Account Balance: ${balance['usdt_balance']:.2f} USDT", "green")

        # Print last AI prediction if available
        if trading_system.last_prediction:
            pred = trading_system.last_prediction
            action_color = "green" if pred["overall_action"] == "buy" else "red" if pred[
                                                                                        "overall_action"] == "sell" else "white"
            print_colored(f"\n▶ LAST AI PREDICTION:", "magenta")
            print_colored(f"  Action: {pred['overall_action'].upper()}", action_color)
            print(f"  Confidence: {pred['overall_confidence']:.2%}")
            print(f"  Signal Strength: {pred['aggregated_signal']:.4f}")

            # Parse timestamp if in ISO format
            if isinstance(pred['timestamp'], str):
                timestamp = datetime.datetime.fromisoformat(pred['timestamp'])
            else:
                timestamp = pred['timestamp']

            print(f"  Timestamp: {timestamp.strftime('%H:%M:%S')}")

        print_colored("▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓", "cyan")
    except Exception as e:
        print(f"Error displaying market summary: {str(e)}")


def main():
    """Main function to run the trading bot."""
    print_colored("=" * 80, "cyan")
    print_colored("   AI-DRIVEN AUTOMATED FUTURES TRADING BOT FOR BYBIT EXCHANGE", "bold")
    print_colored("=" * 80, "cyan")

    # Display config details
    print_colored(f"Symbol: {SYMBOL}", "white")
    print_colored(f"Mode: {'REAL TRADING' if TRADING_MODE == 'real' else 'SIMULATION MODE'}", "yellow")
    print_colored(f"Leverage: {LEVERAGE}x", "white")
    print_colored(f"Max Risk Per Trade: {MAX_RISK_PER_TRADE * 100:.1f}%", "white")
    print_colored(f"Debug Mode: {'Enabled' if DEBUG_MODE else 'Disabled'}", "white")
    print_colored("-" * 80, "cyan")

    # Create trading system
    trading_system = None

    try:
        # Initialize trading system
        trading_system = TradingSystem()

        # Choose operating mode
        print_colored("\nSelect operating mode:", "cyan")
        print("1. Start trading bot (with full functionality)")
        print("2. Run in simulation mode (no real trades)")
        print("3. Check system and API connections only")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            # Get user confirmation for real trading mode
            if TRADING_MODE == "real":
                print_colored("\n⚠️  WARNING: You are about to start the bot in REAL TRADING mode ⚠️", "red")
                print_colored("This will use real funds and execute actual trades on BYBIT!", "red")
                confirm = input(
                    "\nType 'CONFIRM' to proceed with real trading or anything else to use simulation mode: ")

                if confirm != "CONFIRM":
                    print_colored("Switching to simulation mode for safety", "yellow")
                    trading_system.config["TRADING_MODE"] = "simulation"

            # Normal startup
            print_colored("\nStarting trading system in normal mode...", "green")
            trading_system.start()

            # Main loop with improved console output
            try:
                update_interval = 300  # 5 minutes
                print_colored(f"Updates every {update_interval / 60:.1f} minutes. Press Ctrl+C to stop.", "blue")

                while trading_system.running:
                    # Update trading system
                    trading_system.update()

                    # Display market summary
                    display_market_summary(trading_system)

                    # Wait before next update
                    time.sleep(update_interval)

            except KeyboardInterrupt:
                print_colored("\nReceived shutdown signal. Stopping trading system...", "yellow")
                trading_system.stop()
                print_colored("Trading system stopped.", "green")

        elif choice == "2":
            # Ask for simulation duration
            try:
                duration = int(input("\nEnter simulation duration in minutes (default: 60): ") or "60")
                interval = int(input("Enter update interval in seconds (default: 60): ") or "60")
            except ValueError:
                duration = 60
                interval = 60
                print("Invalid input, using defaults")

            # Run simulation
            trading_system.run_simulation(duration, interval)

        elif choice == "3":
            # Check system and API connections only
            print_colored("\nPerforming system check and API connection test...", "blue")

            # Check market data
            print_colored("\nTesting market data connections...", "white")

            # Check API connection by fetching ticker
            try:
                latest_price = trading_system.market_data.get_latest_price()
                if latest_price:
                    print_colored(f"✅ API connection successful - {SYMBOL} price: ${latest_price:.2f}", "green")
                else:
                    print_colored("⚠️ API connection issue - could not get latest price", "yellow")
            except Exception as e:
                print_colored(f"❌ API connection failed: {str(e)}", "red")

            # Check data fetching
            try:
                df = trading_system.market_data.fetch_historical_klines("15", 10)
                if not df.empty:
                    print_colored(f"✅ Successfully fetched {len(df)} klines", "green")
                else:
                    print_colored("⚠️ Data fetching issue - empty dataframe returned", "yellow")
            except Exception as e:
                print_colored(f"❌ Data fetching failed: {str(e)}", "red")

            # Check AI model
            print_colored("\nTesting AI model...", "white")

            try:
                if hasattr(trading_system, 'ai_model') and trading_system.ai_model:
                    print_colored("✅ AI model initialized successfully", "green")

                    # Test prediction with sample data
                    if not df.empty:
                        pred = trading_system.ai_model.predict(df, "15")
                        if pred and pred.get("status") == "success":
                            print_colored(
                                f"✅ AI prediction successful: {pred['action'].upper()} with {pred['confidence']:.2%} confidence",
                                "green")
                        else:
                            print_colored(f"⚠️ AI prediction issue: {pred.get('message', 'Unknown error')}", "yellow")
                else:
                    print_colored("❌ AI model not initialized", "red")
            except Exception as e:
                print_colored(f"❌ AI model test failed: {str(e)}", "red")

            print_colored("\nSystem check completed. Not starting trading operations.", "blue")

        elif choice == "4":
            print_colored("Exiting program...", "blue")

        else:
            print_colored("Invalid choice. Exiting.", "red")

    except Exception as e:
        error_msg = f"Critical error during startup: {str(e)}"
        print_colored(f"\n❌ {error_msg}", "red")
        logging.error(error_msg)
        traceback.print_exc()

    finally:
        # Ensure clean shutdown
        if trading_system and getattr(trading_system, 'running', False):
            try:
                trading_system.stop()
                print_colored("Trading system stopped.", "green")
            except:
                pass

        print_colored("\nProgram finished. Press Enter to exit...", "blue")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_colored(f"Unhandled exception: {str(e)}", "red")
        traceback.print_exc()

    # Keep console open to see results
    input()