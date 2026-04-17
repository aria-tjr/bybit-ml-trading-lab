#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive AI-Driven Automated Futures Trading Script for BYBIT Exchange (Mainnet)
Version: 1.0
Date: March 8, 2025

This script implements a sophisticated AI-driven trading strategy for cryptocurrency
futures trading on the BYBIT mainnet. It uses TensorFlow for predictive modeling,
technical indicators for market analysis, and includes comprehensive risk management.

DISCLAIMER: This script is for educational purposes only. Trading cryptocurrencies
involves substantial risk and may not be suitable for everyone. Past performance is
not indicative of future results. Use this script at your own risk.
"""

# ------------------------------------------------------------------------------
# 1. Imports and Library Compatibility
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
import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import talib
import websocket
import threading
import telegram
from telegram.ext import ApplicationBuilder, CommandHandler
from pybit.unified_trading import HTTP, WebSocket
from typing import Dict, List, Tuple, Union, Optional

# Verify library versions to ensure compatibility

# Note: Some libraries like TA-Lib might not have version attributes easily accessible

# Configure matplotlib for non-interactive environments
matplotlib.use('Agg')

# ------------------------------------------------------------------------------
# 2. BYBIT Mainnet API Configuration
# ------------------------------------------------------------------------------
# IMPORTANT: Replace with your actual BYBIT mainnet API credentials
BYBIT_API_KEY = "YOUR_BYBIT_API_KEY"
BYBIT_API_SECRET = "YOUR_BYBIT_API_SECRET"

# BYBIT Mainnet API endpoints (REAL trading environment)
BYBIT_REST_API_URL = "https://api.bybit.com"
BYBIT_WEBSOCKET_PUBLIC_URL = "wss://stream.bybit.com/v5/public"
BYBIT_WEBSOCKET_PRIVATE_URL = "wss://stream.bybit.com/v5/private"

# Trading parameters
SYMBOL = "BTCUSDT"  # Trading pair
LEVERAGE = 3  # Initial leverage (will be dynamically adjusted by AI)
TIMEFRAMES = ["1", "5", "15", "60"]  # Time intervals in minutes for multi-timeframe analysis
MAX_RISK_PER_TRADE = 0.01  # Default 1% risk per trade (reduced from 5% for safety)
TRADING_MODE = "simulation"  # "simulation" by default, change to "real" for actual trading

# Additional config parameters
DEBUG_MODE = True  # Enable debug output and simplified models for testing
TELEGRAM_ENABLED = False  # Only enable Telegram when credentials are properly set
PERFORMANCE_TRACKING = True  # Track and log performance metrics
HEALTH_CHECK_INTERVAL = 300  # Check system health every 5 minutes (in seconds)
DATA_UPDATE_INTERVAL = 60  # Update market data every 1 minute (in seconds)

# ------------------------------------------------------------------------------
# 3. Telegram Integration and Setup
# ------------------------------------------------------------------------------
# IMPORTANT: Replace with your actual Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"


class TelegramNotifier:
    """Handles all Telegram notifications and commands."""

    def __init__(self, token: str, chat_id: str):
        """Initialize Telegram bot with token and chat ID."""
        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=token)
        self.app = ApplicationBuilder().token(token).build()

        # Register command handlers
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("balance", self._cmd_balance))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(CommandHandler("performance", self._cmd_performance))

        # Start the bot in a separate thread
        self.thread = threading.Thread(target=self.app.run_polling, kwargs={"close_loop": False})
        self.thread.daemon = True
        self.thread.start()

        # Reference to the trading system (will be set later)
        self.trading_system = None

    async def send_message(self, message: str) -> None:
        """Send a text message to the specified chat ID."""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            logging.info(f"Telegram notification sent: {message[:50]}...")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {str(e)}")

    async def send_chart(self, chart_path: str, caption: str = "") -> None:
        """Send a chart image to the specified chat ID."""
        try:
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(chat_id=self.chat_id, photo=photo, caption=caption)
            logging.info(f"Chart sent via Telegram: {chart_path}")
        except Exception as e:
            logging.error(f"Failed to send chart via Telegram: {str(e)}")

    async def send_trade_alert(self, action: str, symbol: str, price: float, quantity: float,
                               confidence: float, risk: float) -> None:
        """Send a well-formatted trade alert."""
        emoji = "🟢" if action.lower() == "buy" else "🔴"
        message = f"{emoji} <b>TRADE EXECUTED: {action.upper()}</b>\n\n" \
                  f"<b>Symbol:</b> {symbol}\n" \
                  f"<b>Price:</b> ${price:.2f}\n" \
                  f"<b>Quantity:</b> {quantity:.6f}\n" \
                  f"<b>Confidence:</b> {confidence:.2%}\n" \
                  f"<b>Risk Level:</b> {risk:.2%}\n" \
                  f"<b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        await self.send_message(message)

    async def send_error_alert(self, error_type: str, error_message: str, critical: bool = False) -> None:
        """Send an error alert."""
        alert = "🚨 <b>CRITICAL ERROR</b> 🚨" if critical else "⚠️ <b>ERROR</b>"
        message = f"{alert}\n\n" \
                  f"<b>Type:</b> {error_type}\n" \
                  f"<b>Message:</b> {error_message}\n" \
                  f"<b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        await self.send_message(message)

    async def send_performance_report(self, data: Dict) -> None:
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
        await self.send_message(message)

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
        await self.send_performance_report(perf)

        # Also send the performance chart if available
        chart_path = self.trading_system.generate_performance_chart()
        if chart_path:
            await self.send_chart(chart_path, "Performance Chart")


# ------------------------------------------------------------------------------
# 4. Logging Configuration
# ------------------------------------------------------------------------------
def setup_logging():
    """Configure logging with rotating file handler and console output."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create timestamped log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")

    # Configure logging format and handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Logging system initialized")
    return log_file


# ------------------------------------------------------------------------------
# 5. AI Model Setup, Training, and Optimization
# ------------------------------------------------------------------------------
class PredictiveModel:
    """AI model for price prediction using TensorFlow/Keras."""

    def __init__(self, symbol: str, timeframes: List[str], model_dir: str = "models"):
        """Initialize the AI predictive model for market analysis."""
        self.symbol = symbol
        self.timeframes = timeframes
        self.model_dir = model_dir

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

        # Debug flag to use simplified model for fast testing
        self.debug_mode = False

        # Initialize or load models for each timeframe
        try:
            for tf in self.timeframes:
                model_path = os.path.join(self.model_dir, f"{self.symbol}_{tf}m_model.h5")
                if os.path.exists(model_path):
                    logging.info(f"Loading existing model for {self.symbol} {tf}m timeframe")
                    try:
                        self.models[tf] = load_model(model_path)
                        print(f"✅ Successfully loaded model for {tf}m timeframe")
                    except Exception as e:
                        logging.error(f"Error loading model for {tf}m timeframe: {str(e)}")
                        print(f"⚠️ Failed to load model for {tf}m timeframe. Creating new model.")
                        self.models[tf] = self._build_model()
                else:
                    logging.info(f"Initializing new model for {self.symbol} {tf}m timeframe")
                    self.models[tf] = self._build_model()
                    print(f"🆕 Created new model for {tf}m timeframe")
        except Exception as e:
            logging.error(f"Error initializing AI model: {str(e)}")
            print(f"⚠️ Error initializing AI model: {str(e)}")
            # Fall back to simplified model for testing
            self.debug_mode = True
            print("🔄 Falling back to simplified prediction model for testing")

    def _build_model(self) -> Sequential:
        """Build and compile the LSTM model architecture."""
        try:
            # Use simplified model in debug mode
            if self.debug_mode:
                # Simple model for testing
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout
                from tensorflow.keras.optimizers import Adam

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

                print("ℹ️ Using simplified model architecture for debug mode")
                return model

            # Regular LSTM model for production
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.look_back, self.features)),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(self.target_steps)  # Predict multiple steps ahead
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error'
            )

            return model

        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            print(f"⚠️ Error building model: {str(e)}")

            # Fall back to even simpler model
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense

            simple_model = Sequential([
                Dense(32, activation='relu', input_shape=(self.features,)),
                Dense(self.target_steps)
            ])

            simple_model.compile(
                optimizer='adam',
                loss='mean_squared_error'
            )

            print("⚠️ Using fallback simple Dense model due to errors")
            return simple_model

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess market data for AI model:
        1. Calculate technical indicators
        2. Normalize features
        3. Create sequences for LSTM input
        """
        # Extract price and volume data
        df = data.copy()

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
        df['tenkan_sen'] = talib.SMA(df['high'].rolling(9).max() + df['low'].rolling(9).min(), timeperiod=9) / 2
        df['kijun_sen'] = talib.SMA(df['high'].rolling(26).max() + df['low'].rolling(26).min(), timeperiod=26) / 2

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
        assert len(feature_cols) == self.features, f"Expected {self.features} features, got {len(feature_cols)}"

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

    def train_model(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Train (or retrain) the model for a specific timeframe."""
        logging.info(f"Training model for {self.symbol} on {timeframe}m timeframe")

        # Preprocess data
        X, y = self.preprocess_data(data)

        if len(X) < 100:  # Check if we have enough data
            logging.warning(f"Not enough data to train model for {timeframe}m timeframe")
            return {
                "status": "error",
                "message": "Not enough data for training"
            }

        # Split data into training and validation sets
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Model checkpoint to save best model
        model_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_model.h5")
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        # Train the model
        model = self.models[timeframe]
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=2
        )

        # Load the best model
        self.models[timeframe] = load_model(model_path)

        # Calculate training and validation metrics
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        return {
            "status": "success",
            "timeframe": timeframe,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epochs_completed": len(history.history['loss']),
            "training_samples": len(X_train),
            "early_stopping": early_stopping.stopped_epoch != 0
        }

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
                    "message": "Insufficient data for prediction"
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
# 6. Historical and Real-time Market Data Fetching
# ------------------------------------------------------------------------------
class MarketDataManager:
    """Manages fetching, processing, and storing market data."""

    def __init__(self, api_key: str, api_secret: str, symbol: str, timeframes: List[str]):
        """Initialize the market data manager."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.timeframes = timeframes

        # Initialize BYBIT REST API client
        self.bybit_client = HTTP(
            testnet=False,  # Use mainnet
            api_key=api_key,
            api_secret=api_secret
        )

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

            # Convert timestamp from milliseconds to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

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
            response = self.bybit_client.get_tickers(
                category="linear",
                symbol=self.symbol
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to fetch ticker: {response['retMsg']}")
                return {}

            ticker = response["result"]["list"][0]

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

    def fetch_order_book(self, limit: int = 50) -> Dict:
        """Fetch order book data for the trading pair."""
        try:
            response = self.bybit_client.get_orderbook(
                category="linear",
                symbol=self.symbol,
                limit=limit
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to fetch order book: {response['retMsg']}")
                return {}

            order_book = response["result"]

            # Update order book data
            self.order_book = {
                "bids": [[float(price), float(qty)] for price, qty in order_book["b"]],
                "asks": [[float(price), float(qty)] for price, qty in order_book["a"]],
                "timestamp": datetime.datetime.now()
            }

            return self.order_book

        except Exception as e:
            logging.error(f"Error fetching order book: {str(e)}")
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

        return (self.latest_prices["bid"] + self.latest_prices["ask"]) / 2

    def get_bid_ask_spread(self) -> float:
        """Get the current bid-ask spread."""
        if self.latest_prices["timestamp"] is None:
            self.fetch_latest_ticker()

        return self.latest_prices["ask"] - self.latest_prices["bid"]


# ------------------------------------------------------------------------------
# 7. Technical Indicator Computation
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
        df_copy = df.copy()

        # Ensure we have the necessary columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df_copy.columns for col in required_cols):
            logging.error("Missing required columns for technical analysis")
            return df_copy

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

        # Fibonacci retracement levels (using recent swing high/low)
        window = 14
        df_copy['swing_high'] = df_copy['high'].rolling(window).max()
        df_copy['swing_low'] = df_copy['low'].rolling(window).min()
        df_copy['fib_range'] = df_copy['swing_high'] - df_copy['swing_low']
        df_copy['fib_0.236'] = df_copy['swing_low'] + 0.236 * df_copy['fib_range']
        df_copy['fib_0.382'] = df_copy['swing_low'] + 0.382 * df_copy['fib_range']
        df_copy['fib_0.5'] = df_copy['swing_low'] + 0.5 * df_copy['fib_range']
        df_copy['fib_0.618'] = df_copy['swing_low'] + 0.618 * df_copy['fib_range']
        df_copy['fib_0.786'] = df_copy['swing_low'] + 0.786 * df_copy['fib_range']

        return df_copy

    @staticmethod
    def analyze_trend(df: pd.DataFrame) -> Dict:
        """
        Analyze the current market trend using technical indicators.

        Args:
            df: DataFrame with technical indicators

        Returns:
            Dictionary with trend analysis results
        """
        # Get the most recent data point
        latest = df.iloc[-1]

        # Analyze trend using EMAs
        ema_trend = 0
        if latest['close'] > latest['ema_50'] and latest['ema_9'] > latest['ema_21']:
            ema_trend = 1  # Bullish
        elif latest['close'] < latest['ema_50'] and latest['ema_9'] < latest['ema_21']:
            ema_trend = -1  # Bearish

        # Analyze RSI
        rsi_signal = 0
        if latest['rsi'] > 70:
            rsi_signal = -1  # Overbought
        elif latest['rsi'] < 30:
            rsi_signal = 1  # Oversold

        # Analyze MACD
        macd_signal = 0
        if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
            macd_signal = 1  # Bullish
        elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
            macd_signal = -1  # Bearish

        # Analyze Bollinger Bands
        bb_signal = 0
        if latest['close'] > latest['bb_upper']:
            bb_signal = -1  # Overbought
        elif latest['close'] < latest['bb_lower']:
            bb_signal = 1  # Oversold

        # Analyze Stochastic
        stoch_signal = 0
        if latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            stoch_signal = -1  # Overbought
        elif latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            stoch_signal = 1  # Oversold

        # Volume analysis
        volume_signal = 0
        if latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5:
            # Significant volume increase
            if latest['close'] > latest['open']:
                volume_signal = 1  # Bullish volume
            else:
                volume_signal = -1  # Bearish volume

        # Calculate overall trend score
        signals = [ema_trend, rsi_signal, macd_signal, bb_signal, stoch_signal, volume_signal]
        trend_score = sum(signals) / len(signals)

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
            "trend_strength": trend_strength,
            "trend_score": trend_score,
            "ema_trend": ema_trend,
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "bb_signal": bb_signal,
            "stoch_signal": stoch_signal,
            "volume_signal": volume_signal
        }

    @staticmethod
    def identify_support_resistance(df: pd.DataFrame, window: int = 20,
                                    min_touches: int = 2) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels.

        Args:
            df: DataFrame with OHLCV data
            window: Window size for identifying levels
            min_touches: Minimum number of touches to consider a valid level

        Returns:
            Dictionary with support and resistance levels
        """
        levels = {
            "support": [],
            "resistance": []
        }

        # Get highs and lows
        highs = df['high'].values
        lows = df['low'].values

        # Identify resistance levels from highs
        for i in range(window, len(df) - window):
            # Check if the current high is a local maximum
            if highs[i] == max(highs[i - window:i + window + 1]):
                # Count number of touches (price approaching within 0.5%)
                touch_count = 0
                for j in range(i - window, i + window + 1):
                    if abs(highs[j] - highs[i]) / highs[i] < 0.005:
                        touch_count += 1

                # Add level if it has enough touches
                if touch_count >= min_touches:
                    levels["resistance"].append(highs[i])

        # Identify support levels from lows
        for i in range(window, len(df) - window):
            # Check if the current low is a local minimum
            if lows[i] == min(lows[i - window:i + window + 1]):
                # Count number of touches (price approaching within 0.5%)
                touch_count = 0
                for j in range(i - window, i + window + 1):
                    if abs(lows[j] - lows[i]) / lows[i] < 0.005:
                        touch_count += 1

                # Add level if it has enough touches
                if touch_count >= min_touches:
                    levels["support"].append(lows[i])

        # Remove duplicates and sort
        levels["support"] = sorted(list(set([round(lvl, 2) for lvl in levels["support"]])))
        levels["resistance"] = sorted(list(set([round(lvl, 2) for lvl in levels["resistance"]])))

        return levels

    @staticmethod
    def detect_divergence(df: pd.DataFrame) -> Dict:
        """
        Detect price/indicator divergences (bullish or bearish).

        Args:
            df: DataFrame with technical indicators

        Returns:
            Dictionary with divergence analysis results
        """
        divergences = {
            "rsi_bullish_divergence": False,
            "rsi_bearish_divergence": False,
            "macd_bullish_divergence": False,
            "macd_bearish_divergence": False
        }

        # We need at least 30 data points for divergence analysis
        if len(df) < 30:
            return divergences

        # Look at the last 30 periods
        df_recent = df.iloc[-30:].copy()

        # Find local minima and maxima for price
        df_recent['price_min'] = df_recent['close'].rolling(5).min().shift(-2)
        df_recent['price_max'] = df_recent['close'].rolling(5).max().shift(-2)

        # Find local minima and maxima for RSI
        df_recent['rsi_min'] = df_recent['rsi'].rolling(5).min().shift(-2)
        df_recent['rsi_max'] = df_recent['rsi'].rolling(5).max().shift(-2)

        # Find local minima and maxima for MACD
        df_recent['macd_min'] = df_recent['macd'].rolling(5).min().shift(-2)
        df_recent['macd_max'] = df_recent['macd'].rolling(5).max().shift(-2)

        # Identify price local minima and maxima
        price_min_idx = df_recent[df_recent['close'] == df_recent['price_min']].index
        price_max_idx = df_recent[df_recent['close'] == df_recent['price_max']].index

        # Identify RSI local minima and maxima
        rsi_min_idx = df_recent[df_recent['rsi'] == df_recent['rsi_min']].index
        rsi_max_idx = df_recent[df_recent['rsi'] == df_recent['rsi_max']].index

        # Identify MACD local minima and maxima
        macd_min_idx = df_recent[df_recent['macd'] == df_recent['macd_min']].index
        macd_max_idx = df_recent[df_recent['macd'] == df_recent['macd_max']].index

        # Check for RSI bullish divergence (price making lower lows, RSI making higher lows)
        if len(price_min_idx) >= 2 and len(rsi_min_idx) >= 2:
            # Get the last two price lows and RSI lows
            price_lows = df_recent.loc[price_min_idx]['close'].values[-2:]
            rsi_lows = df_recent.loc[rsi_min_idx]['rsi'].values[-2:]

            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                # Check if price is making lower lows but RSI is making higher lows
                if price_lows[1] < price_lows[0] and rsi_lows[1] > rsi_lows[0]:
                    divergences["rsi_bullish_divergence"] = True

        # Check for RSI bearish divergence (price making higher highs, RSI making lower highs)
        if len(price_max_idx) >= 2 and len(rsi_max_idx) >= 2:
            # Get the last two price highs and RSI highs
            price_highs = df_recent.loc[price_max_idx]['close'].values[-2:]
            rsi_highs = df_recent.loc[rsi_max_idx]['rsi'].values[-2:]

            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                # Check if price is making higher highs but RSI is making lower highs
                if price_highs[1] > price_highs[0] and rsi_highs[1] < rsi_highs[0]:
                    divergences["rsi_bearish_divergence"] = True

        # Similar checks for MACD divergences
        if len(price_min_idx) >= 2 and len(macd_min_idx) >= 2:
            price_lows = df_recent.loc[price_min_idx]['close'].values[-2:]
            macd_lows = df_recent.loc[macd_min_idx]['macd'].values[-2:]

            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                if price_lows[1] < price_lows[0] and macd_lows[1] > macd_lows[0]:
                    divergences["macd_bullish_divergence"] = True

        if len(price_max_idx) >= 2 and len(macd_max_idx) >= 2:
            price_highs = df_recent.loc[price_max_idx]['close'].values[-2:]
            macd_highs = df_recent.loc[macd_max_idx]['macd'].values[-2:]

            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                if price_highs[1] > price_highs[0] and macd_highs[1] < macd_highs[0]:
                    divergences["macd_bearish_divergence"] = True

        return divergences


# ------------------------------------------------------------------------------
# 8. AI-Driven Futures Trading Strategy
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
                # Calculate technical indicators
                df_with_indicators = self.ta.calculate_indicators(data_dict[tf])

                # Analyze trend
                trend_analysis = self.ta.analyze_trend(df_with_indicators)

                # Identify support and resistance
                support_resistance = self.ta.identify_support_resistance(df_with_indicators)

                # Detect divergences
                divergences = self.ta.detect_divergence(df_with_indicators)

                # Store analysis results for this timeframe
                analysis_results[tf] = {
                    "trend_analysis": trend_analysis,
                    "support_resistance": support_resistance,
                    "divergences": divergences,
                    "last_close": float(df_with_indicators['close'].iloc[-1]),
                    "last_volume": float(df_with_indicators['volume'].iloc[-1]),
                    "last_timestamp": df_with_indicators.index[-1].isoformat()
                }

        # Combine technical analysis with AI predictions
        combined_analysis = {
            "timeframe_analysis": analysis_results,
            "ai_prediction": ai_prediction,
            "timestamp": datetime.datetime.now().isoformat()
        }

        return combined_analysis

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
        ai_action = ai_prediction["overall_action"]
        ai_confidence = ai_prediction["overall_confidence"]

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
            # Evaluate exit conditions
            if self.current_position == "long":
                # Consider exit long if AI suggests sell or technical analysis is bearish
                bearish_timeframes = 0
                total_timeframes = 0

                for tf, analysis in timeframe_analysis.items():
                    if "trend_analysis" in analysis:
                        total_timeframes += 1
                        if analysis["trend_analysis"]["trend_direction"] == "bearish":
                            bearish_timeframes += 1

                bearish_ratio = bearish_timeframes / total_timeframes if total_timeframes > 0 else 0

                if ai_action == "sell" and ai_confidence > self.exit_threshold:
                    signals["exit_signal"] = "close_long"
                    signals["confidence"] = ai_confidence
                    signals["reasoning"].append(f"AI predicts sell with {ai_confidence:.2%} confidence")

                elif bearish_ratio > 0.6:
                    signals["exit_signal"] = "close_long"
                    signals["confidence"] = bearish_ratio
                    signals["reasoning"].append(
                        f"Technical analysis bearish in {bearish_timeframes}/{total_timeframes} timeframes")

                # Check for bearish divergences
                bearish_divergences = 0
                for tf, analysis in timeframe_analysis.items():
                    if "divergences" in analysis:
                        if analysis["divergences"]["rsi_bearish_divergence"]:
                            bearish_divergences += 1
                        if analysis["divergences"]["macd_bearish_divergence"]:
                            bearish_divergences += 1

                if bearish_divergences >= 2:
                    signals["exit_signal"] = "close_long"
                    signals["confidence"] = max(signals["confidence"], 0.7)
                    signals["reasoning"].append(f"Detected {bearish_divergences} bearish divergences")

            elif self.current_position == "short":
                # Consider exit short if AI suggests buy or technical analysis is bullish
                bullish_timeframes = 0
                total_timeframes = 0

                for tf, analysis in timeframe_analysis.items():
                    if "trend_analysis" in analysis:
                        total_timeframes += 1
                        if analysis["trend_analysis"]["trend_direction"] == "bullish":
                            bullish_timeframes += 1

                bullish_ratio = bullish_timeframes / total_timeframes if total_timeframes > 0 else 0

                if ai_action == "buy" and ai_confidence > self.exit_threshold:
                    signals["exit_signal"] = "close_short"
                    signals["confidence"] = ai_confidence
                    signals["reasoning"].append(f"AI predicts buy with {ai_confidence:.2%} confidence")

                elif bullish_ratio > 0.6:
                    signals["exit_signal"] = "close_short"
                    signals["confidence"] = bullish_ratio
                    signals["reasoning"].append(
                        f"Technical analysis bullish in {bullish_timeframes}/{total_timeframes} timeframes")

                # Check for bullish divergences
                bullish_divergences = 0
                for tf, analysis in timeframe_analysis.items():
                    if "divergences" in analysis:
                        if analysis["divergences"]["rsi_bullish_divergence"]:
                            bullish_divergences += 1
                        if analysis["divergences"]["macd_bullish_divergence"]:
                            bullish_divergences += 1

                if bullish_divergences >= 2:
                    signals["exit_signal"] = "close_short"
                    signals["confidence"] = max(signals["confidence"], 0.7)
                    signals["reasoning"].append(f"Detected {bullish_divergences} bullish divergences")

        else:
            # No current position, evaluate entry conditions

            # Calculate technical consensus
            bullish_scores = []
            bearish_scores = []

            for tf, analysis in timeframe_analysis.items():
                if "trend_analysis" in analysis:
                    trend_score = analysis["trend_analysis"]["trend_score"]
                    if trend_score > 0:
                        bullish_scores.append(trend_score)
                    elif trend_score < 0:
                        bearish_scores.append(abs(trend_score))

            # Calculate average scores
            avg_bullish = sum(bullish_scores) / len(bullish_scores) if bullish_scores else 0
            avg_bearish = sum(bearish_scores) / len(bearish_scores) if bearish_scores else 0

            # Combine AI and technical analysis
            if ai_action == "buy" and ai_confidence > self.entry_threshold:
                # AI suggests buy
                if avg_bullish > 0.3:  # Technical analysis confirms
                    signals["entry_signal"] = "long"
                    signals["confidence"] = (ai_confidence + avg_bullish) / 2
                    signals["reasoning"].append(f"AI predicts buy with {ai_confidence:.2%} confidence")
                    signals["reasoning"].append(
                        f"Technical analysis confirms bullish trend with {avg_bullish:.2f} score")

                    # Calculate dynamic position sizing based on confidence
                    signals["position_sizing"] = min(ai_confidence * self.max_risk_per_trade, self.max_risk_per_trade)

                    # Set dynamic take profit and stop loss based on volatility
                    # Higher volatility = wider targets and stops
                    avg_atr_pct = 0
                    count = 0
                    for tf, analysis in timeframe_analysis.items():
                        if "trend_analysis" in analysis:
                            df = self.ta.calculate_indicators(timeframe_analysis[tf])
                            if "atr" in df.columns:
                                latest_close = df["close"].iloc[-1]
                                latest_atr = df["atr"].iloc[-1]
                                atr_pct = latest_atr / latest_close
                                avg_atr_pct += atr_pct
                                count += 1

                    if count > 0:
                        avg_atr_pct = avg_atr_pct / count
                        # Set take profit at 2-3x ATR and stop loss at 1-1.5x ATR
                        signals["take_profit_pct"] = avg_atr_pct * (2 + signals["confidence"])
                        signals["stop_loss_pct"] = avg_atr_pct * (1 + (1 - signals["confidence"]))

            elif ai_action == "sell" and ai_confidence > self.entry_threshold:
                # AI suggests sell
                if avg_bearish > 0.3:  # Technical analysis confirms
                    signals["entry_signal"] = "short"
                    signals["confidence"] = (ai_confidence + avg_bearish) / 2
                    signals["reasoning"].append(f"AI predicts sell with {ai_confidence:.2%} confidence")
                    signals["reasoning"].append(
                        f"Technical analysis confirms bearish trend with {avg_bearish:.2f} score")

                    # Calculate dynamic position sizing based on confidence
                    signals["position_sizing"] = min(ai_confidence * self.max_risk_per_trade, self.max_risk_per_trade)

                    # Set dynamic take profit and stop loss based on volatility
                    avg_atr_pct = 0
                    count = 0
                    for tf, analysis in timeframe_analysis.items():
                        if tf in timeframe_analysis:
                            df = self.ta.calculate_indicators(timeframe_analysis[tf])
                            if "atr" in df.columns:
                                latest_close = df["close"].iloc[-1]
                                latest_atr = df["atr"].iloc[-1]
                                atr_pct = latest_atr / latest_close
                                avg_atr_pct += atr_pct
                                count += 1

                    if count > 0:
                        avg_atr_pct = avg_atr_pct / count
                        # Set take profit at 2-3x ATR and stop loss at 1-1.5x ATR
                        signals["take_profit_pct"] = avg_atr_pct * (2 + signals["confidence"])
                        signals["stop_loss_pct"] = avg_atr_pct * (1 + (1 - signals["confidence"]))

        return signals

    def update_position(self, position_type: str, position_data: Dict) -> None:
        """Update the current position."""
        self.current_position = position_type
        self.current_position_data = position_data

    def record_trade(self, trade_data: Dict) -> None:
        """Record a completed trade."""
        self.trades.append(trade_data)

        # Update performance metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t["pnl"] > 0])

        self.performance_metrics["total_trades"] = total_trades
        self.performance_metrics["winning_trades"] = winning_trades
        self.performance_metrics["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate average profit
        if total_trades > 0:
            self.performance_metrics["avg_profit"] = sum([t["pnl_percent"] for t in self.trades]) / total_trades

        # Calculate max drawdown
        cumulative_returns = np.cumprod([1 + t["pnl_percent"] for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        self.performance_metrics["max_drawdown"] = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if total_trades > 1:
            returns = [t["pnl_percent"] for t in self.trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            self.performance_metrics["sharpe_ratio"] = avg_return / std_return if std_return > 0 else 0

    def get_performance_metrics(self) -> Dict:
        """Get the current performance metrics."""
        return self.performance_metrics


# ------------------------------------------------------------------------------
# 9. Risk Management and Dynamic Position Sizing
# ------------------------------------------------------------------------------
class RiskManager:
    """Manages risk and calculates position sizes based on account metrics."""

    def __init__(self, max_risk_per_trade: float = 0.05):
        """Initialize the risk manager with maximum risk per trade."""
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_position_size(self, account_balance: float, entry_price: float,
                                stop_loss_price: float, confidence: float,
                                leverage: int = 1) -> Dict:
        """
        Calculate the appropriate position size based on risk parameters.

        Args:
            account_balance: Current account balance in USD
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            confidence: Trade confidence (0-1)
            leverage: Position leverage

        Returns:
            Dictionary with position sizing details
        """
        # Calculate risk amount (USD)
        risk_ratio = min(confidence * self.max_risk_per_trade, self.max_risk_per_trade)
        risk_amount = account_balance * risk_ratio

        # Calculate price difference to stop loss
        price_diff_pct = abs((entry_price - stop_loss_price) / entry_price)

        # Calculate position size in USD
        # With leverage, we need to adjust the risk
        position_size_usd = risk_amount / price_diff_pct * leverage

        # Calculate quantity
        quantity = position_size_usd / entry_price

        return {
            "position_size_usd": position_size_usd,
            "quantity": quantity,
            "risk_amount": risk_amount,
            "risk_ratio": risk_ratio,
            "price_diff_pct": price_diff_pct,
            "leverage": leverage
        }

    def calculate_take_profit_levels(self, entry_price: float, stop_loss_price: float,
                                     risk_reward_ratios: List[float] = [2, 3, 5]) -> List[float]:
        """
        Calculate take profit levels based on risk-reward ratios.

        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            risk_reward_ratios: List of risk-reward ratios for multiple take profit levels

        Returns:
            List of take profit prices
        """
        # Calculate risk (distance from entry to stop)
        risk = abs(entry_price - stop_loss_price)

        # Calculate take profit levels
        if entry_price > stop_loss_price:  # Long position
            take_profits = [entry_price + (risk * ratio) for ratio in risk_reward_ratios]
        else:  # Short position
            take_profits = [entry_price - (risk * ratio) for ratio in risk_reward_ratios]

        return take_profits

    def calculate_trailing_stop(self, entry_price: float, current_price: float,
                                initial_stop_price: float, trail_pct: float = 0.01) -> float:
        """
        Calculate trailing stop price based on price movement.

        Args:
            entry_price: Entry price for the trade
            current_price: Current market price
            initial_stop_price: Initial stop loss price
            trail_pct: Trailing percentage

        Returns:
            Updated trailing stop price
        """
        if entry_price > initial_stop_price:  # Long position
            # Calculate profit distance
            profit_distance = current_price - entry_price
            if profit_distance <= 0:
                return initial_stop_price

            # Calculate new stop based on trailing percentage
            new_stop = current_price * (1 - trail_pct)

            # Only update if new stop is higher than current stop
            return max(new_stop, initial_stop_price)

        else:  # Short position
            # Calculate profit distance
            profit_distance = entry_price - current_price
            if profit_distance <= 0:
                return initial_stop_price

            # Calculate new stop based on trailing percentage
            new_stop = current_price * (1 + trail_pct)

            # Only update if new stop is lower than current stop
            return min(new_stop, initial_stop_price)

    def adjust_leverage(self, market_volatility: float, base_leverage: int = 3) -> int:
        """
        Dynamically adjust leverage based on market volatility.

        Args:
            market_volatility: Measure of market volatility (ATR/price ratio)
            base_leverage: Base leverage to adjust from

        Returns:
            Adjusted leverage value
        """
        # Lower volatility allows higher leverage
        if market_volatility < 0.01:  # Very low volatility
            return min(base_leverage + 2, 10)  # Cap at 10x
        elif market_volatility < 0.02:  # Low volatility
            return min(base_leverage + 1, 7)
        elif market_volatility < 0.03:  # Moderate volatility
            return base_leverage
        elif market_volatility < 0.05:  # High volatility
            return max(base_leverage - 1, 1)
        else:  # Very high volatility
            return max(base_leverage - 2, 1)  # Minimum leverage of 1x


# ------------------------------------------------------------------------------
# 10. Trade Execution and Order Management
# ------------------------------------------------------------------------------
class TradeExecutor:
    """Handles the execution of trades on the BYBIT exchange."""

    def __init__(self, api_key: str, api_secret: str, symbol: str,
                 leverage: int = 3, trading_mode: str = "real"):
        """Initialize the trade executor."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.leverage = leverage
        self.trading_mode = trading_mode

        # Initialize BYBIT client
        self.bybit_client = HTTP(
            testnet=False,  # Use mainnet
            api_key=api_key,
            api_secret=api_secret
        )

        # Set initial leverage
        self.set_leverage(leverage)

        # Initialize risk manager
        self.risk_manager = RiskManager()

        # Initialize order tracking
        self.open_orders = {}
        self.positions = {}

        # Order execution status tracking
        self.last_order_result = None

    def set_leverage(self, leverage: int) -> Dict:
        """Set leverage for the trading pair."""
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
        try:
            response = self.bybit_client.get_wallet_balance(
                accountType="CONTRACT",
                coin="USDT"
            )

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
        try:
            response = self.bybit_client.get_positions(
                category="linear",
                symbol=self.symbol
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to get positions: {response['retMsg']}")
                return []

            positions = response["result"]["list"]

            # Format position data
            formatted_positions = []
            for pos in positions:
                # Skip positions with zero size
                if float(pos["size"]) == 0:
                    continue

                formatted_positions.append({
                    "symbol": pos["symbol"],
                    "side": "Buy" if pos["side"] == "Buy" else "Sell",
                    "size": float(pos["size"]),
                    "entry_price": float(pos["entryPrice"]),
                    "mark_price": float(pos["markPrice"]),
                    "unrealized_pnl": float(pos["unrealisedPnl"]),
                    "leverage": float(pos["leverage"]),
                    "created_time": datetime.datetime.fromtimestamp(int(pos["createdTime"]) / 1000).isoformat()
                })

            return formatted_positions

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
        if self.trading_mode != "real":
            logging.warning("Trading mode is not set to 'real'. Order not placed.")
            return {
                "success": False,
                "message": "Trading mode is not set to 'real'",
                "order_id": None
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

            # Track the order
            self.open_orders[order_id] = {
                "symbol": self.symbol,
                "side": side,
                "quantity": quantity,
                "order_type": "Market",
                "status": "New",
                "created_time": datetime.datetime.now().isoformat()
            }

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

    def close_position(self, position_side: str, quantity: float = None) -> Dict:
        """
        Close an open position fully or partially.

        Args:
            position_side: Position side to close ("Buy" or "Sell")
            quantity: Quantity to close, if None closes entire position

        Returns:
            Dictionary with order result
        """
        try:
            # Get current position
            positions = self.get_open_positions()
            position = next((p for p in positions if p["side"] == position_side), None)

            if not position:
                logging.warning(f"No {position_side} position found to close")
                return {
                    "success": False,
                    "message": f"No {position_side} position found",
                    "order_id": None
                }

            # Determine close quantity
            close_qty = quantity if quantity else position["size"]

            # Determine order side (opposite of position)
            order_side = "Sell" if position_side == "Buy" else "Buy"

            # Place market order to close
            response = self.bybit_client.place_order(
                category="linear",
                symbol=self.symbol,
                side=order_side,
                orderType="Market",
                qty=str(close_qty),
                reduceOnly=True,
                closeOnTrigger=True
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to close position: {response['retMsg']}")
                return {
                    "success": False,
                    "message": f"Failed to close position: {response['retMsg']}",
                    "order_id": None
                }

            order_id = response["result"]["orderId"]
            logging.info(f"Position closing order placed successfully. Order ID: {order_id}")

            return {
                "success": True,
                "message": f"Position closing order placed successfully",
                "order_id": order_id
            }

        except Exception as e:
            error_message = f"Error closing position: {str(e)}"
            logging.error(error_message)

            return {
                "success": False,
                "message": error_message,
                "order_id": None
            }

    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders for the symbol."""
        try:
            response = self.bybit_client.cancel_all_orders(
                category="linear",
                symbol=self.symbol
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to cancel all orders: {response['retMsg']}")
                return {
                    "success": False,
                    "message": f"Failed to cancel all orders: {response['retMsg']}"
                }

            logging.info(f"All orders for {self.symbol} canceled successfully")
            return {
                "success": True,
                "message": "All orders canceled successfully"
            }

        except Exception as e:
            error_message = f"Error canceling all orders: {str(e)}"
            logging.error(error_message)

            return {
                "success": False,
                "message": error_message
            }

    def update_trailing_stop(self, position_side: str, new_stop_price: float) -> Dict:
        """
        Update the trailing stop for an open position.

        Args:
            position_side: Position side ("Buy" or "Sell")
            new_stop_price: New stop price

        Returns:
            Dictionary with update result
        """
        try:
            # Get current position
            positions = self.get_open_positions()
            position = next((p for p in positions if p["side"] == position_side), None)

            if not position:
                logging.warning(f"No {position_side} position found to update trailing stop")
                return {
                    "success": False,
                    "message": f"No {position_side} position found",
                    "order_id": None
                }

            # First, cancel existing stop orders
            self.cancel_all_orders()

            # Determine order side (opposite of position)
            order_side = "Sell" if position_side == "Buy" else "Buy"

            # Place new stop order
            response = self.bybit_client.place_order(
                category="linear",
                symbol=self.symbol,
                side=order_side,
                orderType="Market",
                triggerPrice=str(new_stop_price),
                qty=str(position["size"]),
                triggerBy="LastPrice",
                reduceOnly=True,
                closeOnTrigger=True,
                timeInForce="GoodTillCancel",
                positionIdx=0
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to update trailing stop: {response['retMsg']}")
                return {
                    "success": False,
                    "message": f"Failed to update trailing stop: {response['retMsg']}",
                    "order_id": None
                }

            order_id = response["result"]["orderId"]
            logging.info(f"Trailing stop updated to {new_stop_price}. Order ID: {order_id}")

            return {
                "success": True,
                "message": f"Trailing stop updated to {new_stop_price}",
                "order_id": order_id
            }

        except Exception as e:
            error_message = f"Error updating trailing stop: {str(e)}"
            logging.error(error_message)

            return {
                "success": False,
                "message": error_message,
                "order_id": None
            }

    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get order history for the symbol."""
        try:
            response = self.bybit_client.get_order_history(
                category="linear",
                symbol=self.symbol,
                limit=limit
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to get order history: {response['retMsg']}")
                return []

            orders = response["result"]["list"]

            # Format order data
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    "order_id": order["orderId"],
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "order_type": order["orderType"],
                    "price": float(order["price"]),
                    "quantity": float(order["qty"]),
                    "executed_quantity": float(order["cumExecQty"]),
                    "status": order["orderStatus"],
                    "created_time": datetime.datetime.fromtimestamp(int(order["createdTime"]) / 1000).isoformat()
                })

            return formatted_orders

        except Exception as e:
            logging.error(f"Error getting order history: {str(e)}")
            return []

    def get_execution_history(self, limit: int = 50) -> List[Dict]:
        """Get trade execution history for the symbol."""
        try:
            response = self.bybit_client.get_executions(
                category="linear",
                symbol=self.symbol,
                limit=limit
            )

            if response["retCode"] != 0:
                logging.error(f"Failed to get execution history: {response['retMsg']}")
                return []

            executions = response["result"]["list"]

            # Format execution data
            formatted_executions = []
            for exec in executions:
                formatted_executions.append({
                    "execution_id": exec["execId"],
                    "order_id": exec["orderId"],
                    "symbol": exec["symbol"],
                    "side": exec["side"],
                    "price": float(exec["execPrice"]),
                    "quantity": float(exec["execQty"]),
                    "fee": float(exec["execFee"]),
                    "time": datetime.datetime.fromtimestamp(int(exec["execTime"]) / 1000).isoformat()
                })

            return formatted_executions

        except Exception as e:
            logging.error(f"Error getting execution history: {str(e)}")
            return []


# ------------------------------------------------------------------------------
# 11. Error Handling and Reliability
# ------------------------------------------------------------------------------
class ErrorHandler:
    """Handles errors and implements recovery strategies."""

    def __init__(self, telegram_notifier: TelegramNotifier = None):
        """Initialize the error handler."""
        self.telegram = telegram_notifier
        self.error_counts = {}
        self.last_errors = {}
        self.max_retries = 3
        self.backoff_factor = 2  # Exponential backoff

    async def handle_error(self, error_type: str, error: Exception, critical: bool = False) -> Dict:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            error_type: Type of error (e.g., "API", "Websocket", "AI", etc.)
            error: The exception
            critical: Whether this is a critical error

        Returns:
            Dictionary with error handling result
        """
        error_msg = str(error)
        traceback_str = traceback.format_exc()

        # Log the error
        if critical:
            logging.critical(f"{error_type} ERROR: {error_msg}\n{traceback_str}")
        else:
            logging.error(f"{error_type} ERROR: {error_msg}\n{traceback_str}")

        # Update error counts
        if error_type not in self.error_counts:
            self