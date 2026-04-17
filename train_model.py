#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced LSTM Model Training Script for Bybit Trading Bot
Version: 2.0
Date: March 9, 2025

This script implements a professional-grade training pipeline for LSTM models
that can predict cryptocurrency price movements using historical data from
the BYBIT exchange. It includes enhanced data fetching, preprocessing,
feature engineering, advanced model architectures, and ensemble methods.

This script is designed to work with the enhanced_trading_bot.py system and
follows the same architectural patterns and coding style for consistency.
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
import pickle
import asyncio
import aiohttp
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import tensorflow as tf
# Import necessary ML libraries with fallbacks
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. Using custom scaling implementation.")

# Handle different keras/tensorflow imports
try:
    # Try standalone keras (for keras 3.x)
    import tensorflow as tf
    from keras.models import Sequential, load_model, save_model, Model
    from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
    from keras.layers import Conv1D, MaxPooling1D, Input, Reshape, Permute, multiply, Lambda
    from keras.optimizers import Adam, RMSprop
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.regularizers import l1_l2, l2

    KERAS_SOURCE = "standalone"
except ImportError:
    try:
        # Fall back to tensorflow.keras (for TF 2.x with built-in keras)
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model, save_model, Model
        from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input, Reshape, Permute, multiply, Lambda
        from tensorflow.keras.optimizers import Adam, RMSprop
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.regularizers import l1_l2, l2

        KERAS_SOURCE = "tensorflow"
    except ImportError:
        print("ERROR: Neither standalone Keras nor TensorFlow available. Model training cannot proceed.")
        KERAS_SOURCE = None
        sys.exit(1)

# Try to import TPOT for automated ML
try:
    from tpot import TPOTRegressor

    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False
    print("WARNING: TPOT not available. Automated ML pipeline optimization will be disabled.")

# Import components from enhanced_trading_bot.py
# Note: This assumes enhanced_trading_bot.py is in the same directory
try:
    from enhanced_trading_bot import (
        BybitClient, MarketDataManager, TechnicalIndicators,
        ConfigManager, setup_logging, retry_async, retry_sync,
        SimpleScaler  # Fallback scaler if sklearn not available
    )
except ImportError:
    print("ERROR: Could not import required components from enhanced_trading_bot.py")
    print("Make sure enhanced_trading_bot.py is in the same directory as this script.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Constants and Configuration
# ------------------------------------------------------------------------------

# Default configuration parameters
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAMES = ["1", "5", "15", "60"]
DEFAULT_LOOKBACK_PERIOD = 60  # Number of past candles to use for prediction
DEFAULT_TARGET_STEPS = 5  # Number of future steps to predict
DEFAULT_TRAINING_DAYS = 90  # Days of historical data to fetch (increased from 60)
DEFAULT_TEST_SIZE = 0.2  # Proportion of data to use for validation
DEFAULT_BATCH_SIZE = 64  # Batch size for training
DEFAULT_EPOCHS = 100  # Maximum number of epochs
DEFAULT_PATIENCE = 15  # Patience for early stopping
DEFAULT_LSTM_UNITS = [128, 64]  # Units in each LSTM layer
DEFAULT_DROPOUT_RATE = 0.3  # Dropout rate between LSTM layers (increased from 0.2)
DEFAULT_LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
DEFAULT_L2_REG = 0.001  # L2 regularization factor
DEFAULT_DATA_DIR = "data"  # Directory to store data
DEFAULT_MODEL_DIR = "models"  # Directory to store models
DEFAULT_DEBUG_MODE = False  # Debug mode flag
DEFAULT_USE_ENSEMBLE = True  # Whether to use ensemble methods

# ------------------------------------------------------------------------------
# Custom SimpleScaler class (fallback if sklearn not available)
# ------------------------------------------------------------------------------

if not SKLEARN_AVAILABLE:
    class SimpleScaler:
        """Simple scaler implementation for data normalization when sklearn is not available."""

        def __init__(self):
            self.min_ = None
            self.max_ = None
            self.scale_ = None

        def fit(self, X):
            """Compute min and max values to be used for scaling."""
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            # Handle potential division by zero
            self.scale_ = self.max_ - self.min_
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            """Scale data to [0, 1] range."""
            return (X - self.min_) / self.scale_

        def inverse_transform(self, X):
            """Inverse transform scaled data back to original scale."""
            return X * self.scale_ + self.min_

        def fit_transform(self, X):
            """Fit to data, then transform it."""
            return self.fit(X).transform(X)


# ------------------------------------------------------------------------------
# Custom Loss Functions
# ------------------------------------------------------------------------------

def directional_loss(y_true, y_pred):
    """
    Custom loss function that combines MSE with directional accuracy penalty

    Args:
        y_true: True values
        y_pred: Predicted values
    """
    # Import tensorflow if needed
    if 'tf' not in globals():
        import tensorflow as tf

    # Traditional MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Directional loss component
    # Extract the first prediction column for both true and pred
    # This focuses on the immediate next timeframe prediction
    y_true_direction = tf.sign(y_true[:, 0])
    y_pred_direction = tf.sign(y_pred[:, 0])

    # Calculate directional accuracy (opposite of loss, so we negate it)
    # When directions match, this will be 1, else -1
    direction_match = tf.multiply(y_true_direction, y_pred_direction)

    # Transform to a loss (1 when directions don't match, 0 when they do)
    direction_loss = 0.5 * (1.0 - direction_match)

    # Take mean across batch
    mean_direction_loss = tf.reduce_mean(direction_loss)

    # Combine losses with a weight parameter
    # Higher alpha puts more emphasis on directional accuracy
    alpha = 0.7
    combined_loss = (1 - alpha) * mse_loss + alpha * mean_direction_loss

    return combined_loss


# ------------------------------------------------------------------------------
# Data Fetching and Preprocessing
# ------------------------------------------------------------------------------

@retry_async(max_attempts=3, delay=2)
async def fetch_historical_data(
        api_client: BybitClient,
        symbol: str,
        timeframe: str,
        days: int = DEFAULT_TRAINING_DAYS
) -> pd.DataFrame:
    """
    Fetch historical kline data from Bybit for a specific timeframe.

    Args:
        api_client: Initialized BybitClient instance
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Timeframe in minutes ("1", "5", "15", "60")
        days: Number of days of historical data to fetch

    Returns:
        DataFrame containing the historical data
    """
    logging.info(f"Fetching {days} days of historical {timeframe}m data for {symbol}")

    # Calculate start and end time
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days)

    # Calculate how many candles we need to fetch
    # This depends on the timeframe
    minutes_in_tf = int(timeframe)
    candles_per_day = 24 * 60 / minutes_in_tf
    total_candles = int(candles_per_day * days)

    # Bybit has a limit of 1000 candles per request, so we may need multiple requests
    max_candles_per_request = 1000

    # Initialize an empty DataFrame to store all the data
    all_data = pd.DataFrame()

    # Make multiple requests if needed
    current_end_time = end_time

    while len(all_data) < total_candles:
        # Fetch data
        df = await api_client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=max_candles_per_request,
            end_time=current_end_time
        )

        if df.empty:
            logging.warning(f"No data received for {symbol} {timeframe}m up to {current_end_time}")
            break

        # Append data
        all_data = pd.concat([df, all_data])

        # Update end time for next request (use the earliest timestamp we have)
        if not all_data.empty:
            current_end_time = all_data.index.min()

            # Add some logging about the data we've fetched so far
            logging.info(f"Fetched {len(all_data)} candles for {symbol} {timeframe}m, earliest: {current_end_time}")

        # Check if we have enough data or if we've reached the start time
        if len(all_data) >= total_candles or current_end_time <= start_time:
            break

        # Sleep briefly to avoid hitting rate limits
        await asyncio.sleep(1)

    # Remove duplicates if any
    all_data = all_data[~all_data.index.duplicated(keep='first')]

    # Sort by timestamp
    all_data = all_data.sort_index()

    # Limit to the desired timeframe
    all_data = all_data[all_data.index >= start_time]

    logging.info(f"Successfully fetched {len(all_data)} candles for {symbol} {timeframe}m")

    return all_data


def enhanced_feature_engineering(df):
    """
    Advanced feature engineering pipeline with expanded market indicators

    Args:
        df: DataFrame with market data and basic technical indicators

    Returns:
        DataFrame with additional engineered features
    """
    logging.info("Applying enhanced feature engineering")

    # Preserve original dataframe
    df_features = df.copy()

    try:
        # 1. Add advanced price action features
        df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))
        df_features['volatility'] = df_features['log_return'].rolling(window=20).std() * np.sqrt(252)

        # 2. Add market regime features
        if 'volatility' in df_features.columns:
            df_features['regime_volatility'] = (
                        df_features['volatility'] > df_features['volatility'].rolling(window=100).mean()).astype(int)

        if all(col in df_features.columns for col in ['ema_50', 'ema_200']):
            df_features['trend_strength'] = abs(df_features['ema_50'] - df_features['ema_200']) / df_features['close']

        # 3. Add cyclical features to capture time-based patterns
        if isinstance(df_features.index, pd.DatetimeIndex):
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features.index.hour / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features.index.hour / 24)
            df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features.index.dayofweek / 7)
            df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features.index.dayofweek / 7)

        # 4. Add sequence/pattern detection features
        for window in [3, 5, 7]:
            # Calculate percentage of up-candles in window
            df_features[f'up_candles_{window}'] = (df_features['close'] > df_features['open']).rolling(window).mean()

            # Detect price acceleration/deceleration
            df_features[f'price_accel_{window}'] = df_features['close'].pct_change().diff().rolling(window).mean()

        # 5. Add volume-based features
        df_features['volume_price_trend'] = (df_features['close'] - df_features['close'].shift(1)) * df_features[
            'volume']
        df_features['volume_divergence'] = (df_features['volume'] / df_features['volume'].rolling(20).mean() -
                                            df_features['close'].pct_change().abs())

        # 6. Add volatility breakout features
        if all(col in df_features.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            df_features['bb_squeeze'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']

        if 'atr' in df_features.columns:
            df_features['atr_relative'] = df_features['atr'] / df_features['close'] * 100

        # 7. Add mean reversion features
        for window in [10, 20, 50]:
            df_features[f'dist_from_ma_{window}'] = (df_features['close'] / df_features['close'].rolling(
                window).mean() - 1) * 100

        # 8. Add momentum oscillator features
        if 'rsi' in df_features.columns:
            df_features['rsi_diff'] = df_features['rsi'].diff()
            df_features['rsi_ma'] = df_features['rsi'].rolling(window=10).mean()
            df_features['rsi_regime'] = np.select(
                [df_features['rsi'] < 30, df_features['rsi'] > 70],
                [1, -1],  # 1 for oversold, -1 for overbought
                default=0
            )

        logging.info(f"Added {len(df_features.columns) - len(df.columns)} new engineered features")

    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}")
        traceback.print_exc()

    return df_features


async def preprocess_data(
        df: pd.DataFrame,
        lookback_period: int = DEFAULT_LOOKBACK_PERIOD,
        target_steps: int = DEFAULT_TARGET_STEPS,
        test_size: float = DEFAULT_TEST_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Preprocess the raw market data for LSTM model training with enhanced features.

    This function performs the following steps:
    1. Calculate technical indicators
    2. Apply enhanced feature engineering
    3. Select relevant features
    4. Create target variables (future price changes)
    5. Normalize data
    6. Create sequences for LSTM input
    7. Split into training and testing sets

    Args:
        df: DataFrame with market data
        lookback_period: Number of past candles to use for prediction
        target_steps: Number of future steps to predict
        test_size: Proportion of data to use for validation

    Returns:
        Tuple containing:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training targets
        - y_test: Testing targets
        - scaler: Fitted scaler object for future inverse transformations
    """
    logging.info(f"Preprocessing data with lookback_period={lookback_period}, target_steps={target_steps}")

    try:
        # Handle missing data
        if df is None or df.empty or len(df) < lookback_period + target_steps + 10:
            raise ValueError(f"Insufficient data for preprocessing, got {len(df) if df is not None else 0} rows")

        # Calculate technical indicators
        df_with_indicators = TechnicalIndicators.calculate_indicators(df.copy())

        # Apply enhanced feature engineering
        df_enhanced = enhanced_feature_engineering(df_with_indicators)

        # Remove NaN values
        df_enhanced.dropna(inplace=True)

        # Select features (exclude timestamp index which might cause issues)
        # We'll start with all numerical columns and then filter
        feature_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target columns and any other columns we don't want to use as features
        exclude_patterns = ['future_return', 'timestamp', 'index', 'date']
        feature_cols = [col for col in feature_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]

        logging.info(f"Selected {len(feature_cols)} features for model training")

        # Create target variables (future price changes)
        for i in range(1, target_steps + 1):
            df_enhanced[f'future_return_{i}'] = df_enhanced['close'].pct_change(i).shift(-i)

        # Drop rows with NaN values
        df_enhanced.dropna(inplace=True)

        if len(df_enhanced) < lookback_period + target_steps + 10:
            raise ValueError(f"Insufficient data after preprocessing, got {len(df_enhanced)} rows")

        # Perform feature selection if sklearn is available
        if SKLEARN_AVAILABLE:
            try:
                # Use a random forest to gauge feature importance
                X_sample = df_enhanced[feature_cols].values
                y_sample = df_enhanced[f'future_return_1'].values

                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_sample, y_sample)

                # Get feature importances
                importances = list(zip(feature_cols, rf.feature_importances_))
                importances.sort(key=lambda x: x[1], reverse=True)

                # Log top features
                logging.info("Top 10 features by importance:")
                for feature, importance in importances[:10]:
                    logging.info(f"  {feature}: {importance:.4f}")

                # Keep only the top 80% of features by cumulative importance
                cum_importance = 0
                selected_features = []
                for feature, importance in importances:
                    cum_importance += importance
                    selected_features.append(feature)
                    if cum_importance >= 0.8:  # Keep top 80% of importance
                        break

                feature_cols = selected_features
                logging.info(f"Selected {len(feature_cols)} features after importance filtering")

            except Exception as e:
                logging.warning(f"Feature selection error: {str(e)}")

        # Extract features and targets
        X = df_enhanced[feature_cols].values
        y = df_enhanced[[f'future_return_{i}' for i in range(1, target_steps + 1)]].values

        # Scale features
        if SKLEARN_AVAILABLE:
            scaler = RobustScaler()  # More robust to outliers than MinMaxScaler
        else:
            scaler = SimpleScaler()

        X_scaled = scaler.fit_transform(X)

        # Create sequences for LSTM
        X_seq = []
        y_seq = []

        for i in range(len(X_scaled) - lookback_period):
            X_seq.append(X_scaled[i:i + lookback_period])
            y_seq.append(y[i + lookback_period - 1])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Split into training and testing sets
        if SKLEARN_AVAILABLE:
            # Use time-based split to avoid lookahead bias
            split_idx = int(len(X_seq) * (1 - test_size))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        else:
            # Manual split if sklearn not available
            split_idx = int(len(X_seq) * (1 - test_size))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        logging.info(f"Preprocessing complete. Training data shape: {X_train.shape}, {y_train.shape}")
        logging.info(f"Testing data shape: {X_test.shape}, {y_test.shape}")

        # Store feature names for future reference
        scaler.feature_names = feature_cols

        return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        traceback.print_exc()
        raise


# ------------------------------------------------------------------------------
# Model Building and Training
# ------------------------------------------------------------------------------

def build_advanced_model(
        input_shape: Tuple[int, int],
        output_dim: int,
        model_type: str = "bidirectional_lstm",
        lstm_units: List[int] = DEFAULT_LSTM_UNITS,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        recurrent_dropout: float = 0.2,
        l2_reg: float = DEFAULT_L2_REG,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        use_custom_loss: bool = True
) -> Sequential:
    """
    Build an advanced model architecture for time series prediction.

    Args:
        input_shape: Shape of input data (lookback_period, n_features)
        output_dim: Dimension of output (number of steps to predict)
        model_type: Type of model architecture ('lstm', 'bidirectional_lstm', 'cnn_lstm', 'attention_lstm')
        lstm_units: List of units in each LSTM layer
        dropout_rate: Dropout rate between layers
        recurrent_dropout: Dropout rate for recurrent connections
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for optimizer
        use_custom_loss: Whether to use custom directional loss

    Returns:
        Compiled model
    """
    logging.info(f"Building {model_type} model with input_shape={input_shape}, output_dim={output_dim}")

    if model_type == "lstm":
        # Standard LSTM model with regularization
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=input_shape,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
            name="lstm_1"
        ))

        model.add(BatchNormalization(name="batch_norm_1"))

        # Add additional LSTM layers if specified
        for i, units in enumerate(lstm_units[1:], 2):
            return_sequences = i < len(lstm_units)  # Return sequences for all but the last LSTM layer
            model.add(LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg),
                name=f"lstm_{i}"
            ))
            model.add(BatchNormalization(name=f"batch_norm_{i}"))

        # Add dense layers for prediction
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg), name="dense_1"))
        model.add(BatchNormalization(name="batch_norm_final"))
        model.add(Dropout(dropout_rate, name="dropout_final"))
        model.add(Dense(output_dim, activation='linear', name="output"))

    elif model_type == "bidirectional_lstm":
        # Bidirectional LSTM model
        model = Sequential()

        # First Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(
                lstm_units[0],
                return_sequences=len(lstm_units) > 1,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg)
            ),
            input_shape=input_shape,
            name="bidirectional_1"
        ))

        model.add(BatchNormalization(name="batch_norm_1"))

        # Add additional Bidirectional LSTM layers if specified
        for i, units in enumerate(lstm_units[1:], 2):
            return_sequences = i < len(lstm_units)  # Return sequences for all but the last LSTM layer
            model.add(Bidirectional(
                LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l2(l2_reg)
                ),
                name=f"bidirectional_{i}"
            ))
            model.add(BatchNormalization(name=f"batch_norm_{i}"))

        # Add dense layers for prediction
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg), name="dense_1"))
        model.add(BatchNormalization(name="batch_norm_final"))
        model.add(Dropout(dropout_rate, name="dropout_final"))
        model.add(Dense(output_dim, activation='linear', name="output"))

    elif model_type == "cnn_lstm":
        # CNN-LSTM hybrid model
        model = Sequential()

        # CNN layers for feature extraction
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape,
                         name="conv1d_1"))
        model.add(MaxPooling1D(pool_size=2, name="maxpool_1"))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name="conv1d_2"))
        model.add(MaxPooling1D(pool_size=2, name="maxpool_2"))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv1d_3"))

        # LSTM layers
        model.add(LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
            name="lstm_1"
        ))

        model.add(BatchNormalization(name="batch_norm_1"))

        # Add additional LSTM layers if specified
        for i, units in enumerate(lstm_units[1:], 2):
            return_sequences = i < len(lstm_units)  # Return sequences for all but the last LSTM layer
            model.add(LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg),
                name=f"lstm_{i}"
            ))
            model.add(BatchNormalization(name=f"batch_norm_{i}"))

        # Add dense layers for prediction
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg), name="dense_1"))
        model.add(BatchNormalization(name="batch_norm_final"))
        model.add(Dropout(dropout_rate, name="dropout_final"))
        model.add(Dense(output_dim, activation='linear', name="output"))

    elif model_type == "attention_lstm":
        # LSTM with attention mechanism (functional API)
        inputs = Input(shape=input_shape, name="input")

        # First LSTM layer
        lstm_out = LSTM(
            lstm_units[0],
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
            name="lstm_1"
        )(inputs)

        lstm_out = BatchNormalization(name="batch_norm_1")(lstm_out)

        # Attention mechanism
        attention = Dense(1, activation='tanh', name="attention_dense")(lstm_out)
        attention = Reshape((input_shape[0],), name="attention_reshape")(attention)

        # Softmax for attention weights
        attention_probs = tf.keras.layers.Softmax(name="attention_softmax")(attention)

        # Apply attention weights
        context_vector = tf.keras.layers.Dot(axes=(1, 1), name="apply_attention")(
            [lstm_out, tf.expand_dims(attention_probs, -1)])
        context_vector = tf.keras.layers.Reshape((lstm_units[0],), name="context_vector")(context_vector)

        # Dense layers for prediction
        dense_out = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg), name="dense_1")(context_vector)
        dense_out = BatchNormalization(name="batch_norm_final")(dense_out)
        dense_out = Dropout(dropout_rate, name="dropout_final")(dense_out)
        outputs = Dense(output_dim, activation='linear', name="output")(dense_out)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)

    # Use custom loss if specified
    if use_custom_loss:
        loss = directional_loss
    else:
        loss = 'huber'  # Huber loss is more robust to outliers than MSE

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae', 'mse']
    )

    # Print model summary
    model.summary()

    logging.info(f"{model_type.upper()} model built successfully")
    return model


def hyperparameter_tuning(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_shape: Tuple[int, int],
        output_dim: int,
        max_trials: int = 5
) -> Tuple[Sequential, Dict]:
    """
    Perform hyperparameter tuning for the model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        input_shape: Shape of input data
        output_dim: Dimension of output
        max_trials: Maximum number of hyperparameter combinations to try

    Returns:
        Tuple containing:
        - Best model
        - Best hyperparameters
    """
    logging.info(f"Starting hyperparameter tuning with {max_trials} trials")

    # Define hyperparameter search space
    param_grid = {
        'model_type': ['lstm', 'bidirectional_lstm'],
        'lstm_units': [[64, 32], [128, 64], [256, 128]],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64, 128],
        'use_custom_loss': [True, False]
    }

    # For performance reasons, limit trials
    import itertools
    import random

    # Generate all combinations
    all_combinations = list(itertools.product(
        param_grid['model_type'],
        param_grid['lstm_units'],
        param_grid['dropout_rate'],
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['use_custom_loss']
    ))

    # Shuffle and limit to max_trials
    random.shuffle(all_combinations)
    param_combinations = all_combinations[:max_trials]

    # Track best model and parameters
    best_val_loss = float('inf')
    best_model = None
    best_params = None

    # Try each combination
    for i, (model_type, lstm_units, dropout_rate, learning_rate, batch_size, use_custom_loss) in enumerate(
            param_combinations):
        logging.info(
            f"Trial {i + 1}/{len(param_combinations)}: testing {model_type} with lr={learning_rate}, dropout={dropout_rate}")

        try:
            # Build model
            model = build_advanced_model(
                input_shape=input_shape,
                output_dim=output_dim,
                model_type=model_type,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                use_custom_loss=use_custom_loss
            )

            # Train for a few epochs
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=10,  # Fewer epochs for tuning
                batch_size=batch_size,
                verbose=0
            )

            # Get validation loss
            val_loss = min(history.history['val_loss'])

            logging.info(f"  Validation loss: {val_loss:.6f}")

            # Calculate directional accuracy
            y_pred = model.predict(X_test)
            dir_acc = np.mean(np.sign(y_test[:, 0]) == np.sign(y_pred[:, 0]))

            logging.info(f"  Directional accuracy: {dir_acc:.4f}")

            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_params = {
                    'model_type': model_type,
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'use_custom_loss': use_custom_loss
                }

                logging.info(f"  New best model found!")

        except Exception as e:
            logging.error(f"Error during hyperparameter trial: {str(e)}")

    if best_model is None:
        logging.warning("Hyperparameter tuning failed to find a good model, using default parameters")
        best_model = build_advanced_model(input_shape, output_dim)
        best_params = {
            'model_type': 'bidirectional_lstm',
            'lstm_units': DEFAULT_LSTM_UNITS,
            'dropout_rate': DEFAULT_DROPOUT_RATE,
            'learning_rate': DEFAULT_LEARNING_RATE,
            'batch_size': DEFAULT_BATCH_SIZE,
            'use_custom_loss': True
        }
    else:
        logging.info(f"Best hyperparameters found: {best_params}")

    return best_model, best_params


@retry_sync(max_attempts=2, delay=1)
def train_model(
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hyperparams: Dict,
        model_path: str
) -> Tuple[Sequential, Dict]:
    """
    Train the model with enhanced callbacks and settings.

    Args:
        model: Compiled model
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        hyperparams: Hyperparameter dictionary
        model_path: Path to save the best model

    Returns:
        Tuple containing:
        - Trained model
        - Training history dictionary
    """
    logging.info(f"Training model with {len(X_train)} samples")

    # Extract hyperparameters
    batch_size = hyperparams.get('batch_size', DEFAULT_BATCH_SIZE)
    epochs = hyperparams.get('epochs', DEFAULT_EPOCHS)
    patience = hyperparams.get('patience', DEFAULT_PATIENCE)

    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Set up callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # Save the best model
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=int(patience / 4),  # More aggressive LR reduction
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
        workers=4,
        use_multiprocessing=True
    )

    # Load the best model
    if os.path.exists(model_path):
        model = load_model(
            model_path,
            # Custom objects for custom loss
            custom_objects={'directional_loss': directional_loss} if hyperparams.get('use_custom_loss', True) else None
        )
        logging.info(f"Loaded best model from {model_path}")

    # Evaluate the model
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Model evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

    return model, history.history


def time_series_cv(
        X: np.ndarray,
        y: np.ndarray,
        model_builder_func: Callable,
        n_splits: int = 5,
        **kwargs
) -> Dict:
    """
    Perform time series cross-validation to ensure model generalizes well.

    Args:
        X: Input features
        y: Target values
        model_builder_func: Function to build and compile model
        n_splits: Number of CV splits
        **kwargs: Additional parameters for model training

    Returns:
        Dictionary with CV results
    """
    logging.info(f"Performing time series cross-validation with {n_splits} splits")

    # Initialize results dictionary
    cv_results = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_dir_acc': [],
        'val_dir_acc': []
    }

    # Calculate split sizes
    split_size = len(X) // (n_splits + 1)

    # For each split
    for i in range(n_splits):
        # Define train/validation cutoff
        cutoff = (i + 1) * split_size

        # Split data
        X_train, X_val = X[:cutoff], X[cutoff:cutoff + split_size]
        y_train, y_val = y[:cutoff], y[cutoff:cutoff + split_size]

        logging.info(f"Split {i + 1}/{n_splits}: train_size={len(X_train)}, val_size={len(X_val)}")

        try:
            # Build and train model
            model = model_builder_func(X_train.shape[1:], y_train.shape[1], **kwargs)

            # Define callbacks for early stopping
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]

            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=kwargs.get('epochs', 30),
                batch_size=kwargs.get('batch_size', 64),
                callbacks=callbacks,
                verbose=0
            )

            # Get metrics
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_mae = history.history['mae'][-1]
            val_mae = history.history['val_mae'][-1]

            # Calculate directional accuracy
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            train_dir_acc = np.mean(np.sign(y_train[:, 0]) == np.sign(y_pred_train[:, 0]))
            val_dir_acc = np.mean(np.sign(y_val[:, 0]) == np.sign(y_pred_val[:, 0]))

            # Store results
            cv_results['train_loss'].append(train_loss)
            cv_results['val_loss'].append(val_loss)
            cv_results['train_mae'].append(train_mae)
            cv_results['val_mae'].append(val_mae)
            cv_results['train_dir_acc'].append(train_dir_acc)
            cv_results['val_dir_acc'].append(val_dir_acc)

            logging.info(
                f"  Split {i + 1} results: val_loss={val_loss:.6f}, val_mae={val_mae:.6f}, val_dir_acc={val_dir_acc:.4f}")

        except Exception as e:
            logging.error(f"Error in CV split {i + 1}: {str(e)}")

    # Calculate mean and std of results
    for metric in cv_results:
        if cv_results[metric]:  # Check if list is not empty
            mean_val = np.mean(cv_results[metric])
            std_val = np.std(cv_results[metric])
            logging.info(f"CV {metric}: {mean_val:.6f} ± {std_val:.6f}")

    return cv_results


def build_ensemble_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_shape: Tuple[int, int],
        output_dim: int,
        model_dir: str,
        symbol: str,
        timeframe: str
) -> Tuple[List[Sequential], Callable, List[float]]:
    """
    Build an ensemble of models for more robust predictions.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        input_shape: Shape of input data
        output_dim: Dimension of output
        model_dir: Directory to save models
        symbol: Trading symbol
        timeframe: Timeframe

    Returns:
        Tuple containing:
        - List of trained models
        - Prediction function for ensemble
        - List of model weights
    """
    logging.info("Building ensemble model")

    # Define different model configurations
    model_configs = [
        {"model_type": "lstm", "lstm_units": [128, 64], "dropout_rate": 0.3, "learning_rate": 0.001,
         "use_custom_loss": True},
        {"model_type": "bidirectional_lstm", "lstm_units": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.0005,
         "use_custom_loss": True},
        {"model_type": "lstm", "lstm_units": [256, 128], "dropout_rate": 0.4, "learning_rate": 0.001,
         "use_custom_loss": False}
    ]

    # Train each model
    trained_models = []
    model_weights = []

    for i, config in enumerate(model_configs):
        try:
            # Log progress
            logging.info(f"Training ensemble model {i + 1}/{len(model_configs)}: {config['model_type']}")

            # Model path
            model_path = os.path.join(model_dir, f"{symbol.lower()}_{timeframe}m_ensemble_{i}.h5")

            # Build model
            model = build_advanced_model(
                input_shape=input_shape,
                output_dim=output_dim,
                **config
            )

            # Train with early stopping
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            ]

            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=64,
                callbacks=callbacks,
                verbose=1
            )

            # Load best model
            if os.path.exists(model_path):
                model = load_model(
                    model_path,
                    custom_objects={'directional_loss': directional_loss} if config.get('use_custom_loss',
                                                                                        True) else None
                )

            # Calculate validation loss
            val_loss = model.evaluate(X_test, y_test, verbose=0)[0]

            # Calculate directional accuracy
            y_pred = model.predict(X_test)
            dir_acc = np.mean(np.sign(y_test[:, 0]) == np.sign(y_pred[:, 0]))

            logging.info(f"  Ensemble model {i + 1} val_loss: {val_loss:.6f}, dir_acc: {dir_acc:.4f}")

            # Store model and calculate weight (inverse of loss)
            trained_models.append(model)

            # Higher weight for models with better directional accuracy
            model_weight = (1.0 / val_loss) * (0.5 + 0.5 * dir_acc)  # Blend loss and directional accuracy
            model_weights.append(model_weight)

        except Exception as e:
            logging.error(f"Error training ensemble model {i + 1}: {str(e)}")

    # Normalize weights
    if model_weights:
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [w / total_weight for w in model_weights]
        else:
            # Equal weights if all models failed
            model_weights = [1.0 / len(trained_models)] * len(trained_models)

    logging.info(f"Ensemble model weights: {model_weights}")

    # Create prediction function
    def ensemble_predict(X):
        if not trained_models:
            raise ValueError("No trained models in ensemble")

        predictions = []
        for model in trained_models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Error in ensemble prediction: {str(e)}")

        if not predictions:
            raise ValueError("All ensemble models failed to predict")

        # Apply weights
        weighted_predictions = [w * pred for w, pred in zip(model_weights, predictions)]

        # Sum predictions
        return sum(weighted_predictions)

    return trained_models, ensemble_predict, model_weights


# ------------------------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------------------------

def evaluate_model(
        model: Sequential,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler: Any = None,
        ensemble_predict_func: Callable = None,
        feature_index: int = 3  # Index of 'close' price in features
) -> Dict:
    """
    Evaluate model performance with enhanced metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        scaler: Fitted scaler
        ensemble_predict_func: Optional ensemble prediction function
        feature_index: Index of close price in feature array

    Returns:
        Dictionary with evaluation metrics
    """
    logging.info("Evaluating model performance")

    # Make predictions
    if ensemble_predict_func is not None:
        try:
            predictions = ensemble_predict_func(X_test)
            logging.info("Using ensemble predictions for evaluation")
        except Exception as e:
            logging.error(f"Ensemble prediction failed: {str(e)}")
            predictions = model.predict(X_test)
    else:
        predictions = model.predict(X_test)

    # Calculate basic metrics
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(mse)

    # Calculate directional accuracy for each future step
    direction_accuracy = []
    for i in range(y_test.shape[1]):
        dir_acc = np.mean(np.sign(y_test[:, i]) == np.sign(predictions[:, i]))
        direction_accuracy.append(float(dir_acc))

    # Overall directional accuracy (first step)
    overall_dir_acc = direction_accuracy[0]

    # Calculate profit factor (simplified)
    # For first step predictions only
    correct_ups = np.logical_and(y_test[:, 0] > 0, predictions[:, 0] > 0)
    correct_downs = np.logical_and(y_test[:, 0] < 0, predictions[:, 0] < 0)

    # Sum the actual returns for correct predictions
    up_returns = np.sum(y_test[:, 0][correct_ups]) if np.any(correct_ups) else 0
    down_returns = -np.sum(y_test[:, 0][correct_downs]) if np.any(correct_downs) else 0

    # Wrong predictions
    wrong_ups = np.logical_and(y_test[:, 0] < 0, predictions[:, 0] > 0)
    wrong_downs = np.logical_and(y_test[:, 0] > 0, predictions[:, 0] < 0)

    # Sum losses from wrong predictions
    wrong_up_losses = -np.sum(y_test[:, 0][wrong_ups]) if np.any(wrong_ups) else 0
    wrong_down_losses = np.sum(y_test[:, 0][wrong_downs]) if np.any(wrong_downs) else 0

    # Calculate profit factor
    total_gains = up_returns + down_returns
    total_losses = wrong_up_losses + wrong_down_losses

    if total_losses > 0:
        profit_factor = total_gains / total_losses
    else:
        profit_factor = float('inf') if total_gains > 0 else 0

    # Get original scale metrics if scaler is available
    scaled_metrics = {}
    if scaler is not None and hasattr(scaler, 'inverse_transform'):
        try:
            # Calculate percentage of profitable trades
            profitable_trades = np.sum(np.logical_or(correct_ups, correct_downs))
            total_trades = np.sum(np.logical_or(np.logical_or(correct_ups, correct_downs),
                                                np.logical_or(wrong_ups, wrong_downs)))

            if total_trades > 0:
                win_rate = profitable_trades / total_trades
            else:
                win_rate = 0

            scaled_metrics = {
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor)
            }

        except Exception as e:
            logging.error(f"Error calculating scaled metrics: {str(e)}")

    # Create evaluation results
    eval_results = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "direction_accuracy": direction_accuracy,
        "overall_direction_accuracy": float(overall_dir_acc),
        "profit_factor": float(profit_factor),
        "scaled_metrics": scaled_metrics
    }

    logging.info(f"Evaluation metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    logging.info(f"Direction accuracy: {overall_dir_acc:.4f}, Profit factor: {profit_factor:.4f}")

    return eval_results


# ------------------------------------------------------------------------------
# TPOT Integration
# ------------------------------------------------------------------------------

def tpot_optimization(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        export_path: str = "tpot_best_pipeline.py"
) -> Any:
    """
    Use TPOT for automated machine learning pipeline optimization.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        export_path: Path to export best pipeline

    Returns:
        TPOT optimizer object or None if TPOT is not available
    """
    if not TPOT_AVAILABLE:
        logging.warning("TPOT not available. Install with: pip install tpot")
        return None

    logging.info("Starting TPOT optimization")

    try:
        # Reshape data for TPOT (flattening sequences)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Focus on first target for TPOT
        y_train_target = y_train[:, 0]
        y_test_target = y_test[:, 0]

        # Initialize TPOT
        tpot_optimizer = TPOTRegressor(
            generations=5,
            population_size=20,
            verbosity=2,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            max_time_mins=60,
            config_dict='TPOT sparse'
        )

        # Fit TPOT
        tpot_optimizer.fit(X_train_flat, y_train_target)

        # Evaluate
        tpot_score = tpot_optimizer.score(X_test_flat, y_test_target)
        logging.info(f"TPOT best pipeline score: {tpot_score}")

        # Export best pipeline
        tpot_optimizer.export(export_path)
        logging.info(f"Best pipeline exported to {export_path}")

        # Evaluate directional accuracy
        y_pred = tpot_optimizer.predict(X_test_flat)
        dir_acc = np.mean(np.sign(y_test_target) == np.sign(y_pred))
        logging.info(f"TPOT directional accuracy: {dir_acc:.4f}")

        return tpot_optimizer

    except Exception as e:
        logging.error(f"Error in TPOT optimization: {str(e)}")
        traceback.print_exc()
        return None


# ------------------------------------------------------------------------------
# Model Saving and Loading
# ------------------------------------------------------------------------------

def save_model_and_metadata(
        model: Sequential,
        scaler: Any,
        metadata: Dict,
        model_dir: str,
        model_name: str,
        ensemble_models: List[Sequential] = None,
        ensemble_weights: List[float] = None
) -> bool:
    """
    Save the trained model, scaler, and metadata for future use.

    Args:
        model: Trained model
        scaler: Fitted scaler
        metadata: Model metadata dictionary
        model_dir: Directory to save model
        model_name: Base name for model files
        ensemble_models: Optional list of ensemble models
        ensemble_weights: Optional list of ensemble model weights

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Save main model
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")

        # Save scaler
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Scaler saved to {scaler_path}")

        # Save ensemble models if provided
        if ensemble_models and ensemble_weights:
            # Save ensemble weights
            weights_path = os.path.join(model_dir, f"{model_name}_ensemble_weights.json")
            with open(weights_path, 'w') as f:
                json.dump(ensemble_weights, f)
            logging.info(f"Ensemble weights saved to {weights_path}")

            # Add ensemble info to metadata
            metadata['ensemble'] = {
                'num_models': len(ensemble_models),
                'weights': ensemble_weights
            }

        # Save metadata
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logging.info(f"Metadata saved to {metadata_path}")

        return True

    except Exception as e:
        logging.error(f"Error saving model and metadata: {str(e)}")
        traceback.print_exc()
        return False


# ------------------------------------------------------------------------------
# Main Training Pipeline
# ------------------------------------------------------------------------------

async def enhanced_train_for_timeframe(
        api_client: BybitClient,
        symbol: str,
        timeframe: str,
        config: Dict
) -> Optional[Dict]:
    """
    Enhanced training pipeline with advanced architectures and ensembles.

    Args:
        api_client: BybitClient instance
        symbol: Trading symbol
        timeframe: Timeframe
        config: Configuration dictionary

    Returns:
        Dictionary with metadata or None if training failed
    """
    logging.info(f"Starting enhanced training pipeline for {symbol} on {timeframe}m timeframe")

    try:
        # Extract configuration
        lookback_period = config.get('lookback_period', DEFAULT_LOOKBACK_PERIOD)
        target_steps = config.get('target_steps', DEFAULT_TARGET_STEPS)
        training_days = config.get('training_days', DEFAULT_TRAINING_DAYS)
        test_size = config.get('test_size', DEFAULT_TEST_SIZE)
        batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
        epochs = config.get('epochs', DEFAULT_EPOCHS)
        patience = config.get('patience', DEFAULT_PATIENCE)
        model_dir = config.get('model_dir', DEFAULT_MODEL_DIR)
        use_ensemble = config.get('use_ensemble', DEFAULT_USE_ENSEMBLE)

        # Create a model name that includes symbol and timeframe
        model_name = f"{symbol.lower()}_{timeframe}m"
        model_path = os.path.join(model_dir, f"{model_name}.h5")

        # Step 1: Fetch historical data
        df = await fetch_historical_data(
            api_client=api_client,
            symbol=symbol,
            timeframe=timeframe,
            days=training_days
        )

        if df.empty:
            logging.error(f"No data available for {symbol} {timeframe}m")
            return None

        # Step 2: Enhanced preprocessing with feature engineering
        X_train, X_test, y_train, y_test, scaler = await preprocess_data(
            df=df,
            lookback_period=lookback_period,
            target_steps=target_steps,
            test_size=test_size
        )

        # Step 3: Hyperparameter tuning
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1]

        model, best_params = hyperparameter_tuning(
            X_train, y_train, X_test, y_test, input_shape, output_dim
        )

        # Step 4: Train final model
        model, history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparams=best_params if best_params else {'batch_size': batch_size, 'epochs': epochs,
                                                         'patience': patience},
            model_path=model_path
        )

        # Step 5: Optional ensemble model
        ensemble_models = None
        ensemble_predict_func = None
        ensemble_weights = None

        if use_ensemble:
            ensemble_models, ensemble_predict_func, ensemble_weights = build_ensemble_model(
                X_train, y_train, X_test, y_test, input_shape, output_dim, model_dir, symbol, timeframe
            )

        # Step 6: Evaluate model
        eval_results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            scaler=scaler,
            ensemble_predict_func=ensemble_predict_func
        )

        # Step 7: Optional TPOT optimization
        tpot_export_path = os.path.join(model_dir, f"{model_name}_tpot_pipeline.py")
        tpot_model = tpot_optimization(X_train, y_train, X_test, y_test, tpot_export_path)

        # Step 8: Save model and metadata
        feature_names = getattr(scaler, 'feature_names', [])

        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback_period": lookback_period,
            "target_steps": target_steps,
            "training_days": training_days,
            "data_rows": len(df),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features": X_train.shape[2],
            "feature_names": feature_names,
            "hyperparameters": best_params,
            "training_history": {
                "final_loss": float(history.get('loss', [0])[-1]),
                "final_val_loss": float(history.get('val_loss', [0])[-1]),
                "epochs_trained": len(history.get('loss', [])),
            },
            "evaluation": eval_results,
            "tpot_used": tpot_model is not None,
            "training_date": datetime.datetime.now().isoformat()
        }

        save_model_and_metadata(
            model=model,
            scaler=scaler,
            metadata=metadata,
            model_dir=model_dir,
            model_name=model_name,
            ensemble_models=ensemble_models,
            ensemble_weights=ensemble_weights
        )

        logging.info(f"Enhanced training completed for {symbol} {timeframe}m")
        return metadata

    except Exception as e:
        logging.error(f"Error in enhanced training for {symbol} {timeframe}m: {str(e)}")
        traceback.print_exc()
        return None


async def train_all_timeframes(
        api_client: BybitClient,
        symbol: str = DEFAULT_SYMBOL,
        timeframes: List[str] = DEFAULT_TIMEFRAMES,
        config: Dict = None
) -> Dict[str, Dict]:
    """
    Train models for all specified timeframes.

    Args:
        api_client: BybitClient instance
        symbol: Trading symbol
        timeframes: List of timeframes
        config: Configuration dictionary

    Returns:
        Dictionary mapping timeframes to metadata
    """
    if config is None:
        config = {}

    logging.info(f"Training models for {symbol} on timeframes: {', '.join(timeframes)}")

    results = {}

    for tf in timeframes:
        try:
            logging.info(f"Starting training for {tf}m timeframe")
            result = await enhanced_train_for_timeframe(
                api_client=api_client,
                symbol=symbol,
                timeframe=tf,
                config=config
            )

            results[tf] = result
            logging.info(f"Completed training for {tf}m timeframe")

        except Exception as e:
            logging.error(f"Error in training pipeline for {tf}m timeframe: {str(e)}")
            results[tf] = {"error": str(e)}

    return results


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

async def main():
    """Main function to run the enhanced training pipeline."""
    # Set up logging
    logger = setup_logging(debug_mode=DEFAULT_DEBUG_MODE)

    # Initialize configuration
    config_manager = ConfigManager()

    # Extract configuration for training
    symbol = config_manager.get("symbol", DEFAULT_SYMBOL)
    timeframes = config_manager.get("timeframes", DEFAULT_TIMEFRAMES)
    use_testnet = config_manager.get("use_testnet", False)

    # Create directories
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

    # Print training configuration
    print("=" * 80)
    print(f"ENHANCED LSTM MODEL TRAINING FOR {symbol}")
    print("=" * 80)
    print(f"Timeframes: {', '.join(timeframes)}m")
    print(f"Lookback Period: {DEFAULT_LOOKBACK_PERIOD} candles")
    print(f"Target Steps: {DEFAULT_TARGET_STEPS} future steps")
    print(f"Training Days: {DEFAULT_TRAINING_DAYS} days")
    print(f"Model Directory: {DEFAULT_MODEL_DIR}")
    print(f"Using Testnet: {'Yes' if use_testnet else 'No'}")
    print(f"Using Ensemble Models: {'Yes' if DEFAULT_USE_ENSEMBLE else 'No'}")
    print("-" * 80)

    # Initialize API client — credentials via environment, never hardcoded.
    import os
    api_client = BybitClient(
        api_key=os.getenv("BYBIT_API_KEY", ""),
        api_secret=os.getenv("BYBIT_API_SECRET", ""),
        use_testnet=use_testnet
    )

    try:
        # Initialize API client
        await api_client.initialize()

        # Training configuration
        training_config = {
            'lookback_period': DEFAULT_LOOKBACK_PERIOD,
            'target_steps': DEFAULT_TARGET_STEPS,
            'training_days': DEFAULT_TRAINING_DAYS,
            'test_size': DEFAULT_TEST_SIZE,
            'batch_size': DEFAULT_BATCH_SIZE,
            'epochs': DEFAULT_EPOCHS,
            'patience': DEFAULT_PATIENCE,
            'model_dir': DEFAULT_MODEL_DIR,
            'data_dir': DEFAULT_DATA_DIR,
            'use_ensemble': DEFAULT_USE_ENSEMBLE
        }

        # Run the training pipeline for all timeframes
        results = await train_all_timeframes(
            api_client=api_client,
            symbol=symbol,
            timeframes=timeframes,
            config=training_config
        )

        # Print summary of results
        print("\nTRAINING RESULTS SUMMARY")
        print("-" * 80)

        for tf, result in results.items():
            if result is None or "error" in result:
                status = "FAILED"
                details = result.get("error", "Unknown error") if result else "No result"
            else:
                status = "SUCCESS"
                eval_metrics = result.get("evaluation", {})
                dir_acc = eval_metrics.get("overall_direction_accuracy", "N/A")

                if isinstance(dir_acc, float):
                    dir_acc = f"{dir_acc:.2%}"

                details = f"MSE: {eval_metrics.get('mse', 'N/A'):.6f}, " \
                          f"Direction Accuracy: {dir_acc}"

            print(f"{tf}m Timeframe: {status} - {details}")

        print("-" * 80)
        print("Training complete.")

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        traceback.print_exc()

    finally:
        # Close the API client
        await api_client.close()


if __name__ == "__main__":
    asyncio.run(main())