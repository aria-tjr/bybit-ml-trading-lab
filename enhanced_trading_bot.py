#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced AI-Driven Automated Futures Trading System for BYBIT Exchange
Version: 3.0
Date: March 9, 2025

This script implements a professional-grade AI-driven trading system for
cryptocurrency futures trading on the BYBIT exchange with improved error handling,
asynchronous operations, and advanced machine learning capabilities.

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
import uuid
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
import aiohttp
import concurrent.futures
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import uuid
import queue
import socket
import functools
import re
import pickle
import hashlib
import hmac

import logging
import numpy as np
from aiohttp import ClientWSTimeout


import self

# Configure matplotlib for non-interactive environments
matplotlib.use('Agg')

# Optional libraries with graceful degradation
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Handle different keras imports
try:
    # Try standalone keras imports first (for keras 3.x)
    from keras.models import Sequential, load_model, save_model
    from keras.layers import Dense, LSTM, Dropout, GRU, BatchNormalization, Bidirectional
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.regularizers import l1_l2

    KERAS_SOURCE = "standalone"
except ImportError:
    try:
        # Fall back to tensorflow.keras (for TF 2.x with built-in keras)
        from tensorflow.keras.models import Sequential, load_model, save_model
        from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, BatchNormalization, Bidirectional
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.regularizers import l1_l2

        KERAS_SOURCE = "tensorflow"
    except ImportError:
        print("WARNING: Neither standalone Keras nor TensorFlow available. AI predictions will use fallback mode.")
        KERAS_SOURCE = None

# Import ML/TA libraries with fallbacks
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not available. Using fallback technical indicators.")

try:
    import sklearn
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. Advanced ML models will be disabled.")

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available. Deep learning models will be disabled.")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not available. Ensemble gradient boosting models will be disabled.")

try:
    import telegram
    from telegram.ext import ApplicationBuilder, CommandHandler

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("WARNING: Telegram libraries not available. Notifications will be disabled.")

try:
    from pybit.unified_trading import HTTP, WebSocket

    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    print("WARNING: PyBit not available. Using fallback HTTP requests for API access.")

# ------------------------------------------------------------------------------
# Constants and Configuration
# ------------------------------------------------------------------------------

# Default API Credentials (load from environment in production)
DEFAULT_BYBIT_API_KEY = ""
DEFAULT_BYBIT_API_SECRET = ""
DEFAULT_TELEGRAM_BOT_TOKEN = ""
DEFAULT_TELEGRAM_CHAT_ID = ""

# Default Trading parameters
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LEVERAGE = 3
DEFAULT_TIMEFRAMES = ["1", "5", "15", "60"]
DEFAULT_MAX_RISK_PER_TRADE = 0.01  # 1% max risk per trade
DEFAULT_TRADING_MODE = "simulation"  # Use "real" for actual trading

# Default configuration parameters
DEFAULT_DEBUG_MODE = False
DEFAULT_TELEGRAM_ENABLED = True
DEFAULT_PERFORMANCE_TRACKING = True
DEFAULT_MAX_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2  # seconds
DEFAULT_CONNECTION_TIMEOUT = 10  # seconds
DEFAULT_CONNECTION_POOL_SIZE = 20
DEFAULT_WEBSOCKET_RECONNECT_DELAY = 5  # seconds

# API endpoints
BYBIT_REST_API_URL = "https://api.bybit.com"
BYBIT_TESTNET_REST_API_URL = "https://api-testnet.bybit.com"
BYBIT_WEBSOCKET_PUBLIC_URL = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WEBSOCKET_PRIVATE_URL = "wss://stream.bybit.com/v5/private"
BYBIT_TESTNET_WEBSOCKET_PUBLIC_URL = "wss://stream-testnet.bybit.com/v5/public"
BYBIT_TESTNET_WEBSOCKET_PRIVATE_URL = "wss://stream-testnet.bybit.com/v5/private"


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def generate_transaction_id():
    """Generate a unique transaction ID for tracking operations."""
    return str(uuid.uuid4())


def safe_float(value, default=0.0):
    """Safely convert a value to float."""
    if value == '' or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def retry_async(max_attempts=3, delay=1):
    """Decorator for retrying async functions with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff
                    logging.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                                    f"Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
            # If we get here, all attempts failed
            logging.error(f"All {max_attempts} attempts failed for {func.__name__}: {str(last_exception)}")
            raise last_exception

        return wrapper

    return decorator


def retry_sync(max_attempts=3, delay=1):
    """Decorator for retrying synchronous functions with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff
                    logging.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                                    f"Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
            # If we get here, all attempts failed
            logging.error(f"All {max_attempts} attempts failed for {func.__name__}: {str(last_exception)}")
            raise last_exception

        return wrapper

    return decorator


class AsyncTaskManager:
    """Manages asynchronous tasks with proper cancellation and error handling."""

    def __init__(self):
        self.tasks = {}
        self.queue = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._processor_task = None

    async def start(self):
        """Start the task processor."""
        if self._processor_task is None:
            self._stop_event.clear()
            self._processor_task = asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stop the task processor and cancel all tasks."""
        if self._processor_task:
            self._stop_event.set()

            # Cancel all pending tasks
            for task_id, task in list(self.tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

            # Clear the queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # Wait for processor to stop
            await self._processor_task
            self._processor_task = None

    async def add_task(self, coro, task_id=None, callback=None):
        """Add a task to the queue."""
        if task_id is None:
            task_id = generate_transaction_id()

        await self.queue.put((task_id, coro, callback))
        return task_id

    async def _process_queue(self):
        """Process tasks from the queue."""
        while not self._stop_event.is_set():
            try:
                # Get a task with timeout to allow for stop checking
                try:
                    task_id, coro, callback = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                # Create and track the task
                task = asyncio.create_task(coro)
                self.tasks[task_id] = task

                # Add done callback to clean up and execute user callback
                task.add_done_callback(
                    lambda t, tid=task_id, cb=callback: self._task_done(tid, t, cb))

                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in task processor: {str(e)}")
                await asyncio.sleep(1)  # Avoid tight loop on recurring errors

    def _task_done(self, task_id, task, callback):
        """Handle task completion and cleanup."""
        # Remove task from tracking
        self.tasks.pop(task_id, None)

        # Execute callback if provided
        if callback:
            try:
                exception = None
                result = None

                if task.cancelled():
                    exception = asyncio.CancelledError("Task was cancelled")
                else:
                    try:
                        result = task.result()
                    except Exception as e:
                        exception = e

                # Use asyncio.create_task to avoid blocking
                asyncio.create_task(self._execute_callback(callback, task_id, result, exception))
            except Exception as e:
                logging.error(f"Error in task completion handler: {str(e)}")

    async def _execute_callback(self, callback, task_id, result, exception):
        """Execute the callback with proper error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(task_id, result, exception)
            else:
                callback(task_id, result, exception)
        except Exception as e:
            logging.error(f"Error in task callback for {task_id}: {str(e)}")


class MessageQueue:
    """Thread-safe message queue for IPC between components."""

    def __init__(self, maxsize=1000):
        self.queue = queue.Queue(maxsize=maxsize)

    def put(self, message, block=True, timeout=None):
        """Put a message on the queue."""
        try:
            self.queue.put(message, block=block, timeout=timeout)
            return True
        except queue.Full:
            logging.warning("Message queue is full, dropping message")
            return False

    def get(self, block=True, timeout=None):
        """Get a message from the queue."""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self):
        """Mark a task as done."""
        self.queue.task_done()

    def empty(self):
        """Check if the queue is empty."""
        return self.queue.empty()

    def size(self):
        """Get the approximate size of the queue."""
        return self.queue.qsize()


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
class EnhancedLogger:
    """Enhanced logging with transaction tracking and structured output."""

    def __init__(self, log_dir="logs", debug_mode=False, app_name="trading_bot"):
        """Initialize the enhanced logger."""
        self.log_dir = log_dir
        self.debug_mode = debug_mode
        self.app_name = app_name
        self.session_id = generate_transaction_id()
        self.log_files = {}

        # Create log directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configure root logger
        self._setup_logger()

    def _setup_logger(self):
        """Configure the logging system."""
        # Generate timestamp for log files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up main log file
        main_log_file = os.path.join(self.log_dir, f"{self.app_name}_{timestamp}.log")
        self.log_files['main'] = main_log_file

        # Configure log level based on debug mode
        log_level = logging.DEBUG if self.debug_mode else logging.INFO

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] [%(name)s] [txid:%(txid)s] %(message)s',
            handlers=[
                logging.FileHandler(main_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        # Set up specialized loggers
        self._setup_specialized_logger('trade', f"{self.app_name}_trades_{timestamp}.log")
        self._setup_specialized_logger('api', f"{self.app_name}_api_{timestamp}.log")
        self._setup_specialized_logger('performance', f"{self.app_name}_performance_{timestamp}.log")
        self._setup_specialized_logger('error', f"{self.app_name}_errors_{timestamp}.log")

        # Add filter to include transaction IDs
        for handler in logging.root.handlers:
            handler.addFilter(self.TransactionFilter())

        logging.info("Logging system initialized", extra={"txid": self.session_id})
        return main_log_file

    def _setup_specialized_logger(self, name, filename):
        """Set up a specialized logger for a specific aspect of the system."""
        logger = logging.getLogger(name)
        log_file = os.path.join(self.log_dir, filename)

        # Store the file path
        self.log_files[name] = log_file

        # Create a file handler
        handler = logging.FileHandler(log_file)

        # Set the format
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [txid:%(txid)s] %(message)s')
        handler.setFormatter(formatter)

        # Add the transaction filter
        handler.addFilter(self.TransactionFilter())

        # Add the handler to the logger
        logger.addHandler(handler)

        # Don't propagate to root logger to avoid duplicate entries
        logger.propagate = False

        return logger

    def get_logger(self, name):
        """Get a named logger with transaction tracking."""
        return logging.getLogger(name)

    def log_trade(self, action, data, txid=None):
        """Log a trade with structured data."""
        logger = logging.getLogger('trade')
        if txid is None:
            txid = generate_transaction_id()

        json_data = json.dumps(data, default=str)
        logger.info(f"{action}: {json_data}", extra={"txid": txid})

    def log_api(self, method, endpoint, request_data, response_data, elapsed, txid=None):
        """Log an API call with request and response details."""
        logger = logging.getLogger('api')
        if txid is None:
            txid = generate_transaction_id()

        log_data = {
            "method": method,
            "endpoint": endpoint,
            "request": request_data,
            "response": response_data,
            "elapsed_ms": int(elapsed * 1000)
        }

        json_data = json.dumps(log_data, default=str)
        logger.info(json_data, extra={"txid": txid})

    def log_error(self, component, error_msg, exception=None, txid=None):
        """Log an error with detailed exception information."""
        logger = logging.getLogger('error')
        if txid is None:
            txid = generate_transaction_id()

        error_data = {
            "component": component,
            "message": error_msg,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_msg": str(exception) if exception else None
        }

        if exception:
            error_data["traceback"] = traceback.format_exc()

        json_data = json.dumps(error_data, default=str)
        logger.error(json_data, extra={"txid": txid})

    def log_performance(self, component, operation, duration, metrics=None, txid=None):
        """Log performance metrics."""
        logger = logging.getLogger('performance')
        if txid is None:
            txid = generate_transaction_id()

        perf_data = {
            "component": component,
            "operation": operation,
            "duration_ms": int(duration * 1000),
            "metrics": metrics or {}
        }

        json_data = json.dumps(perf_data, default=str)
        logger.info(json_data, extra={"txid": txid})

    class TransactionFilter(logging.Filter):
        """Filter to add transaction ID to log records."""

        def filter(self, record):
            if not hasattr(record, 'txid'):
                record.txid = 'default'
            return True


def setup_logging(log_dir="logs", debug_mode=False, app_name="trading_bot"):
    """Set up enhanced logging system."""
    logger = EnhancedLogger(log_dir, debug_mode, app_name)
    return logger


# ------------------------------------------------------------------------------
# Configuration Management
# ------------------------------------------------------------------------------
class ConfigManager:
    """Manages application configuration with validation and persistence."""

    def __init__(self, config_path=None, defaults=None):
        """Initialize configuration manager with defaults."""
        self.config_path = config_path or "config.json"
        self.config = {}
        self.defaults = defaults or {}

        # Load configuration from file or use defaults
        self.load_config()

    def load_config(self):
        """Load configuration from file or environment variables."""
        # Start with defaults
        self.config = self.defaults.copy()

        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                logging.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logging.error(f"Error loading configuration from {self.config_path}: {str(e)}")

        # Load from environment if available
        if DOTENV_AVAILABLE:
            try:
                load_dotenv()
                logging.info("Environment variables loaded from .env file")
            except Exception as e:
                logging.warning(f"Error loading .env file: {str(e)}")

        # Override with environment variables
        self._load_from_env()

        # Validate configuration
        self._validate_config()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mapping = {
            "BYBIT_API_KEY": "bybit_api_key",
            "BYBIT_API_SECRET": "bybit_api_secret",
            "TELEGRAM_BOT_TOKEN": "telegram_bot_token",
            "TELEGRAM_CHAT_ID": "telegram_chat_id",
            "TRADING_SYMBOL": "symbol",
            "TRADING_LEVERAGE": "leverage",
            "TRADING_MODE": "trading_mode",
            "MAX_RISK_PER_TRADE": "max_risk_per_trade",
            "DEBUG_MODE": "debug_mode",
            "USE_TESTNET": "use_testnet"
        }

        for env_var, config_key in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert boolean strings
                if value.lower() in ("true", "yes", "1"):
                    value = True
                elif value.lower() in ("false", "no", "0"):
                    value = False
                # Convert numeric strings
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                    if value.is_integer():
                        value = int(value)

                self.config[config_key] = value

    def _validate_config(self):
        """Validate configuration values."""
        # Ensure required fields are present
        required_fields = []

        # Only require API credentials in real trading mode
        if self.config.get('trading_mode') == 'real':
            required_fields.extend(['bybit_api_key', 'bybit_api_secret'])

        # Check for missing required fields
        missing_fields = [field for field in required_fields if not self.config.get(field)]
        if missing_fields:
            logging.warning(f"Missing required configuration: {', '.join(missing_fields)}")
            if self.config.get('trading_mode') == 'real' and (
                    'bybit_api_key' in missing_fields or 'bybit_api_secret' in missing_fields):
                logging.error("API credentials are required for real trading - switching to simulation mode")
                self.config['trading_mode'] = 'simulation'

        # Ensure numeric fields have appropriate values
        if 'leverage' in self.config:
            self.config['leverage'] = max(1, min(int(self.config['leverage']), 100))

        if 'max_risk_per_trade' in self.config:
            self.config['max_risk_per_trade'] = max(0.001, min(float(self.config['max_risk_per_trade']), 0.1))

    def save_config(self):
        """Save configuration to file."""
        # Don't save sensitive information
        save_config = self.config.copy()
        sensitive_keys = ['bybit_api_key', 'bybit_api_secret', 'telegram_bot_token']
        for key in sensitive_keys:
            if key in save_config:
                save_config[key] = '***REDACTED***'

        try:
            with open(self.config_path, 'w') as f:
                json.dump(save_config, f, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving configuration to {self.config_path}: {str(e)}")
            return False

    def get(self, key, default=None):
        """Get a configuration value with fallback to default."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a configuration value."""
        self.config[key] = value
        return True

    def update(self, new_config):
        """Update multiple configuration values."""
        self.config.update(new_config)
        self._validate_config()
        return True

    def get_api_urls(self):
        """Get the appropriate API URLs based on testnet setting."""
        use_testnet = self.get('use_testnet', False)

        if use_testnet:
            return {
                'rest': BYBIT_TESTNET_REST_API_URL,
                'ws_public': BYBIT_TESTNET_WEBSOCKET_PUBLIC_URL,
                'ws_private': BYBIT_TESTNET_WEBSOCKET_PRIVATE_URL
            }
        else:
            return {
                'rest': BYBIT_REST_API_URL,
                'ws_public': BYBIT_WEBSOCKET_PUBLIC_URL,
                'ws_private': BYBIT_WEBSOCKET_PRIVATE_URL
            }


# ------------------------------------------------------------------------------
# API Client with Async Support
# ------------------------------------------------------------------------------
class BybitClient:
    """Enhanced Bybit API client with async support and connection management."""

    def __init__(self, api_key="", api_secret="", use_testnet=False,
                 timeout=DEFAULT_CONNECTION_TIMEOUT, max_retries=DEFAULT_MAX_RETRY_ATTEMPTS):
        """Initialize the Bybit client."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.timeout = timeout
        self.max_retries = max_retries

        # Determine API URLs
        if use_testnet:
            self.rest_api_url = BYBIT_TESTNET_REST_API_URL
            self.ws_public_url = BYBIT_TESTNET_WEBSOCKET_PUBLIC_URL
            self.ws_private_url = BYBIT_TESTNET_WEBSOCKET_PRIVATE_URL
        else:
            self.rest_api_url = BYBIT_REST_API_URL
            self.ws_public_url = BYBIT_WEBSOCKET_PUBLIC_URL
            self.ws_private_url = BYBIT_WEBSOCKET_PRIVATE_URL

        # Initialize standard client if available
        self.client = None
        if PYBIT_AVAILABLE and api_key and api_secret:
            try:
                self.client = HTTP(
                    testnet=use_testnet,
                    api_key=api_key,
                    api_secret=api_secret
                )
                logging.info("PyBit client initialized")
            except Exception as e:
                logging.error(f"Error initializing PyBit client: {str(e)}")
                self.client = None

        # Initialize async HTTP session
        self.session = None
        self.ws_public = None
        self.ws_private = None
        self.callbacks = {
            'orderbook': [],
            'trade': [],
            'kline': [],
            'ticker': [],
            'position': [],
            'execution': [],
            'order': []
        }

        # Initialize websocket management
        self.ws_task = None
        self.ws_reconnect_delay = DEFAULT_WEBSOCKET_RECONNECT_DELAY
        self.ws_connected = False
        self.ws_stop_event = asyncio.Event()

        # Initialize task manager
        self.task_manager = AsyncTaskManager()

    async def initialize(self):
        """Initialize async client components."""
        # Create ClientSession with connection pooling
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(
            limit=DEFAULT_CONNECTION_POOL_SIZE,
            ttl_dns_cache=300,  # 5-minute DNS cache
            ssl=False  # Let aiohttp handle SSL verification
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "EnhancedBotAsyncClient/3.0"}
        )

        # Start task manager
        await self.task_manager.start()

        logging.info("Async Bybit client initialized")
        return True

    async def close(self):
        """Close all connections and release resources."""
        # Stop websockets
        await self.stop_websockets()

        # Stop task manager
        await self.task_manager.stop()

        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None

        logging.info("Async Bybit client closed")

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def request(self, method, endpoint, params=None, data=None, auth=False):
        """Make an authenticated request to the Bybit API."""
        txid = generate_transaction_id()
        start_time = time.time()

        # Ensure session is initialized
        if not self.session:
            await self.initialize()

        url = f"{self.rest_api_url}{endpoint}"
        headers = {}

        # Add authentication if required
        if auth:
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret are required for authenticated requests")

            timestamp = int(time.time() * 1000)
            params = params or {}

            # Add API key and timestamp to parameters
            params['api_key'] = self.api_key
            params['timestamp'] = timestamp

            # Generate signature
            signature_payload = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Add signature to parameters
            params['sign'] = signature

        try:
            # Make request
            async with self.session.request(
                    method, url, params=params, json=data, headers=headers
            ) as response:
                # Parse response
                if response.content_type == 'application/json':
                    result = await response.json()
                else:
                    result = await response.text()

                # Log API call
                elapsed = time.time() - start_time
                logging.getLogger('api').info(
                    f"{method} {endpoint} - {response.status}",
                    extra={"txid": txid}
                )

                # Check for API errors
                if isinstance(result, dict) and 'retCode' in result and result['retCode'] != 0:
                    error_msg = f"API error: {result.get('retMsg', 'Unknown error')}"
                    logging.error(f"{error_msg} - {endpoint}", extra={"txid": txid})
                    raise Exception(error_msg)

                return result
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.error(f"HTTP error in {endpoint}: {str(e)}", extra={"txid": txid})
            raise
        except Exception as e:
            logging.error(f"Error in API request to {endpoint}: {str(e)}", extra={"txid": txid})
            raise

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def get_klines(self, symbol, interval, limit=500, start_time=None, end_time=None):
        """Get kline/candlestick data."""
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['start'] = int(start_time.timestamp() * 1000)

        if end_time:
            params['end'] = int(end_time.timestamp() * 1000)

        result = await self.request('GET', '/v5/market/kline', params=params)

        # Parse and convert klines to DataFrame
        if result and 'result' in result and 'list' in result['result']:
            klines = result['result']['list']

            if not klines:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            # Convert types
            numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")

            # Sort by timestamp (oldest first)
            df = df.sort_values("timestamp")

            # Set timestamp as index
            df.set_index("timestamp", inplace=True)

            return df

        return pd.DataFrame()

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def get_ticker(self, symbol):
        """Get latest ticker data for a symbol."""
        params = {
            'category': 'linear',
            'symbol': symbol
        }

        result = await self.request('GET', '/v5/market/tickers', params=params)

        if result and 'result' in result and 'list' in result['result']:
            tickers = result['result']['list']
            if tickers:
                ticker = tickers[0]
                return {
                    'symbol': ticker['symbol'],
                    'last_price': float(ticker['lastPrice']),
                    'bid': float(ticker['bid1Price']),
                    'ask': float(ticker['ask1Price']),
                    'volume_24h': float(ticker['volume24h']),
                    'price_24h_change': float(ticker['price24hPcnt']),
                    'high_24h': float(ticker['highPrice24h']),
                    'low_24h': float(ticker['lowPrice24h']),
                    'timestamp': datetime.datetime.now().isoformat()
                }

        return {}

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def get_orderbook(self, symbol, limit=25):
        """Get orderbook data for a symbol."""
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': limit
        }

        result = await self.request('GET', '/v5/market/orderbook', params=params)

        if result and 'result' in result:
            orderbook = result['result']
            return {
                'symbol': symbol,
                'bids': [{"price": float(b[0]), "size": float(b[1])} for b in orderbook.get('b', [])],
                'asks': [{"price": float(a[0]), "size": float(a[1])} for a in orderbook.get('a', [])],
                'timestamp': datetime.datetime.now().isoformat(),
                'update_id': orderbook.get('u', 0)
            }

        return {}

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def get_account_info(self):
        """Get account information."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not set"}

        result = await self.request(
            'GET', '/v5/account/wallet-balance',
            params={'accountType': 'UNIFIED'},
            auth=True
        )

        if result and 'result' in result:
            return result['result']

        return {}

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def get_positions(self, symbol=None):
        """Get current positions."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not set"}

        params = {
            'category': 'linear'
        }

        if symbol:
            params['symbol'] = symbol

        result = await self.request(
            'GET', '/v5/position/list',
            params=params,
            auth=True
        )

        if result and 'result' in result and 'list' in result['result']:
            positions = result['result']['list']
            return [
                {
                    'symbol': pos['symbol'],
                    'side': pos['side'],
                    'size': float(pos['size']),
                    'position_value': float(pos['positionValue']),
                    'entry_price': float(pos['entryPrice']),
                    'leverage': float(pos['leverage']),
                    'unrealized_pnl': float(pos['unrealisedPnl']),
                    'take_profit': pos.get('takeProfit', '0'),
                    'stop_loss': pos.get('stopLoss', '0')
                }
                for pos in positions if float(pos['size']) > 0
            ]

        return []

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not set"}

        data = {
            'category': 'linear',
            'symbol': symbol,
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage)
        }

        result = await self.request(
            'POST', '/v5/position/set-leverage',
            data=data,
            auth=True
        )

        return result

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def place_order(self, symbol, side, order_type, qty,
                          price=None, time_in_force="GoodTillCancel",
                          take_profit=None, stop_loss=None, reduce_only=False,
                          close_on_trigger=False):
        """Place an order."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not set"}

        # Create order data
        data = {
            'category': 'linear',
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'qty': str(qty),
            'timeInForce': time_in_force,
            'reduceOnly': reduce_only,
            'closeOnTrigger': close_on_trigger
        }

        # Add price for limit orders
        if order_type.lower() == 'limit' and price is not None:
            data['price'] = str(price)

        # Add take profit and stop loss if provided
        if take_profit is not None:
            data['takeProfit'] = str(take_profit)

        if stop_loss is not None:
            data['stopLoss'] = str(stop_loss)

        # Generate client order ID
        data['orderLinkId'] = f"bot_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        logging.getLogger('trade').info(
            f"Placing order: {symbol} {side} {order_type} {qty}",
            extra={"order_data": data}
        )

        result = await self.request(
            'POST', '/v5/order/create',
            data=data,
            auth=True
        )

        return result

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def place_conditional_order(self, symbol, side, order_type, qty,
                                      trigger_price, trigger_by="LastPrice",
                                      price=None, time_in_force="GoodTillCancel",
                                      reduce_only=False, close_on_trigger=False):
        """Place a conditional order."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not set"}

        # Create order data
        data = {
            'category': 'linear',
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'qty': str(qty),
            'triggerPrice': str(trigger_price),
            'triggerBy': trigger_by,
            'timeInForce': time_in_force,
            'reduceOnly': reduce_only,
            'closeOnTrigger': close_on_trigger
        }

        # Add price for limit orders
        if order_type.lower() == 'limit' and price is not None:
            data['price'] = str(price)

        # Generate client order ID
        data['orderLinkId'] = f"bot_cond_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        logging.getLogger('trade').info(
            f"Placing conditional order: {symbol} {side} {order_type} {qty} @ {trigger_price}",
            extra={"order_data": data}
        )

        result = await self.request(
            'POST', '/v5/order/create-conditional',
            data=data,
            auth=True
        )

        return result

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def cancel_order(self, symbol, order_id=None, order_link_id=None):
        """Cancel an order."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not set"}

        data = {
            'category': 'linear',
            'symbol': symbol
        }

        if order_id:
            data['orderId'] = order_id
        elif order_link_id:
            data['orderLinkId'] = order_link_id
        else:
            return {"error": "Either order_id or order_link_id must be provided"}

        result = await self.request(
            'POST', '/v5/order/cancel',
            data=data,
            auth=True
        )

        return result

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def get_open_orders(self, symbol=None):
        """Get open orders."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not set"}

        params = {
            'category': 'linear'
        }

        if symbol:
            params['symbol'] = symbol

        result = await self.request(
            'GET', '/v5/order/realtime',
            params=params,
            auth=True
        )

        if result and 'result' in result and 'list' in result['result']:
            orders = result['result']['list']
            return [
                {
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'order_type': order['orderType'],
                    'price': float(order.get('price', '0')),
                    'qty': float(order['qty']),
                    'order_id': order['orderId'],
                    'order_link_id': order.get('orderLinkId', ''),
                    'create_time': datetime.datetime.fromtimestamp(int(order['createdTime']) / 1000).isoformat(),
                    'order_status': order['orderStatus']
                }
                for order in orders
            ]

        return []

    async def start_websockets(self, subscribe_public=True, subscribe_private=True):
        """Start websocket connections."""
        self.ws_stop_event.clear()

        if subscribe_public:
            await self.start_public_websocket()

        if subscribe_private and self.api_key and self.api_secret:
            await self.start_private_websocket()

        # Start the websocket management task
        if not self.ws_task or self.ws_task.done():
            self.ws_task = asyncio.create_task(self._manage_websockets())

        return True

    async def stop_websockets(self):
        """Stop all websocket connections."""
        self.ws_stop_event.set()

        if self.ws_task:
            try:
                await asyncio.wait_for(self.ws_task, timeout=5)
            except asyncio.TimeoutError:
                pass

        # Close websocket connections
        if self.ws_public:
            await self.ws_public.close()
            self.ws_public = None

        if self.ws_private:
            await self.ws_private.close()
            self.ws_private = None

        self.ws_connected = False
        return True

    async def start_public_websocket(self):
        """Start public websocket connection."""
        if self.ws_public:
            await self.ws_public.close()

        try:
            # Connect to public websocket
            self.ws_public = await self.session.ws_connect(
                self.ws_public_url,
                heartbeat=30.0,
                timeout=ClientWSTimeout(ws_close=10)
            )

            # Subscribe to channels
            symbol = self.symbol if hasattr(self, 'symbol') else "BTCUSDT"

            # Form subscription message
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    f"orderbook.25.{symbol}",
                    f"publicTrade.{symbol}",
                    f"tickers.{symbol}",
                    f"kline.1.{symbol}",
                    f"kline.5.{symbol}",
                    f"kline.15.{symbol}",
                    f"kline.60.{symbol}"
                ]
            }

            await self.ws_public.send_json(subscribe_msg)

            # Start message handler
            asyncio.create_task(self._handle_public_messages())

            logging.info("Public websocket connected and subscribed")
            return True

        except Exception as e:
            logging.error(f"Error connecting to public websocket: {str(e)}")
            if self.ws_public:
                await self.ws_public.close()
                self.ws_public = None
            return False

    async def start_private_websocket(self):
        """Start private websocket connection."""
        if not self.api_key or not self.api_secret:
            logging.warning("Cannot start private websocket without API credentials")
            return False

        if self.ws_private:
            await self.ws_private.close()

        try:
            # Connect to private websocket
            self.ws_private = await self.session.ws_connect(
                self.ws_private_url,
                heartbeat=30.0,
                timeout=ClientWSTimeout(ws_close=10)
            )

            # Authenticate
            expires = int((time.time() + 10) * 1000)  # 10 seconds from now
            signature_payload = f"GET/realtime{expires}"
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            auth_message = {
                "op": "auth",
                "args": [self.api_key, expires, signature]
            }

            await self.ws_private.send_json(auth_message)

            # Wait for auth response
            auth_response = await self.ws_private.receive_json(timeout=5.0)
            if auth_response.get("success"):
                # Subscribe to channels
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [
                        "position",
                        "execution",
                        "order"
                    ]
                }

                await self.ws_private.send_json(subscribe_msg)

                # Start message handler
                asyncio.create_task(self._handle_private_messages())

                logging.info("Private websocket connected, authenticated, and subscribed")
                return True
            else:
                logging.error(f"Private websocket authentication failed: {auth_response}")
                await self.ws_private.close()
                self.ws_private = None
                return False

        except Exception as e:
            logging.error(f"Error connecting to private websocket: {str(e)}")
            if self.ws_private:
                await self.ws_private.close()
                self.ws_private = None
            return False

    async def _handle_public_messages(self):
        """Handle messages from public websocket."""
        if not self.ws_public:
            return

        try:
            async for msg in self.ws_public:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Handle different types of messages
                    if 'topic' in data:
                        topic = data['topic']

                        # Handle orderbook updates
                        if topic.startswith('orderbook'):
                            await self._process_orderbook(data)

                        # Handle trade updates
                        elif topic.startswith('publicTrade'):
                            await self._process_trade(data)

                        # Handle kline updates
                        elif topic.startswith('kline'):
                            await self._process_kline(data)

                        # Handle ticker updates
                        elif topic.startswith('tickers'):
                            await self._process_ticker(data)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logging.error(f"Public websocket error: {msg}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logging.warning("Public websocket closed")
                    break

        except asyncio.CancelledError:
            logging.info("Public websocket handler cancelled")

        except Exception as e:
            logging.error(f"Error in public websocket handler: {str(e)}")

        finally:
            if self.ws_public and not self.ws_public.closed:
                await self.ws_public.close()
            self.ws_public = None

    async def _handle_private_messages(self):
        """Handle messages from private websocket."""
        if not self.ws_private:
            return

        try:
            async for msg in self.ws_private:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Handle different types of messages
                    if 'topic' in data:
                        topic = data['topic']

                        # Handle position updates
                        if topic == 'position':
                            await self._process_position(data)

                        # Handle execution updates
                        elif topic == 'execution':
                            await self._process_execution(data)

                        # Handle order updates
                        elif topic == 'order':
                            await self._process_order(data)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logging.error(f"Private websocket error: {msg}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logging.warning("Private websocket closed")
                    break

        except asyncio.CancelledError:
            logging.info("Private websocket handler cancelled")

        except Exception as e:
            logging.error(f"Error in private websocket handler: {str(e)}")

        finally:
            if self.ws_private and not self.ws_private.closed:
                await self.ws_private.close()
            self.ws_private = None

    async def _process_orderbook(self, data):
        """Process orderbook update."""
        try:
            for callback in self.callbacks.get('orderbook', []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            logging.error(f"Error processing orderbook update: {str(e)}")

    async def _process_trade(self, data):
        """Process trade update."""
        try:
            for callback in self.callbacks.get('trade', []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            logging.error(f"Error processing trade update: {str(e)}")

    async def _process_kline(self, data):
        """Process kline update."""
        try:
            for callback in self.callbacks.get('kline', []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            logging.error(f"Error processing kline update: {str(e)}")

    async def _process_ticker(self, data):
        """Process ticker update."""
        try:
            for callback in self.callbacks.get('ticker', []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            logging.error(f"Error processing ticker update: {str(e)}")

    async def _process_position(self, data):
        """Process position update."""
        try:
            for callback in self.callbacks.get('position', []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            logging.error(f"Error processing position update: {str(e)}")

    async def _process_execution(self, data):
        """Process execution update."""
        try:
            for callback in self.callbacks.get('execution', []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            logging.error(f"Error processing execution update: {str(e)}")

    async def _process_order(self, data):
        """Process order update."""
        try:
            for callback in self.callbacks.get('order', []):
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
        except Exception as e:
            logging.error(f"Error processing order update: {str(e)}")

    async def _manage_websockets(self):
        """Manage websocket connections and handle reconnection."""
        while not self.ws_stop_event.is_set():
            try:
                # Check public websocket
                if not self.ws_public or self.ws_public.closed:
                    logging.info("Public websocket disconnected, reconnecting...")
                    await self.start_public_websocket()

                # Check private websocket
                if self.api_key and self.api_secret and (not self.ws_private or self.ws_private.closed):
                    logging.info("Private websocket disconnected, reconnecting...")
                    await self.start_private_websocket()

                # Wait before next check
                await asyncio.sleep(self.ws_reconnect_delay)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logging.error(f"Error in websocket manager: {str(e)}")
                await asyncio.sleep(self.ws_reconnect_delay)

    def add_callback(self, channel, callback):
        """Add a callback for a specific channel."""
        if channel in self.callbacks:
            self.callbacks[channel].append(callback)
            return True
        return False

    def remove_callback(self, channel, callback):
        """Remove a callback for a specific channel."""
        if channel in self.callbacks and callback in self.callbacks[channel]:
            self.callbacks[channel].remove(callback)
            return True
        return False


# ------------------------------------------------------------------------------
# Market Data Management
# ------------------------------------------------------------------------------
class MarketDataManager:
    """Enhanced market data manager with async support and caching."""

    def __init__(self, api_client, symbol, timeframes=None):
        """Initialize the market data manager."""
        self.api_client = api_client
        self.symbol = symbol
        self.timeframes = timeframes or ["1", "5", "15", "60"]

        # Initialize data storage
        self.data = {tf: pd.DataFrame() for tf in self.timeframes}
        self.orderbook = {
            "bids": [],
            "asks": [],
            "timestamp": None
        }
        self.ticker = {
            "last": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "timestamp": None
        }

        # Initialize update tracking
        self.last_update = {tf: None for tf in self.timeframes}
        self.update_interval = {
            "1": 60,  # 1 minute
            "5": 300,  # 5 minutes
            "15": 900,  # 15 minutes
            "60": 3600,  # 60 minutes
            "240": 14400  # 4 hours
        }

        # Initialize cache
        self._cache_dir = "data_cache"
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        # Set up websocket callbacks
        self.api_client.add_callback('orderbook', self._handle_orderbook_update)
        self.api_client.add_callback('ticker', self._handle_ticker_update)
        self.api_client.add_callback('kline', self._handle_kline_update)

        logging.info(f"Market data manager initialized for {symbol}")

    async def initialize(self):
        """Initialize the market data manager by loading initial data."""
        tasks = []
        for tf in self.timeframes:
            tasks.append(self.fetch_historical_data(tf))

        # Wait for all historical data to be fetched
        await asyncio.gather(*tasks)

        # Fetch orderbook and ticker
        await self.fetch_orderbook()
        await self.fetch_ticker()

        logging.info(f"Market data initialized for {self.symbol}")
        return True

    def _get_cache_path(self, timeframe):
        """Get the cache file path for a timeframe."""
        return os.path.join(
            self._cache_dir,
            f"{self.symbol}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        )

    def _load_from_cache(self, timeframe):
        """Load data from cache for a timeframe."""
        cache_path = self._get_cache_path(timeframe)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                # Check if the data is recent enough
                if not df.empty and df.index[-1] >= pd.Timestamp.now() - pd.Timedelta(hours=6):
                    logging.info(f"Loaded {len(df)} rows from cache for {self.symbol} {timeframe}m")
                    return df
            except Exception as e:
                logging.warning(f"Error loading from cache: {str(e)}")
        return None

    def _save_to_cache(self, timeframe, df):
        """Save data to cache for a timeframe."""
        if df is None or df.empty:
            return False

        try:
            cache_path = self._get_cache_path(timeframe)
            df.to_csv(cache_path)
            logging.info(f"Saved {len(df)} rows to cache for {self.symbol} {timeframe}m")
            return True
        except Exception as e:
            logging.warning(f"Error saving to cache: {str(e)}")
            return False

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def fetch_historical_data(self, timeframe, limit=500):
        """Fetch historical kline data for a specific timeframe."""
        # Try loading from cache first
        cached_data = self._load_from_cache(timeframe)
        if cached_data is not None and not cached_data.empty:
            self.data[timeframe] = cached_data
            self.last_update[timeframe] = datetime.datetime.now()
            return cached_data

        try:
            df = await self.api_client.get_klines(self.symbol, timeframe, limit)

            if not df.empty:
                # Store data
                self.data[timeframe] = df
                self.last_update[timeframe] = datetime.datetime.now()

                # Save to cache
                self._save_to_cache(timeframe, df)

                logging.info(f"Fetched {len(df)} historical klines for {self.symbol} {timeframe}m")
                return df
            else:
                logging.warning(f"No historical data received for {self.symbol} {timeframe}m")
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"Error fetching historical data for {timeframe}m: {str(e)}")
            return pd.DataFrame()

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def fetch_ticker(self):
        """Fetch latest ticker data."""
        try:
            ticker = await self.api_client.get_ticker(self.symbol)

            if ticker:
                self.ticker = {
                    "last": ticker.get('last_price', 0.0),
                    "bid": ticker.get('bid', 0.0),
                    "ask": ticker.get('ask', 0.0),
                    "timestamp": datetime.datetime.now()
                }
                return self.ticker
            else:
                logging.warning(f"No ticker data received for {self.symbol}")
                return self.ticker

        except Exception as e:
            logging.error(f"Error fetching ticker: {str(e)}")
            return self.ticker

    @retry_async(max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS, delay=DEFAULT_RETRY_DELAY)
    async def fetch_orderbook(self, limit=25):
        """Fetch orderbook data."""
        try:
            orderbook = await self.api_client.get_orderbook(self.symbol, limit)

            if orderbook:
                self.orderbook = {
                    "bids": orderbook.get('bids', []),
                    "asks": orderbook.get('asks', []),
                    "timestamp": datetime.datetime.now()
                }
                return self.orderbook
            else:
                logging.warning(f"No orderbook data received for {self.symbol}")
                return self.orderbook

        except Exception as e:
            logging.error(f"Error fetching orderbook: {str(e)}")
            return self.orderbook

    async def update_all(self):
        """Update data for all timeframes."""
        tasks = []
        for tf in self.timeframes:
            if self._should_update(tf):
                tasks.append(self.fetch_historical_data(tf))

        if tasks:
            await asyncio.gather(*tasks)

        # Also update ticker and orderbook
        await asyncio.gather(
            self.fetch_ticker(),
            self.fetch_orderbook()
        )

        return {tf: self.data[tf] for tf in self.timeframes}

    def _should_update(self, timeframe):
        """Check if a timeframe should be updated based on its interval."""
        if timeframe not in self.last_update or self.last_update[timeframe] is None:
            return True

        # Get update interval in seconds
        interval = self.update_interval.get(timeframe, 300)  # Default to 5 minutes

        # Check if enough time has passed since last update
        time_since_update = (datetime.datetime.now() - self.last_update[timeframe]).total_seconds()
        return time_since_update >= interval

    def get_data(self, timeframe):
        """Get the latest data for a specific timeframe."""
        return self.data.get(timeframe, pd.DataFrame())

    def get_all_data(self):
        """Get the latest data for all timeframes."""
        return self.data

    def get_latest_price(self):
        """Get the latest mid-price."""
        if self.ticker["bid"] > 0 and self.ticker["ask"] > 0:
            return (self.ticker["bid"] + self.ticker["ask"]) / 2
        return self.ticker["last"]

    def get_bid_ask_spread(self):
        """Get the current bid-ask spread."""
        if self.ticker["bid"] > 0 and self.ticker["ask"] > 0:
            return self.ticker["ask"] - self.ticker["bid"]
        return 0.0

    async def _handle_orderbook_update(self, data):
        """Handle orderbook update from websocket."""
        try:
            if 'data' in data:
                orderbook_data = data['data']

                self.orderbook = {
                    "bids": [{"price": float(b[0]), "size": float(b[1])} for b in orderbook_data.get('b', [])],
                    "asks": [{"price": float(a[0]), "size": float(a[1])} for a in orderbook_data.get('a', [])],
                    "timestamp": datetime.datetime.now()
                }
        except Exception as e:
            logging.error(f"Error handling orderbook update: {str(e)}")

    async def _handle_ticker_update(self, data):
        """Handle ticker update from websocket."""
        try:
            if 'data' in data:
                ticker_data = data['data']

                self.ticker = {
                    "last": float(ticker_data.get('lastPrice', 0.0)),
                    "bid": float(ticker_data.get('bid1Price', 0.0)),
                    "ask": float(ticker_data.get('ask1Price', 0.0)),
                    "timestamp": datetime.datetime.now()
                }
        except Exception as e:
            logging.error(f"Error handling ticker update: {str(e)}")

    async def _handle_kline_update(self, data):
        """Handle kline update from websocket."""
        try:
            if 'data' in data and 'topic' in data:
                # Extract timeframe from topic
                topic_parts = data['topic'].split('.')
                if len(topic_parts) >= 3:
                    timeframe = topic_parts[1]

                    if timeframe in self.timeframes:
                        kline_data = data['data'][0]

                        # Create a single-row DataFrame
                        df = pd.DataFrame([{
                            "timestamp": pd.to_datetime(int(kline_data['start']), unit='ms'),
                            "open": float(kline_data['open']),
                            "high": float(kline_data['high']),
                            "low": float(kline_data['low']),
                            "close": float(kline_data['close']),
                            "volume": float(kline_data['volume']),
                            "turnover": float(kline_data['turnover'])
                        }])

                        # Set timestamp as index
                        df.set_index("timestamp", inplace=True)

                        # Update the DataFrame
                        if not self.data[timeframe].empty:
                            # Check if the timestamp already exists
                            if df.index[0] in self.data[timeframe].index:
                                # Update the existing row
                                self.data[timeframe].loc[df.index[0]] = df.iloc[0]
                            else:
                                # Append the new row
                                self.data[timeframe] = pd.concat([self.data[timeframe], df])
                                # Sort by timestamp
                                self.data[timeframe] = self.data[timeframe].sort_index()
                        else:
                            self.data[timeframe] = df
        except Exception as e:
            logging.error(f"Error handling kline update: {str(e)}")


# ------------------------------------------------------------------------------
# Technical Analysis Indicators
# ------------------------------------------------------------------------------
class TechnicalIndicators:
    """Calculate technical indicators with fallbacks for when TA-Lib is not available."""

    @staticmethod
    def calculate_indicators(df):
        """Calculate all technical indicators for a DataFrame."""
        if df is None or df.empty:
            return df

        # Create a copy to avoid modifying the original
        df_result = df.copy()

        # Calculate indicators
        df_result = TechnicalIndicators.add_moving_averages(df_result)
        df_result = TechnicalIndicators.add_bollinger_bands(df_result)
        df_result = TechnicalIndicators.add_rsi(df_result)
        df_result = TechnicalIndicators.add_macd(df_result)
        df_result = TechnicalIndicators.add_stochastic(df_result)
        df_result = TechnicalIndicators.add_atr(df_result)
        df_result = TechnicalIndicators.add_volume_indicators(df_result)
        df_result = TechnicalIndicators.add_momentum_indicators(df_result)

        return df_result

    @staticmethod
    def add_moving_averages(df):
        """Add various moving averages to the DataFrame."""
        if TALIB_AVAILABLE:
            # Use TA-Lib for EMA calculations
            try:
                df['ema_9'] = talib.EMA(df['close'].values, timeperiod=9)
                df['ema_21'] = talib.EMA(df['close'].values, timeperiod=21)
                df['ema_50'] = talib.EMA(df['close'].values, timeperiod=50)
                df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)
                df['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
                df['sma_30'] = talib.SMA(df['close'].values, timeperiod=30)
            except Exception as e:
                logging.error(f"Error calculating moving averages with TA-Lib: {str(e)}")
                # Fall back to pandas implementation
                df = TechnicalIndicators._calculate_mas_pandas(df)
        else:
            # Use pandas implementation
            df = TechnicalIndicators._calculate_mas_pandas(df)

        # Calculate price/MA ratios
        for ma in ['ema_9', 'ema_21', 'ema_50', 'ema_200']:
            if ma in df.columns:
                df[f'price_{ma}_ratio'] = df['close'] / df[ma]

        return df

    @staticmethod
    def _calculate_mas_pandas(df):
        """Calculate moving averages using pandas."""
        try:
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_30'] = df['close'].rolling(window=30).mean()
        except Exception as e:
            logging.error(f"Error calculating moving averages with pandas: {str(e)}")

        return df

    @staticmethod
    def add_bollinger_bands(df, period=20, std_dev=2):
        """Add Bollinger Bands to the DataFrame."""
        if TALIB_AVAILABLE:
            try:
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                    df['close'].values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
                )
            except Exception as e:
                logging.error(f"Error calculating Bollinger Bands with TA-Lib: {str(e)}")
                # Fall back to pandas implementation
                df = TechnicalIndicators._calculate_bbands_pandas(df, period, std_dev)
        else:
            # Use pandas implementation
            df = TechnicalIndicators._calculate_bbands_pandas(df, period, std_dev)

        # Calculate BB width
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        return df

    @staticmethod
    def _calculate_bbands_pandas(df, period=20, std_dev=2):
        """Calculate Bollinger Bands using pandas."""
        try:
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands with pandas: {str(e)}")

        return df

    @staticmethod
    def add_rsi(df, period=14):
        """Add Relative Strength Index to the DataFrame."""
        if TALIB_AVAILABLE:
            try:
                df['rsi'] = talib.RSI(df['close'].values, timeperiod=period)
            except Exception as e:
                logging.error(f"Error calculating RSI with TA-Lib: {str(e)}")
                # Fall back to pandas implementation
                df = TechnicalIndicators._calculate_rsi_pandas(df, period)
        else:
            # Use pandas implementation
            df = TechnicalIndicators._calculate_rsi_pandas(df, period)

        return df

    @staticmethod
    def _calculate_rsi_pandas(df, period=14):
        """Calculate RSI using pandas."""
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logging.error(f"Error calculating RSI with pandas: {str(e)}")

        return df

    @staticmethod
    def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """Add MACD to the DataFrame."""
        if TALIB_AVAILABLE:
            try:
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                    df['close'].values, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period
                )
            except Exception as e:
                logging.error(f"Error calculating MACD with TA-Lib: {str(e)}")
                # Fall back to pandas implementation
                df = TechnicalIndicators._calculate_macd_pandas(df, fast_period, slow_period, signal_period)
        else:
            # Use pandas implementation
            df = TechnicalIndicators._calculate_macd_pandas(df, fast_period, slow_period, signal_period)

        return df

    @staticmethod
    def _calculate_macd_pandas(df, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD using pandas."""
        try:
            df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Drop temporary columns
            df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True, errors='ignore')
        except Exception as e:
            logging.error(f"Error calculating MACD with pandas: {str(e)}")

        return df

    @staticmethod
    def add_stochastic(df, k_period=14, d_period=3):
        """Add Stochastic Oscillator to the DataFrame."""
        if TALIB_AVAILABLE:
            try:
                df['stoch_k'], df['stoch_d'] = talib.STOCH(
                    df['high'].values, df['low'].values, df['close'].values,
                    fastk_period=k_period, slowk_period=d_period, slowk_matype=0, slowd_period=d_period, slowd_matype=0
                )
            except Exception as e:
                logging.error(f"Error calculating Stochastic with TA-Lib: {str(e)}")
                # Fall back to pandas implementation
                df = TechnicalIndicators._calculate_stochastic_pandas(df, k_period, d_period)
        else:
            # Use pandas implementation
            df = TechnicalIndicators._calculate_stochastic_pandas(df, k_period, d_period)

        return df

    @staticmethod
    def _calculate_stochastic_pandas(df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator using pandas."""
        try:
            # Calculate %K
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))

            # Calculate %D
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        except Exception as e:
            logging.error(f"Error calculating Stochastic with pandas: {str(e)}")

        return df

    @staticmethod
    def add_atr(df, period=14):
        """Add Average True Range to the DataFrame."""
        if TALIB_AVAILABLE:
            try:
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            except Exception as e:
                logging.error(f"Error calculating ATR with TA-Lib: {str(e)}")
                # Fall back to pandas implementation
                df = TechnicalIndicators._calculate_atr_pandas(df, period)
        else:
            # Use pandas implementation
            df = TechnicalIndicators._calculate_atr_pandas(df, period)

        return df

    @staticmethod
    def _calculate_atr_pandas(df, period=14):
        """Calculate Average True Range using pandas."""
        try:
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=period).mean()

            # Drop temporary columns
            df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True, errors='ignore')
        except Exception as e:
            logging.error(f"Error calculating ATR with pandas: {str(e)}")

        return df

    @staticmethod
    def add_volume_indicators(df):
        """Add volume-based indicators to the DataFrame."""
        try:
            # Volume EMA
            df['volume_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ema']

            # On-Balance Volume (OBV)
            if TALIB_AVAILABLE:
                try:
                    df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
                except Exception:
                    df = TechnicalIndicators._calculate_obv_pandas(df)
            else:
                df = TechnicalIndicators._calculate_obv_pandas(df)

            # Money Flow Index (MFI)
            if TALIB_AVAILABLE:
                try:
                    df['mfi'] = talib.MFI(
                        df['high'].values, df['low'].values,
                        df['close'].values, df['volume'].values,
                        timeperiod=14
                    )
                except Exception:
                    pass  # Skip if error, MFI is not critical

            # Chaikin Money Flow (simple implementation)
            df['mf_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            df['money_flow_volume'] = df['mf_multiplier'] * df['volume']
            df['cmf'] = df['money_flow_volume'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

            # Clean up temporary columns
            df.drop(['mf_multiplier', 'money_flow_volume'], axis=1, inplace=True, errors='ignore')

        except Exception as e:
            logging.error(f"Error calculating volume indicators: {str(e)}")

        return df

    @staticmethod
    def _calculate_obv_pandas(df):
        """Calculate On-Balance Volume using pandas."""
        try:
            df['obv'] = 0
            df.loc[1:, 'obv'] = ((df['close'] > df['close'].shift(1)).astype(int) * 2 - 1) * df['volume']
            df['obv'] = df['obv'].cumsum()
        except Exception as e:
            logging.error(f"Error calculating OBV with pandas: {str(e)}")

        return df

    @staticmethod
    def add_momentum_indicators(df):
        """Add momentum indicators to the DataFrame."""
        # Rate of Change (ROC)
        if TALIB_AVAILABLE:
            try:
                df['roc'] = talib.ROC(df['close'].values, timeperiod=10)
                df['mom'] = talib.MOM(df['close'].values, timeperiod=10)
                df['willr'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            except Exception as e:
                logging.error(f"Error calculating momentum indicators with TA-Lib: {str(e)}")
                # Fall back to pandas implementation
                df = TechnicalIndicators._calculate_momentum_pandas(df)
        else:
            # Use pandas implementation
            df = TechnicalIndicators._calculate_momentum_pandas(df)

        return df

    @staticmethod
    def _calculate_momentum_pandas(df):
        """Calculate momentum indicators using pandas."""
        try:
            # Rate of Change (ROC)
            df['roc'] = df['close'].pct_change(periods=10) * 100

            # Momentum
            df['mom'] = df['close'] - df['close'].shift(10)

            # Williams %R (approximation)
            period = 14
            df['highest_high'] = df['high'].rolling(window=period).max()
            df['lowest_low'] = df['low'].rolling(window=period).min()
            df['willr'] = -100 * (df['highest_high'] - df['close']) / (df['highest_high'] - df['lowest_low'])

            # Clean up temporary columns
            df.drop(['highest_high', 'lowest_low'], axis=1, inplace=True, errors='ignore')

        except Exception as e:
            logging.error(f"Error calculating momentum indicators with pandas: {str(e)}")

        return df


# ------------------------------------------------------------------------------
# Enhanced AI Predictive Models
# ------------------------------------------------------------------------------
class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        محاسبه میانگین و انحراف معیار داده‌های X.
        پارامتر:
            X: آرایه‌ای از داده‌ها (مثلاً numpy array)
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # جلوگیری از تقسیم بر صفر
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        """
        داده‌ها را با استفاده از میانگین و انحراف معیار محاسبه‌شده مقیاس‌بندی می‌کند.
        """
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        ترکیبی از fit و transform.
        """
        return self.fit(X).transform(X)


class EnhancedAIModel:
    """Advanced AI model with multiple algorithms and feature engineering."""

    def __init__(self, symbol, timeframes, model_dir="models", debug_mode=False):
        """Initialize the AI model."""
        self.symbol = symbol
        self.timeframes = timeframes
        self.model_dir = model_dir
        self.debug_mode = debug_mode

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Model parameters
        self.look_back = 60  # How many previous time points to use for prediction
        self.target_steps = 5  # How many steps ahead to predict
        self.prediction_threshold = 0.005  # 0.5% price change threshold for signals
        self.confidence_threshold = 0.6  # Minimum confidence for taking action

        # Dictionary to store models for each timeframe
        self.models = {}
        self.ensemble_models = {}
        self.feature_importance = {}
        self.scalers = {}

        # Model performance tracking
        self.prediction_history = []
        self.accuracy_metrics = {tf: {"rmse": [], "mae": [], "r2": []} for tf in timeframes}
        self.last_training = {tf: None for tf in timeframes}

        # Feature engineering pipeline
        self.feature_engineering = FeatureEngineering()

        # Initialize models for each timeframe
        for tf in self.timeframes:
            self._load_or_initialize_models(tf)

        logging.info(f"Enhanced AI model initialized for {symbol}")

    def _load_or_initialize_models(self, timeframe):
        """Load existing models or initialize new ones for a timeframe."""
        # Paths for different model types
        lstm_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_lstm.h5")
        xgboost_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_xgb.json")
        rf_path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_rf.pkl")

        models = {}

        # Try to load LSTM model if available
        if TF_AVAILABLE and os.path.exists(lstm_path):
            try:
                models['lstm'] = load_model(lstm_path)
                logging.info(f"Loaded LSTM model for {timeframe}m timeframe")
            except Exception as e:
                logging.error(f"Error loading LSTM model: {str(e)}")
                models['lstm'] = None

        # Try to load XGBoost model if available
        if XGBOOST_AVAILABLE and os.path.exists(xgboost_path):
            try:
                models['xgboost'] = xgb.Booster()
                models['xgboost'].load_model(xgboost_path)
                logging.info(f"Loaded XGBoost model for {timeframe}m timeframe")
            except Exception as e:
                logging.error(f"Error loading XGBoost model: {str(e)}")
                models['xgboost'] = None
        else:
            models['xgboost'] = None

        # Try to load RandomForest model if available
        if SKLEARN_AVAILABLE and os.path.exists(rf_path):
            try:
                with open(rf_path, 'rb') as f:
                    models['random_forest'] = pickle.load(f)
                logging.info(f"Loaded RandomForest model for {timeframe}m timeframe")
            except Exception as e:
                logging.error(f"Error loading RandomForest model: {str(e)}")
                models['random_forest'] = None
        else:
            models['random_forest'] = None

        # Store models for this timeframe
        self.models[timeframe] = models

        # Initialize scaler based on scikit-learn availability
        try:
            if SKLEARN_AVAILABLE:
                from sklearn.preprocessing import StandardScaler
                self.scalers[timeframe] = StandardScaler()
            else:
                raise ImportError("scikit-learn not available")
        except ImportError as e:
            logging.warning(f"{e}. Using SimpleScaler fallback.")
            self.scalers[timeframe] = SimpleScaler()

    def preprocess_data(self, data, timeframe):
        """Preprocess market data for AI model prediction."""
        if data is None or data.empty or len(data) < self.look_back + 10:
            logging.warning(f"Insufficient data for preprocessing ({timeframe}m)")
            return None, None

        try:
            # Calculate technical indicators
            df = TechnicalIndicators.calculate_indicators(data.copy())

            # Apply feature engineering
            features_df = self.feature_engineering.transform(df)

            # Handle NaN values
            features_df.dropna(inplace=True)

            # Ensure we have enough data after preprocessing
            if len(features_df) < self.look_back + 10:
                logging.warning(f"Insufficient data after preprocessing ({timeframe}m)")
                return None, None

            # Select features and target
            feature_cols = [col for col in features_df.columns
                            if col not in ['open', 'high', 'low', 'close', 'volume', 'turnover']]

            # Create target variables (future price changes)
            for i in range(1, self.target_steps + 1):
                features_df[f'future_return_{i}'] = features_df['close'].pct_change(i).shift(-i)

            # Drop rows with NaN values
            features_df.dropna(inplace=True)

            # Extract features and targets
            X = features_df[feature_cols].values
            y = features_df[[f'future_return_{i}' for i in range(1, self.target_steps + 1)]].values

            # Scale features
            X_scaled = self.scalers[timeframe].fit_transform(X)

            # Create sequences for LSTM
            X_seq = []
            y_seq = []
            for i in range(len(X_scaled) - self.look_back):
                X_seq.append(X_scaled[i:i + self.look_back])
                y_seq.append(y[i + self.look_back - 1])

            if not X_seq:
                logging.warning(f"No sequences generated after preprocessing ({timeframe}m)")
                return None, None

            return np.array(X_seq), np.array(y_seq)

        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            traceback.print_exc()
            return None, None

    def predict(self, data, timeframe):
        """Make predictions using the trained models for a timeframe."""
        logging.info(f"Making prediction for {self.symbol} on {timeframe}m timeframe")

        try:
            # Handle empty or insufficient data
            if data is None or data.empty or len(data) < self.look_back + 10:
                logging.warning(f"Insufficient data for prediction on {timeframe}m timeframe")
                return self._generate_fallback_prediction(timeframe)

            # Generate a fallback prediction in debug mode
            if self.debug_mode:
                return self._generate_debug_prediction(data, timeframe)

            # Get models for this timeframe
            models = self.models.get(timeframe, {})

            # If no models are available, return a fallback prediction
            if not models or all(model is None for model in models.values()):
                logging.warning(f"No models available for {timeframe}m timeframe")
                return self._generate_fallback_prediction(timeframe)

            # Preprocess data
            X_seq, _ = self.preprocess_data(data, timeframe)

            if X_seq is None or len(X_seq) == 0:
                return self._generate_fallback_prediction(timeframe)

            # Use the most recent sequence for prediction
            X_pred = X_seq[-1:]

            # Make predictions with each available model
            predictions = {}

            # LSTM prediction
            if models.get('lstm') is not None and TF_AVAILABLE:
                try:
                    lstm_pred = models['lstm'].predict(X_pred, verbose=0)[0]
                    predictions['lstm'] = lstm_pred
                except Exception as e:
                    logging.error(f"Error making LSTM prediction: {str(e)}")

            # XGBoost prediction
            if models.get('xgboost') is not None and XGBOOST_AVAILABLE:
                try:
                    # Reshape for XGBoost (flatten the sequence)
                    xgb_X = X_pred.reshape(X_pred.shape[0], -1)
                    dmatrix = xgb.DMatrix(xgb_X)
                    xgb_pred = models['xgboost'].predict(dmatrix)

                    # Reshape to match target dimensions if needed
                    if xgb_pred.ndim == 1:
                        xgb_pred = xgb_pred.reshape(1, -1)[0]
                    elif xgb_pred.ndim > 2:
                        xgb_pred = xgb_pred.reshape(1, -1)[0]

                    # Ensure correct shape
                    if len(xgb_pred) == self.target_steps:
                        predictions['xgboost'] = xgb_pred
                    else:
                        logging.warning(
                            f"XGBoost prediction shape mismatch. Expected {self.target_steps}, got {len(xgb_pred)}")
                except Exception as e:
                    logging.error(f"Error making XGBoost prediction: {str(e)}")

            # RandomForest prediction
            if models.get('random_forest') is not None and SKLEARN_AVAILABLE:
                try:
                    # Reshape for RandomForest (flatten the sequence)
                    rf_X = X_pred.reshape(X_pred.shape[0], -1)
                    rf_pred = models['random_forest'].predict(rf_X)[0]
                    predictions['random_forest'] = rf_pred
                except Exception as e:
                    logging.error(f"Error making RandomForest prediction: {str(e)}")

            # If no predictions were made, return a fallback prediction
            if not predictions:
                return self._generate_fallback_prediction(timeframe)

            # Combine predictions (ensemble)
            y_pred = np.zeros(self.target_steps)
            for model_name, pred in predictions.items():
                # Ensure prediction length matches target steps
                if len(pred) == self.target_steps:
                    # Add weighted prediction (equal weights for now)
                    y_pred += pred / len(predictions)

            # Calculate confidence and signal strength
            confidence = 0.7  # Base confidence

            # Adjust confidence based on model agreement
            if len(predictions) > 1:
                # Calculate the standard deviation of predictions
                all_preds = np.array([pred for pred in predictions.values() if len(pred) == self.target_steps])
                pred_std = np.std(all_preds, axis=0)

                # Lower confidence if models disagree
                avg_std = np.mean(pred_std)
                confidence_adj = 1.0 - min(avg_std * 10, 0.5)  # Cap adjustment at 0.5
                confidence *= confidence_adj

            # Apply decay weights to future predictions (closer predictions matter more)
            weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])[:len(y_pred)]
            normalized_weights = weights / np.sum(weights)

            # Calculate weighted average of predictions for overall signal
            signal_strength = np.sum(y_pred * normalized_weights)

            # Adjust confidence based on prediction magnitude (stronger signals = higher confidence)
            magnitude_factor = min(abs(signal_strength) * 10, 1.0)
            confidence = min(confidence * (1.0 + magnitude_factor), 0.95)

            # Determine action based on signal strength and prediction threshold
            if signal_strength > self.prediction_threshold:
                action = "buy"
            elif signal_strength < -self.prediction_threshold:
                action = "sell"
            else:
                action = "hold"
                # Lower confidence for hold signals
                confidence *= 0.8

            # Store prediction for tracking
            self.prediction_history.append({
                "timeframe": timeframe,
                "timestamp": datetime.datetime.now().isoformat(),
                "prediction": y_pred.tolist(),
                "models_used": list(predictions.keys()),
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
                "models_used": list(predictions.keys()),
                "timestamp": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Prediction error for {timeframe}m timeframe: {str(e)}")
            traceback.print_exc()
            return self._generate_fallback_prediction(timeframe)

    def _generate_fallback_prediction(self, timeframe):
        """Generate a conservative fallback prediction when normal prediction fails."""
        return {
            "status": "fallback",
            "timeframe": timeframe,
            "prediction": [0.0] * self.target_steps,
            "signal_strength": 0.0,
            "confidence": 0.5,
            "action": "hold",
            "models_used": ["fallback"],
            "timestamp": datetime.datetime.now().isoformat()
        }

    def _generate_debug_prediction(self, data, timeframe):
        """Generate a synthetic prediction for debugging purposes."""
        logging.info(f"Using debug prediction for {timeframe}m timeframe")

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

        return {
            "status": "debug",
            "timeframe": timeframe,
            "prediction": y_pred,
            "signal_strength": float(signal_strength),
            "confidence": float(confidence),
            "action": action,
            "models_used": ["debug"],
            "timestamp": datetime.datetime.now().isoformat()
        }

    def multi_timeframe_prediction(self, data_dict):
        """Perform predictions across multiple timeframes and aggregate the results."""
        predictions = {}
        confidence_weighted_signals = []
        total_confidence = 0

        # Get predictions for each timeframe
        for tf in self.timeframes:
            if tf in data_dict and not data_dict[tf].empty:
                pred = self.predict(data_dict[tf], tf)
                predictions[tf] = pred

                if pred["status"] in ["success", "debug"]:
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
        if aggregated_signal > self.prediction_threshold:
            overall_action = "buy"
            overall_confidence = min(abs(aggregated_signal) * 5, 0.95)  # Cap confidence at 95%
        elif aggregated_signal < -self.prediction_threshold:
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

    def save_models(self, timeframe):
        """Save all models for a timeframe."""
        models = self.models.get(timeframe, {})

        # Save LSTM model
        if models.get('lstm') is not None:
            try:
                path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_lstm.h5")
                models['lstm'].save(path)
                logging.info(f"Saved LSTM model for {timeframe}m timeframe")
            except Exception as e:
                logging.error(f"Error saving LSTM model: {str(e)}")

        # Save XGBoost model
        if models.get('xgboost') is not None:
            try:
                path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_xgb.json")
                models['xgboost'].save_model(path)
                logging.info(f"Saved XGBoost model for {timeframe}m timeframe")
            except Exception as e:
                logging.error(f"Error saving XGBoost model: {str(e)}")

        # Save RandomForest model
        if models.get('random_forest') is not None:
            try:
                path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_rf.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(models['random_forest'], f)
                logging.info(f"Saved RandomForest model for {timeframe}m timeframe")
            except Exception as e:
                logging.error(f"Error saving RandomForest model: {str(e)}")

        # Save scaler
        if timeframe in self.scalers:
            try:
                path = os.path.join(self.model_dir, f"{self.symbol}_{timeframe}m_scaler.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(self.scalers[timeframe], f)
                logging.info(f"Saved scaler for {timeframe}m timeframe")
            except Exception as e:
                logging.error(f"Error saving scaler: {str(e)}")


class FeatureEngineering:
    """Feature engineering pipeline for creating advanced trading features."""

    def __init__(self):
        """Initialize the feature engineering pipeline."""
        pass

    def transform(self, df):
        """Apply feature engineering to the input DataFrame."""
        if df is None or df.empty:
            return df

        # Make a copy to avoid modifying the original
        df_result = df.copy()

        # Add price action features
        df_result = self._add_price_action_features(df_result)

        # Add volatility features
        df_result = self._add_volatility_features(df_result)

        # Add pattern recognition features
        df_result = self._add_pattern_features(df_result)

        # Add custom ratio features
        df_result = self._add_ratio_features(df_result)

        # Add time-based features
        df_result = self._add_time_features(df_result)

        return df_result

    def _add_price_action_features(self, df):
        """Add price action based features."""
        try:
            # Calculate price change features
            df['pct_change'] = df['close'].pct_change()
            df['pct_change_abs'] = df['pct_change'].abs()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

            # Calculate candle features
            df['body_size'] = abs(df['close'] - df['open'])
            df['body_to_range'] = df['body_size'] / (df['high'] - df['low'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['is_bullish'] = (df['close'] > df['open']).astype(int)

            # Calculate trend detection features
            df['close_minus_open'] = df['close'] - df['open']
            df['high_minus_low'] = df['high'] - df['low']
            df['high_minus_close'] = df['high'] - df['close']
            df['close_minus_low'] = df['close'] - df['low']

            # Price position within range
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

            # Calculate rolling features
            for window in [5, 10, 20]:
                # Calculate price momentum over window
                df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1

                # Calculate rolling mean of changes
                df[f'pct_change_mean_{window}'] = df['pct_change'].rolling(window=window).mean()

                # Calculate rolling standard deviation of changes
                df[f'pct_change_std_{window}'] = df['pct_change'].rolling(window=window).std()

            # Calculate higher level features
            df['momentum_change'] = df['momentum_5'] - df['momentum_5'].shift(1)
            df['acceleration'] = df['momentum_change'] - df['momentum_change'].shift(1)

        except Exception as e:
            logging.error(f"Error adding price action features: {str(e)}")

        return df

    def _add_volatility_features(self, df):
        """Add volatility based features."""
        try:
            # Ensure we have ATR
            if 'atr' not in df.columns and df.shape[0] > 14:
                if TALIB_AVAILABLE:
                    df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                else:
                    # Calculate manually
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift())
                    low_close = np.abs(df['low'] - df['close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    df['atr'] = tr.rolling(14).mean()

            # Normalized ATR (ATR as a percentage of price)
            if 'atr' in df.columns:
                df['atr_pct'] = df['atr'] / df['close']

            # Bollinger Band width and position
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                # Relative position within Bollinger Bands
                df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

                # Normalized Bollinger Band width
                df['bb_width_norm'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

                # Bollinger Band squeeze (lower values indicate a squeeze)
                for window in [20, 50]:
                    df[f'bb_squeeze_{window}'] = df['bb_width_norm'] / df['bb_width_norm'].rolling(window).max()

            # Historical volatility - Standard deviation of log returns
            for window in [5, 10, 20, 30]:
                if 'log_return' in df.columns:
                    df[f'hist_vol_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(252)  # Annualized

            # Volatility ratio - Short-term vs long-term volatility
            if 'hist_vol_10' in df.columns and 'hist_vol_30' in df.columns:
                df['vol_ratio'] = df['hist_vol_10'] / df['hist_vol_30']

            # Garman-Klass volatility estimator
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            df['gk_vol'] = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

            # Yang-Zhang volatility estimator (simplified)
            if 'open' in df.columns and df.shape[0] > 1:
                open_to_close = np.log(df['close'] / df['open'])
                close_to_open = np.log(df['open'] / df['close'].shift(1))
                df['yz_vol'] = open_to_close ** 2 + close_to_open ** 2

                # Smooth it
                df['yz_vol_smooth'] = df['yz_vol'].rolling(window=10).mean()

        except Exception as e:
            logging.error(f"Error adding volatility features: {str(e)}")

        return df

    def _add_pattern_features(self, df):
        """Add candlestick pattern recognition features."""
        try:
            if TALIB_AVAILABLE:
                # Add common candlestick patterns
                pattern_functions = [
                    ('doji', talib.CDLDOJI),
                    ('engulfing', talib.CDLENGULFING),
                    ('hammer', talib.CDLHAMMER),
                    ('hanging_man', talib.CDLHANGINGMAN),
                    ('morning_star', talib.CDLMORNINGSTAR),
                    ('evening_star', talib.CDLEVENINGSTAR),
                    ('shooting_star', talib.CDLSHOOTINGSTAR),
                    ('marubozu', talib.CDLMARUBOZU)
                ]

                for pattern_name, pattern_func in pattern_functions:
                    try:
                        df[f'pattern_{pattern_name}'] = pattern_func(
                            df['open'].values, df['high'].values,
                            df['low'].values, df['close'].values
                        )
                    except Exception:
                        # Skip if this pattern function fails
                        pass
            else:
                # Add basic pattern detection without TA-Lib
                # Doji pattern (open and close very close)
                body_size = abs(df['close'] - df['open'])
                shadow_size = df['high'] - df['low']
                df['pattern_doji'] = ((body_size / shadow_size) < 0.1).astype(int)

                # Hammer pattern (small body, long lower shadow, small upper shadow)
                body_mid = (df['open'] + df['close']) / 2
                upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
                lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

                hammer_condition = (
                        (body_size / shadow_size < 0.3) &
                        (lower_shadow > 2 * body_size) &
                        (upper_shadow < 0.5 * body_size)
                )
                df['pattern_hammer'] = hammer_condition.astype(int)

                # Bullish Engulfing
                bullish_engulfing = (
                        (df['close'] > df['open']) &  # Current candle is bullish
                        (df['open'].shift(1) > df['close'].shift(1)) &  # Previous candle is bearish
                        (df['open'] < df['close'].shift(1)) &  # Current open below previous close
                        (df['close'] > df['open'].shift(1))  # Current close above previous open
                )
                df['pattern_bullish_engulfing'] = bullish_engulfing.astype(int)

                # Bearish Engulfing
                bearish_engulfing = (
                        (df['close'] < df['open']) &  # Current candle is bearish
                        (df['open'].shift(1) < df['close'].shift(1)) &  # Previous candle is bullish
                        (df['open'] > df['close'].shift(1)) &  # Current open above previous close
                        (df['close'] < df['open'].shift(1))  # Current close below previous open
                )
                df['pattern_bearish_engulfing'] = bearish_engulfing.astype(int)

        except Exception as e:
            logging.error(f"Error adding pattern features: {str(e)}")

        return df

    def _add_ratio_features(self, df):
        """Add ratio-based features."""
        try:
            # Price to moving average ratios
            ma_columns = [col for col in df.columns if col.startswith('ema_') or col.startswith('sma_')]
            for ma_col in ma_columns:
                df[f'price_to_{ma_col}'] = df['close'] / df[ma_col]

            # Add cross-MA features
            if 'ema_9' in df.columns and 'ema_21' in df.columns:
                df['ema_9_to_21'] = df['ema_9'] / df['ema_21']
                df['ema_9_cross_21'] = ((df['ema_9'] > df['ema_21']) &
                                        (df['ema_9'].shift(1) <= df['ema_21'].shift(1))).astype(int)
                df['ema_21_cross_9'] = ((df['ema_9'] < df['ema_21']) &
                                        (df['ema_9'].shift(1) >= df['ema_21'].shift(1))).astype(int)

            # Volume to moving average ratio
            if 'volume' in df.columns and 'volume_ema' in df.columns:
                df['volume_ratio'] = df['volume'] / df['volume_ema']

            # Fibonacci levels
            # Use rolling window to calculate recent swing high/low
            window = 20
            df['swing_high'] = df['high'].rolling(window=window).max()
            df['swing_low'] = df['low'].rolling(window=window).min()
            df['fib_range'] = df['swing_high'] - df['swing_low']

            # Calculate price position relative to Fibonacci levels
            fibonacci_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            for i, level in enumerate(fibonacci_levels):
                fib_level = df['swing_low'] + level * df['fib_range']
                df[f'fib_{level}'] = fib_level

                # Distance to Fibonacci level
                df[f'dist_to_fib_{level}'] = (df['close'] - fib_level) / df['close']

            # Near which Fibonacci level
            distances = [abs(df[f'dist_to_fib_{level}']) for level in fibonacci_levels]
            df['nearest_fib'] = pd.DataFrame(distances).idxmin(axis=0)

        except Exception as e:
            logging.error(f"Error adding ratio features: {str(e)}")

        return df

    def _add_time_features(self, df):
        """Add time-based features."""
        try:
            # Check if we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                return df

            # Extract time components
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
            df['day_of_month'] = df.index.day
            df['week_of_year'] = df.index.isocalendar().week
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year

            # Trading session flags
            df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_us_session'] = ((df['hour'] >= 14) & (df['hour'] < 21)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Cyclical encoding of time features
            # Hours
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Day of week
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            # Month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        except Exception as e:
            logging.error(f"Error adding time features: {str(e)}")

        return df


# ------------------------------------------------------------------------------
# Trading Strategy Implementation
# ------------------------------------------------------------------------------
class TradingStrategy:
    """Advanced trading strategy with dynamic parameters."""

    def __init__(self, symbol, max_risk_per_trade=0.01):
        """Initialize the trading strategy."""
        self.symbol = symbol
        self.max_risk_per_trade = max_risk_per_trade

        # Strategy parameters
        self.entry_threshold = 0.6  # Minimum confidence for entry
        self.exit_threshold = 0.4  # Minimum confidence for exit
        self.profit_target = 0.03  # 3% take profit target
        self.stop_loss = 0.015  # 1.5% stop loss
        self.max_trades = 3  # Maximum concurrent trades
        self.min_trade_interval = 4 * 3600  # Minimum seconds between trades (4 hours)

        # Dynamic parameters (adjusted based on market conditions)
        self.dynamic_params = {
            'profit_target_adjust': 1.0,  # Multiplier for profit target
            'stop_loss_adjust': 1.0,  # Multiplier for stop loss
            'risk_adjust': 1.0,  # Multiplier for position sizing
            'confidence_adjust': 1.0  # Multiplier for confidence thresholds
        }

        # Strategy state
        self.current_position = None  # 'long', 'short', or None
        self.position_data = {}
        self.last_trade_time = 0

        # Performance tracking
        self.trades = []

        logging.info(f"Trading strategy initialized for {symbol}")

    def analyze_market(self, market_data, ai_prediction):
        """
        Analyze market conditions and generate trade signals.

        Args:
            market_data: Dictionary with market data for different timeframes
            ai_prediction: AI model prediction results

        Returns:
            Dictionary with market analysis and trade signals
        """
        try:
            # Extract data from inputs
            latest_price = market_data.get_latest_price()
            ai_action = ai_prediction.get("overall_action", "hold")
            ai_confidence = ai_prediction.get("overall_confidence", 0.5)

            # Get market conditions from the 15m timeframe (mid-term)
            df_15m = market_data.get_data("15")

            # Check if we have enough data for analysis
            if df_15m is None or df_15m.empty:
                logging.warning("Insufficient data for market analysis")
                return self._generate_hold_signal("Insufficient data")

            # Calculate market condition indicators
            market_conditions = self._calculate_market_conditions(df_15m)

            # Adjust strategy parameters based on market conditions
            self._adjust_strategy_parameters(market_conditions)

            # Generate entry/exit signals
            return self._generate_signals(
                ai_action, ai_confidence, latest_price, market_conditions
            )

        except Exception as e:
            logging.error(f"Error in market analysis: {str(e)}")
            traceback.print_exc()
            return self._generate_hold_signal(f"Error: {str(e)}")

    def _calculate_market_conditions(self, df):
        """Calculate various market condition indicators."""
        try:
            # Get the most recent candle
            latest = df.iloc[-1] if not df.empty else None

            if latest is None:
                return {
                    "trend": "neutral",
                    "volatility": "normal",
                    "volume": "normal",
                    "momentum": "neutral"
                }

            # Determine trend
            trend = "neutral"
            trend_strength = 0

            # Check EMA trend
            if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
                ema_9 = latest['ema_9']
                ema_21 = latest['ema_21']
                ema_50 = latest['ema_50']

                if ema_9 > ema_21 > ema_50:
                    trend = "bullish"
                    trend_strength += 1
                elif ema_9 < ema_21 < ema_50:
                    trend = "bearish"
                    trend_strength += 1

            # Check MACD
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                if latest['macd'] > latest['macd_signal']:
                    if trend == "bearish":
                        trend = "neutral"
                    else:
                        trend = "bullish"
                    trend_strength += 1
                elif latest['macd'] < latest['macd_signal']:
                    if trend == "bullish":
                        trend = "neutral"
                    else:
                        trend = "bearish"
                    trend_strength += 1

            # Determine volatility
            volatility = "normal"
            if 'atr' in df.columns and 'atr_pct' in df.columns:
                # Look at ATR as a percentage of price
                atr_pct = latest['atr_pct']

                if atr_pct > 0.015:  # More than 1.5% ATR/price ratio
                    volatility = "high"
                elif atr_pct < 0.005:  # Less than 0.5% ATR/price ratio
                    volatility = "low"

            # Determine volume
            volume = "normal"
            if 'volume_ratio' in df.columns:
                volume_ratio = latest['volume_ratio']

                if volume_ratio > 1.5:
                    volume = "high"
                elif volume_ratio < 0.5:
                    volume = "low"

            # Determine momentum
            momentum = "neutral"
            if 'rsi' in df.columns:
                rsi = latest['rsi']

                if rsi > 70:
                    momentum = "overbought"
                elif rsi < 30:
                    momentum = "oversold"

            return {
                "trend": trend,
                "trend_strength": trend_strength,
                "volatility": volatility,
                "volume": volume,
                "momentum": momentum
            }

        except Exception as e:
            logging.error(f"Error calculating market conditions: {str(e)}")
            return {
                "trend": "neutral",
                "volatility": "normal",
                "volume": "normal",
                "momentum": "neutral"
            }

    def _adjust_strategy_parameters(self, market_conditions):
        """Adjust strategy parameters based on market conditions."""
        try:
            # Reset adjustments
            for param in self.dynamic_params:
                self.dynamic_params[param] = 1.0

            # Adjust based on volatility
            if market_conditions["volatility"] == "high":
                # In high volatility: wider stops, higher targets, lower risk
                self.dynamic_params["profit_target_adjust"] = 1.3
                self.dynamic_params["stop_loss_adjust"] = 1.5
                self.dynamic_params["risk_adjust"] = 0.7
                self.dynamic_params["confidence_adjust"] = 1.2  # Require higher confidence
            elif market_conditions["volatility"] == "low":
                # In low volatility: tighter stops, lower targets, higher risk
                self.dynamic_params["profit_target_adjust"] = 0.8
                self.dynamic_params["stop_loss_adjust"] = 0.8
                self.dynamic_params["risk_adjust"] = 1.2

            # Adjust based on trend
            if market_conditions["trend"] == "bullish" and self.current_position != "long":
                # More aggressive for long entries in bullish trend
                self.dynamic_params["confidence_adjust"] *= 0.9  # Lower confidence requirement
            elif market_conditions["trend"] == "bearish" and self.current_position != "short":
                # More aggressive for short entries in bearish trend
                self.dynamic_params["confidence_adjust"] *= 0.9  # Lower confidence requirement

            # Adjust based on momentum
            if market_conditions["momentum"] == "overbought" and self.current_position != "short":
                # More conservative for long entries in overbought conditions
                self.dynamic_params["confidence_adjust"] *= 1.2  # Higher confidence requirement
            elif market_conditions["momentum"] == "oversold" and self.current_position != "long":
                # More conservative for short entries in oversold conditions
                self.dynamic_params["confidence_adjust"] *= 1.2  # Higher confidence requirement

        except Exception as e:
            logging.error(f"Error adjusting strategy parameters: {str(e)}")

    def _generate_signals(self, ai_action, ai_confidence, price, market_conditions):
        """Generate trade signals based on AI prediction and market conditions."""
        current_time = time.time()
        adjusted_confidence = ai_confidence * self.dynamic_params["confidence_adjust"]

        # Initialize signals
        signals = {
            "entry": None,  # 'long' or 'short'
            "exit": None,  # 'exit_long' or 'exit_short'
            "position_size": 0,  # Percentage of available capital
            "take_profit": None,  # Take profit price
            "stop_loss": None,  # Stop loss price
            "confidence": adjusted_confidence,
            "market_conditions": market_conditions,
            "reasoning": []
        }

        # Check trading frequency limit
        too_frequent = (current_time - self.last_trade_time) < self.min_trade_interval
        if too_frequent:
            signals["reasoning"].append(
                f"Trade frequency limit: Last trade was {(current_time - self.last_trade_time) / 3600:.1f} hours ago")

        # Handle existing position
        if self.current_position:
            # Check for exit signals
            if self.current_position == "long":
                if ai_action == "sell" and adjusted_confidence > self.exit_threshold:
                    signals["exit"] = "exit_long"
                    signals["reasoning"].append(
                        f"AI sell signal with {adjusted_confidence:.2%} confidence (vs. {self.exit_threshold:.2%} threshold)")
                    signals["reasoning"].append(
                        f"Market conditions: {market_conditions['trend']} trend, {market_conditions['volatility']} volatility")
                    return signals

            elif self.current_position == "short":
                if ai_action == "buy" and adjusted_confidence > self.exit_threshold:
                    signals["exit"] = "exit_short"
                    signals["reasoning"].append(
                        f"AI buy signal with {adjusted_confidence:.2%} confidence (vs. {self.exit_threshold:.2%} threshold)")
                    signals["reasoning"].append(
                        f"Market conditions: {market_conditions['trend']} trend, {market_conditions['volatility']} volatility")
                    return signals

        # No open position, check for entry signals
        else:
            # Apply entry threshold with adjustment
            adjusted_entry_threshold = self.entry_threshold * self.dynamic_params["confidence_adjust"]

            # Check if we're within trading frequency limits
            if not too_frequent:
                # Long entry
                if ai_action == "buy" and adjusted_confidence > adjusted_entry_threshold:
                    # Calculate position size
                    position_size = min(
                        self.max_risk_per_trade * self.dynamic_params["risk_adjust"],
                        self.max_risk_per_trade
                    )

                    # Calculate take profit and stop loss prices
                    take_profit = price * (1 + self.profit_target * self.dynamic_params["profit_target_adjust"])
                    stop_loss = price * (1 - self.stop_loss * self.dynamic_params["stop_loss_adjust"])

                    signals["entry"] = "long"
                    signals["position_size"] = position_size
                    signals["take_profit"] = take_profit
                    signals["stop_loss"] = stop_loss

                    signals["reasoning"].append(
                        f"AI buy signal with {adjusted_confidence:.2%} confidence (vs. {adjusted_entry_threshold:.2%} threshold)")
                    signals["reasoning"].append(
                        f"Market conditions: {market_conditions['trend']} trend, {market_conditions['volatility']} volatility")
                    signals["reasoning"].append(
                        f"Risk: {position_size:.2%}, TP: ${take_profit:.2f}, SL: ${stop_loss:.2f}")

                    return signals

                # Short entry
                elif ai_action == "sell" and adjusted_confidence > adjusted_entry_threshold:
                    # Calculate position size
                    position_size = min(
                        self.max_risk_per_trade * self.dynamic_params["risk_adjust"],
                        self.max_risk_per_trade
                    )

                    # Calculate take profit and stop loss prices
                    take_profit = price * (1 - self.profit_target * self.dynamic_params["profit_target_adjust"])
                    stop_loss = price * (1 + self.stop_loss * self.dynamic_params["stop_loss_adjust"])

                    signals["entry"] = "short"
                    signals["position_size"] = position_size
                    signals["take_profit"] = take_profit
                    signals["stop_loss"] = stop_loss

                    signals["reasoning"].append(
                        f"AI sell signal with {adjusted_confidence:.2%} confidence (vs. {adjusted_entry_threshold:.2%} threshold)")
                    signals["reasoning"].append(
                        f"Market conditions: {market_conditions['trend']} trend, {market_conditions['volatility']} volatility")
                    signals["reasoning"].append(
                        f"Risk: {position_size:.2%}, TP: ${take_profit:.2f}, SL: ${stop_loss:.2f}")

                    return signals

        # Default to hold
        signals["reasoning"].append(f"AI action: {ai_action} with confidence {adjusted_confidence:.2%}")
        if adjusted_confidence < self.entry_threshold:
            signals["reasoning"].append(f"Confidence below entry threshold of {self.entry_threshold:.2%}")

        if too_frequent:
            time_until_next = self.min_trade_interval - (current_time - self.last_trade_time)
            signals["reasoning"].append(f"Next trade allowed in {time_until_next / 60:.1f} minutes")

        return signals

    def _generate_hold_signal(self, reason="No signal"):
        """Generate a hold signal with a reason."""
        return {
            "entry": None,
            "exit": None,
            "position_size": 0,
            "take_profit": None,
            "stop_loss": None,
            "confidence": 0,
            "market_conditions": {
                "trend": "neutral",
                "volatility": "normal",
                "volume": "normal",
                "momentum": "neutral"
            },
            "reasoning": [reason]
        }

    def update_position(self, position_type, position_data):
        """Update the current position data."""
        self.current_position = position_type  # 'long', 'short', or None
        self.position_data = position_data
        self.last_trade_time = time.time()

        # Log the position change
        if position_type:
            logging.info(f"Position updated to {position_type}: {position_data}")
        else:
            logging.info("Position closed")

    def record_trade(self, trade_data):
        """Record a completed trade for performance tracking."""
        self.trades.append(trade_data)

        # Keep the trades list manageable
        if len(self.trades) > 1000:
            self.trades = self.trades[-1000:]


# ------------------------------------------------------------------------------
# Improved Order Management
# ------------------------------------------------------------------------------
class OrderManager:
    """Enhanced order management with better error handling."""

    def __init__(self, api_client, symbol, trading_mode="simulation", leverage=3):
        """Initialize the order manager."""
        self.api_client = api_client
        self.symbol = symbol
        self.trading_mode = trading_mode
        self.leverage = leverage

        # Order tracking
        self.orders = []
        self.positions = []
        self.order_history = []

        # Initialize leverage
        self.leverage_set = False

        logging.info(f"Order manager initialized for {symbol} in {trading_mode} mode")

    async def initialize(self):
        """Initialize the order manager."""
        if self.trading_mode == "real":
            # Set leverage
            await self.set_leverage(self.leverage)

            # Get current positions
            await self.update_positions()

        return True

    async def set_leverage(self, leverage):
        """Set leverage for trading."""
        if self.trading_mode != "real":
            logging.info(f"[SIMULATION] Setting leverage to {leverage}x")
            self.leverage = leverage
            self.leverage_set = True
            return {"success": True, "message": "Leverage set in simulation mode"}

        try:
            response = await self.api_client.set_leverage(self.symbol, leverage)

            if response and 'retCode' in response and response['retCode'] == 0:
                logging.info(f"Leverage set to {leverage}x for {self.symbol}")
                self.leverage = leverage
                self.leverage_set = True
                return {"success": True, "message": f"Leverage set to {leverage}x"}
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logging.error(f"Failed to set leverage: {error_msg}")
                return {"success": False, "message": f"Failed to set leverage: {error_msg}"}

        except Exception as e:
            logging.error(f"Error setting leverage: {str(e)}")
            return {"success": False, "message": f"Error setting leverage: {str(e)}"}

    async def update_positions(self):
        """Update the list of current positions."""
        if self.trading_mode != "real":
            return self.positions

        try:
            positions = await self.api_client.get_positions(self.symbol)

            if positions:
                # Filter positions with size > 0
                active_positions = [pos for pos in positions if pos.get('size', 0) > 0]
                self.positions = active_positions

                if active_positions:
                    logging.info(f"Current positions: {len(active_positions)}")
                    for pos in active_positions:
                        logging.info(f"  {pos['symbol']} {pos['side']} {pos['size']} @ {pos['entry_price']}")
                else:
                    logging.info("No open positions")

                return active_positions
            else:
                logging.warning("No position data received")
                return []

        except Exception as e:
            logging.error(f"Error updating positions: {str(e)}")
            return self.positions

    async def place_market_order(self, side, quantity, take_profit=None, stop_loss=None):
        """Place a market order with optional take profit and stop loss."""
        order_id = f"sim-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        if self.trading_mode != "real":
            # Log simulated order
            logging.info(f"[SIMULATION] {side} market order for {quantity} {self.symbol}")

            # Create simulated order result
            order_result = {
                "success": True,
                "message": "Order placed in simulation mode",
                "order_id": order_id,
                "symbol": self.symbol,
                "side": side,
                "order_type": "Market",
                "qty": quantity,
                "price": None,  # Market order, no price specified
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Track the order
            self.orders.append(order_result)

            # Create simulated position
            if side == "Buy":
                position_side = "long"
            else:
                position_side = "short"

            self.positions.append({
                "symbol": self.symbol,
                "side": side,
                "size": quantity,
                "position_value": 0,  # Would need current price
                "entry_price": 0,  # Would need current price
                "leverage": self.leverage,
                "unrealized_pnl": 0,
                "take_profit": take_profit,
                "stop_loss": stop_loss
            })

            return order_result

        try:
            # Ensure leverage is set
            if not self.leverage_set:
                await self.set_leverage(self.leverage)

            # Place the order
            response = await self.api_client.place_order(
                symbol=self.symbol,
                side=side,
                order_type="Market",
                qty=quantity,
                take_profit=take_profit,
                stop_loss=stop_loss
            )

            # Check for success
            if response and 'retCode' in response and response['retCode'] == 0:
                order_data = response.get('result', {})
                order_id = order_data.get('orderId')

                logging.info(f"Market order placed: {side} {quantity} {self.symbol}")

                # Create order result
                order_result = {
                    "success": True,
                    "message": "Order placed successfully",
                    "order_id": order_id,
                    "symbol": self.symbol,
                    "side": side,
                    "order_type": "Market",
                    "qty": quantity,
                    "price": None,  # Market order, no price specified
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "timestamp": datetime.datetime.now().isoformat()
                }

                # Track the order
                self.orders.append(order_result)

                # Update positions
                await self.update_positions()

                return order_result
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logging.error(f"Failed to place market order: {error_msg}")

                return {
                    "success": False,
                    "message": f"Failed to place market order: {error_msg}",
                    "order_id": None
                }

        except Exception as e:
            error_message = f"Error placing market order: {str(e)}"
            logging.error(error_message)
            traceback.print_exc()

            return {
                "success": False,
                "message": error_message,
                "order_id": None
            }

    async def close_position(self, side=None):
        """Close all positions or positions of a specific side."""
        if self.trading_mode != "real":
            logging.info(f"[SIMULATION] Closing positions: {side if side else 'all'}")

            # Filter positions to close
            if side:
                positions_to_close = [pos for pos in self.positions if pos['side'] == side]
            else:
                positions_to_close = self.positions.copy()

            # Clear positions
            if side:
                self.positions = [pos for pos in self.positions if pos['side'] != side]
            else:
                self.positions = []

            return {
                "success": True,
                "message": f"Closed {len(positions_to_close)} positions in simulation mode",
                "positions_closed": positions_to_close
            }

        try:
            # Get current positions
            positions = await self.update_positions()

            if not positions:
                logging.info("No positions to close")
                return {
                    "success": True,
                    "message": "No positions to close",
                    "positions_closed": []
                }

            # Filter positions to close
            if side:
                positions_to_close = [pos for pos in positions if pos['side'] == side]
            else:
                positions_to_close = positions

            if not positions_to_close:
                logging.info(f"No {side} positions to close")
                return {
                    "success": True,
                    "message": f"No {side} positions to close",
                    "positions_closed": []
                }

            results = []

            # Close each position
            for position in positions_to_close:
                # Determine order side (opposite of position side)
                order_side = "Sell" if position['side'] == "Buy" else "Buy"

                # Place market order to close position
                response = await self.api_client.place_order(
                    symbol=self.symbol,
                    side=order_side,
                    order_type="Market",
                    qty=position['size'],
                    reduce_only=True
                )

                # Check for success
                if response and 'retCode' in response and response['retCode'] == 0:
                    order_data = response.get('result', {})
                    order_id = order_data.get('orderId')

                    logging.info(f"Closed position: {position['side']} {position['size']} {self.symbol}")

                    results.append({
                        "success": True,
                        "position": position,
                        "order_id": order_id
                    })
                else:
                    error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                    logging.error(f"Failed to close position: {error_msg}")

                    results.append({
                        "success": False,
                        "position": position,
                        "message": error_msg
                    })

            # Update positions
            await self.update_positions()

            # Determine overall success
            all_success = all(result['success'] for result in results)

            return {
                "success": all_success,
                "message": f"Closed {len(results)} positions" if all_success else "Some positions failed to close",
                "positions_closed": results
            }

        except Exception as e:
            error_message = f"Error closing positions: {str(e)}"
            logging.error(error_message)
            traceback.print_exc()

            return {
                "success": False,
                "message": error_message,
                "positions_closed": []
            }

    async def get_order_status(self, order_id):
        """Get the status of an order."""
        if self.trading_mode != "real":
            # Find the order in our simulated orders
            for order in self.orders:
                if order['order_id'] == order_id:
                    return {
                        "success": True,
                        "order": order,
                        "status": "Filled"  # Assume all simulated orders are filled
                    }

            return {
                "success": False,
                "message": "Order not found",
                "status": None
            }

        try:
            # Get order details
            response = await self.api_client.request(
                'GET',
                '/v5/order/history',
                params={
                    'category': 'linear',
                    'symbol': self.symbol,
                    'orderId': order_id
                },
                auth=True
            )

            if response and 'retCode' in response and response['retCode'] == 0:
                orders = response.get('result', {}).get('list', [])

                if orders:
                    order = orders[0]

                    return {
                        "success": True,
                        "order": order,
                        "status": order.get('orderStatus')
                    }
                else:
                    logging.warning(f"Order {order_id} not found in history")

                    # Try active orders
                    response = await self.api_client.request(
                        'GET',
                        '/v5/order/realtime',
                        params={
                            'category': 'linear',
                            'symbol': self.symbol,
                            'orderId': order_id
                        },
                        auth=True
                    )

                    if response and 'retCode' in response and response['retCode'] == 0:
                        orders = response.get('result', {}).get('list', [])

                        if orders:
                            order = orders[0]

                            return {
                                "success": True,
                                "order": order,
                                "status": order.get('orderStatus')
                            }

                    return {
                        "success": False,
                        "message": "Order not found",
                        "status": None
                    }
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                logging.error(f"Failed to get order status: {error_msg}")

                return {
                    "success": False,
                    "message": f"Failed to get order status: {error_msg}",
                    "status": None
                }

        except Exception as e:
            error_message = f"Error getting order status: {str(e)}"
            logging.error(error_message)

            return {
                "success": False,
                "message": error_message,
                "status": None
            }


# ------------------------------------------------------------------------------
# Notification System
# ------------------------------------------------------------------------------
class NotificationSystem:
    """Enhanced notification system with multiple channels."""

    def __init__(self, telegram_token=None, telegram_chat_id=None):
        """Initialize the notification system."""
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.telegram_enabled = False
        self.telegram_bot = None

        # Initialize Telegram if credentials are provided
        if TELEGRAM_AVAILABLE and telegram_token and telegram_chat_id:
            try:
                self.telegram_bot = telegram.Bot(token=telegram_token)
                self.telegram_enabled = True
                logging.info("Telegram notifications enabled")
            except Exception as e:
                logging.error(f"Failed to initialize Telegram: {str(e)}")
        else:
            logging.info("Telegram notifications disabled")

    async def send_message(self, message, level="info"):
        """Send a notification message."""
        # Log the message
        if level == "error":
            logging.error(message)
        elif level == "warning":
            logging.warning(message)
        else:
            logging.info(message)

        # Send via Telegram if enabled
        if self.telegram_enabled:
            try:
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='HTML'
                )
                return True
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {str(e)}")
                return False

        return True

    async def send_trade_alert(self, action, symbol, price, quantity, confidence, risk):
        """Send a trade alert notification."""
        emoji = "🟢" if action.lower() == "buy" else "🔴"
        message = f"{emoji} <b>TRADE EXECUTED: {action.upper()}</b>\n\n" \
                  f"<b>Symbol:</b> {symbol}\n" \
                  f"<b>Price:</b> ${price:.2f}\n" \
                  f"<b>Quantity:</b> {quantity:.6f}\n" \
                  f"<b>Confidence:</b> {confidence:.2%}\n" \
                  f"<b>Risk Level:</b> {risk:.2%}\n" \
                  f"<b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return await self.send_message(message)

    async def send_error_alert(self, error_type, error_message, critical=False):
        """Send an error alert notification."""
        alert = "🚨 <b>CRITICAL ERROR</b> 🚨" if critical else "⚠️ <b>ERROR</b>"
        message = f"{alert}\n\n" \
                  f"<b>Type:</b> {error_type}\n" \
                  f"<b>Message:</b> {error_message}\n" \
                  f"<b>Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return await self.send_message(message, level="error")

    async def send_performance_report(self, data):
        """Send a performance report notification."""
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

        return await self.send_message(message)

    async def send_status_update(self, status):
        """Send a system status update notification."""
        message = f"🤖 <b>SYSTEM STATUS UPDATE</b>\n\n" \
                  f"<b>Running:</b> {'Yes' if status.get('running', False) else 'No'}\n" \
                  f"<b>Trading enabled:</b> {'Yes' if status.get('trading_enabled', False) else 'No'}\n" \
                  f"<b>Current symbol:</b> {status.get('symbol', 'N/A')}\n" \
                  f"<b>Current leverage:</b> {status.get('leverage', 'N/A')}x\n" \
                  f"<b>Last prediction:</b> {status.get('last_prediction', 'N/A')}\n" \
                  f"<b>Last trade:</b> {status.get('last_trade', 'N/A')}\n" \
                  f"<b>Balance:</b> ${status.get('balance', 0):.2f}\n" \
                  f"<b>P&L:</b> ${status.get('pnl', 0):.2f} ({status.get('pnl_percent', 0):.2%})\n"

        return await self.send_message(message)

    async def send_chart(self, chart_path, caption=""):
        """Send a chart image."""
        if not self.telegram_enabled:
            logging.info(f"Chart would be sent via Telegram: {chart_path}")
            return False

        if not os.path.exists(chart_path):
            logging.error(f"Chart file not found: {chart_path}")
            return False

        try:
            with open(chart_path, 'rb') as photo:
                await self.telegram_bot.send_photo(
                    chat_id=self.telegram_chat_id,
                    photo=photo,
                    caption=caption
                )
            return True
        except Exception as e:
            logging.error(f"Failed to send chart via Telegram: {str(e)}")
            return False


# ------------------------------------------------------------------------------
# Main Trading System
# ------------------------------------------------------------------------------
class EnhancedTradingSystem:
    """Main trading system that integrates all components with async support."""

    def __init__(self, config=None):
        """Initialize the trading system."""
        # Initialize configuration
        if config is None:
            # Use default configuration
            defaults = {
                "symbol": DEFAULT_SYMBOL,
                "timeframes": DEFAULT_TIMEFRAMES,
                "leverage": DEFAULT_LEVERAGE,
                "max_risk_per_trade": DEFAULT_MAX_RISK_PER_TRADE,
                "trading_mode": DEFAULT_TRADING_MODE,
                "bybit_api_key": DEFAULT_BYBIT_API_KEY,
                "bybit_api_secret": DEFAULT_BYBIT_API_SECRET,
                "telegram_bot_token": DEFAULT_TELEGRAM_BOT_TOKEN,
                "telegram_chat_id": DEFAULT_TELEGRAM_CHAT_ID,
                "debug_mode": DEFAULT_DEBUG_MODE,
                "use_testnet": False
            }
            self.config_manager = ConfigManager(defaults=defaults)
        else:
            self.config_manager = config

        # Extract configuration for convenience
        self.symbol = self.config_manager.get("symbol", DEFAULT_SYMBOL)
        self.timeframes = self.config_manager.get("timeframes", DEFAULT_TIMEFRAMES)
        self.leverage = self.config_manager.get("leverage", DEFAULT_LEVERAGE)
        self.trading_mode = self.config_manager.get("trading_mode", DEFAULT_TRADING_MODE)
        self.debug_mode = self.config_manager.get("debug_mode", DEFAULT_DEBUG_MODE)

        # Initialize logging
        self.logger = setup_logging(debug_mode=self.debug_mode)

        # Create component placeholders
        self.api_client = None
        self.market_data = None
        self.ai_model = None
        self.strategy = None
        self.order_manager = None
        self.notification = None
        self.visualizer = None

        # System state
        self.running = False
        self.trading_enabled = True
        self.start_time = None
        self.last_update_time = None
        self.last_prediction = None
        self.last_trade_signal = None
        self.last_trade = None
        self.update_interval = 60  # seconds

        # Performance tracking
        self.trades = []
        self.account_history = []

        # Event loop and tasks
        self.loop = None
        self.main_task = None
        self.stop_event = None
        self.task_manager = None

        logging.info(f"Trading system initialized for {self.symbol}")

    async def _update_account_history(self):
        """به‌روز کردن تاریخچه حساب."""
        try:
            balance = await self._get_account_balance()
            timestamp = datetime.datetime.now().isoformat()
            entry = {
                'timestamp': timestamp,
                'equity': balance.get('equity', 0),
                'available_balance': balance.get('available_balance', 0),
                'position_value': balance.get('position_value', 0)
            }
            self.account_history.append(entry)
            self.account_history = self.account_history[-1000:]
            return True
        except Exception as e:
            logging.error(f"Error updating account history: {str(e)}")
            return False

    async def initialize(self):
        """Initialize all system components."""
        try:
            logging.info("Initializing trading system components...")

            # Initialize API client
            self.api_client = BybitClient(
                api_key=self.config_manager.get("bybit_api_key", ""),
                api_secret=self.config_manager.get("bybit_api_secret", ""),
                use_testnet=self.config_manager.get("use_testnet", False)
            )
            await self.api_client.initialize()

            # Initialize market data manager
            self.market_data = MarketDataManager(
                api_client=self.api_client,
                symbol=self.symbol,
                timeframes=self.timeframes
            )
            await self.market_data.initialize()

            # Initialize AI model
            self.ai_model = EnhancedAIModel(
                symbol=self.symbol,
                timeframes=self.timeframes,
                debug_mode=self.debug_mode
            )

            # Initialize trading strategy
            self.strategy = TradingStrategy(
                symbol=self.symbol,
                max_risk_per_trade=self.config_manager.get("max_risk_per_trade", DEFAULT_MAX_RISK_PER_TRADE)
            )

            # Initialize order manager
            self.order_manager = OrderManager(
                api_client=self.api_client,
                symbol=self.symbol,
                trading_mode=self.trading_mode,
                leverage=self.leverage
            )
            await self.order_manager.initialize()

            # Initialize notification system
            self.notification = NotificationSystem(
                telegram_token=self.config_manager.get("telegram_bot_token", ""),
                telegram_chat_id=self.config_manager.get("telegram_chat_id", "")
            )

            # Initialize task manager
            self.task_manager = AsyncTaskManager()
            await self.task_manager.start()

            # Initialize event
            self.stop_event = asyncio.Event()

            logging.info("All components initialized successfully")

            # Send startup notification
            if self.notification:
                try:
                    await self.notification.send_message(
                        f"🚀 <b>Trading Bot Initialized</b>\n\n"
                        f"<b>Symbol:</b> {self.symbol}\n"
                        f"<b>Mode:</b> {self.trading_mode.upper()}\n"
                        f"<b>Leverage:</b> {self.leverage}x\n"
                        f"<b>Timeframes:</b> {', '.join(self.timeframes)}m\n"
                    )
                except Exception as e:
                    logging.error(f"Failed to send startup notification: {str(e)}")

            return True

        except Exception as e:
            logging.error(f"Error initializing trading system: {str(e)}")
            traceback.print_exc()

            if self.notification:
                try:
                    await self.notification.send_error_alert(
                        "Initialization Error",
                        str(e),
                        critical=True
                    )
                except:
                    pass

            return False

    async def start(self):
        """Start the trading system."""
        if self.running:
            logging.warning("Trading system is already running")
            return False

        try:
            # Initialize components if not already done
            if not self.api_client:
                if not await self.initialize():
                    return False

            logging.info("Starting trading system...")
            self.running = True
            self.start_time = datetime.datetime.now()

            # Start websockets
            await self.api_client.start_websockets()

            # Start the main loop
            self.main_task = asyncio.create_task(self._main_loop())

            # Send startup notification
            if self.notification:
                try:
                    await self.notification.send_message(
                        f"🚀 <b>Trading Bot Started</b>\n\n"
                        f"<b>Symbol:</b> {self.symbol}\n"
                        f"<b>Mode:</b> {self.trading_mode.upper()}\n"
                        f"<b>Start Time:</b> {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                except Exception as e:
                    logging.error(f"Failed to send startup notification: {str(e)}")

            logging.info("Trading system started successfully")
            return True

        except Exception as e:
            self.running = False
            logging.error(f"Error starting trading system: {str(e)}")
            traceback.print_exc()

            if self.notification:
                try:
                    await self.notification.send_error_alert(
                        "Startup Error",
                        str(e),
                        critical=True
                    )
                except:
                    pass

            return False

    async def stop(self):
        """Stop the trading system."""
        if not self.running:
            logging.warning("Trading system is not running")
            return True

        try:
            logging.info("Stopping trading system...")

            # Signal the main loop to stop
            self.stop_event.set()

            # Wait for the main loop to finish
            if self.main_task:
                try:
                    await asyncio.wait_for(self.main_task, timeout=30)
                except asyncio.TimeoutError:
                    logging.warning("Timeout waiting for main loop to finish, cancelling...")
                    self.main_task.cancel()
                    try:
                        await self.main_task
                    except asyncio.CancelledError:
                        pass

            # Stop all websockets
            await self.api_client.stop_websockets()

            # Stop the task manager
            await self.task_manager.stop()

            # Close API client
            await self.api_client.close()

            self.running = False

            # Send shutdown notification
            if self.notification:
                try:
                    runtime = datetime.datetime.now() - self.start_time if self.start_time else datetime.timedelta(0)
                    await self.notification.send_message(
                        f"🛑 <b>Trading Bot Stopped</b>\n\n"
                        f"<b>Runtime:</b> {runtime.total_seconds() / 3600:.1f} hours\n"
                        f"<b>Stop Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                except Exception as e:
                    logging.error(f"Failed to send shutdown notification: {str(e)}")

            logging.info("Trading system stopped successfully")
            return True

        except Exception as e:
            logging.error(f"Error stopping trading system: {str(e)}")
            traceback.print_exc()
            self.running = False
            return False

    async def _main_loop(self):
        """Main trading system loop."""
        logging.info("Starting main trading loop")

        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()

                try:
                    # Update market data
                    logging.info("Updating market data...")
                    await self.market_data.update_all()

                    # Generate AI predictions
                    logging.info("Generating AI predictions...")
                    prediction = self.ai_model.multi_timeframe_prediction(self.market_data.get_all_data())
                    self.last_prediction = prediction

                    # Generate trade signals
                    logging.info("Generating trade signals...")
                    trade_signal = self.strategy.analyze_market(self.market_data, prediction)
                    self.last_trade_signal = trade_signal

                    # Log the results
                    logging.info(
                        f"AI prediction: {prediction['overall_action']} with {prediction['overall_confidence']:.2%} confidence")

                    if trade_signal.get('entry'):
                        logging.info(
                            f"Trade signal: ENTRY {trade_signal['entry'].upper()} with {trade_signal['confidence']:.2%} confidence")
                    elif trade_signal.get('exit'):
                        logging.info(
                            f"Trade signal: EXIT {trade_signal['exit'].upper()} with {trade_signal['confidence']:.2%} confidence")
                    else:
                        logging.info("Trade signal: HOLD")

                    # Execute trades if trading is enabled
                    if self.trading_enabled:
                        await self._execute_trades(trade_signal)

                    # Update account history
                    await self._update_account_history()

                    # Update last update time
                    self.last_update_time = datetime.datetime.now()

                except Exception as e:
                    logging.error(f"Error in main loop iteration: {str(e)}")
                    traceback.print_exc()

                    if self.notification:
                        try:
                            await self.notification.send_error_alert(
                                "Main Loop Error",
                                str(e),
                                critical=False
                            )
                        except:
                            pass

                # Calculate time to sleep
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, self.update_interval - elapsed)

                # Log update completion
                logging.info(f"Update completed in {elapsed:.2f}s, next update in {sleep_time:.2f}s")

                # Sleep until next update
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=sleep_time)
                    if self.stop_event.is_set():
                        break
                except asyncio.TimeoutError:
                    pass

            logging.info("Main trading loop stopped")

        except asyncio.CancelledError:
            logging.info("Main trading loop cancelled")

        except Exception as e:
            logging.error(f"Unhandled error in main loop: {str(e)}")
            traceback.print_exc()

            if self.notification:
                try:
                    await self.notification.send_error_alert(
                        "Critical Main Loop Error",
                        str(e),
                        critical=True
                    )
                except:
                    pass

    async def _execute_trades(self, trade_signal):
        """Execute trades based on signals."""
        try:
            # Handle entry signals
            if trade_signal.get('entry'):
                entry_type = trade_signal['entry']
                position_size = trade_signal['position_size']
                take_profit = trade_signal['take_profit']
                stop_loss = trade_signal['stop_loss']
                confidence = trade_signal['confidence']

                # Get current price
                current_price = self.market_data.get_latest_price()

                # Get account balance
                balance = await self._get_account_balance()
                available_balance = balance.get('available_balance', 0)

                # Calculate quantity based on position size and leverage
                position_value = available_balance * position_size
                quantity = (position_value * self.leverage) / current_price

                # Format to 4 decimal places (adjust based on asset)
                quantity = round(quantity, 4)

                if quantity <= 0:
                    logging.warning(f"Invalid quantity calculated: {quantity}")
                    return False

                # Determine order side
                if entry_type == 'long':
                    side = 'Buy'
                elif entry_type == 'short':
                    side = 'Sell'
                else:
                    logging.warning(f"Invalid entry type: {entry_type}")
                    return False

                logging.info(f"Executing {entry_type.upper()} entry: {quantity} {self.symbol} @ ${current_price:.2f}")
                logging.info(f"Take profit: ${take_profit:.2f}, Stop loss: ${stop_loss:.2f}")

                # Place order
                order_result = await self.order_manager.place_market_order(
                    side=side,
                    quantity=quantity,
                    take_profit=take_profit,
                    stop_loss=stop_loss
                )

                # Update strategy position
                if order_result['success']:
                    self.strategy.update_position(
                        position_type=entry_type,
                        position_data={
                            'entry_price': current_price,
                            'quantity': quantity,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'order_id': order_result['order_id'],
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                    )

                    # Log trade
                    self.last_trade = {
                        'action': side,
                        'type': 'entry',
                        'price': current_price,
                        'quantity': quantity,
                        'timestamp': datetime.datetime.now().isoformat()
                    }

                    # Add to trades history
                    self.trades.append({
                        'type': 'entry',
                        'side': side,
                        'price': current_price,
                        'quantity': quantity,
                        'value': position_value,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'leverage': self.leverage,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'order_id': order_result['order_id']
                    })

                    # Send notification
                    if self.notification:
                        await self.notification.send_trade_alert(
                            action=side.lower(),
                            symbol=self.symbol,
                            price=current_price,
                            quantity=quantity,
                            confidence=confidence,
                            risk=position_size
                        )

                    return True
                else:
                    logging.error(f"Order placement failed: {order_result['message']}")

                    # Send error notification
                    if self.notification:
                        await self.notification.send_error_alert(
                            "Order Error",
                            f"Failed to place {entry_type} order: {order_result['message']}",
                            critical=False
                        )

                    return False

            # Handle exit signals
            elif trade_signal.get('exit'):
                exit_type = trade_signal['exit']
                confidence = trade_signal['confidence']

                # Get current price
                current_price = self.market_data.get_latest_price()

                # Determine position to close
                if exit_type == 'exit_long':
                    side = 'Sell'
                    position_side = 'Buy'  # Close long position
                elif exit_type == 'exit_short':
                    side = 'Buy'
                    position_side = 'Sell'  # Close short position
                else:
                    logging.warning(f"Invalid exit type: {exit_type}")
                    return False

                logging.info(f"Executing {exit_type.upper()} @ ${current_price:.2f}")

                # Close position
                result = await self.order_manager.close_position(side=position_side)

                # Update strategy position
                if result['success']:
                    # Find entry trade
                    entry_side = 'Buy' if position_side == 'Buy' else 'Sell'
                    entry_trade = next(
                        (t for t in reversed(self.trades) if t['type'] == 'entry' and t['side'] == entry_side),
                        None
                    )

                    if entry_trade:
                        # Calculate profit/loss
                        entry_price = entry_trade['price']
                        exit_price = current_price
                        quantity = entry_trade['quantity']

                        if position_side == 'Buy':  # Long position
                            pnl = (exit_price - entry_price) * quantity
                            pnl_percent = (exit_price - entry_price) / entry_price
                        else:  # Short position
                            pnl = (entry_price - exit_price) * quantity
                            pnl_percent = (entry_price - exit_price) / entry_price

                        # Add exit trade to history
                        self.trades.append({
                            'type': 'exit',
                            'side': side,
                            'price': current_price,
                            'quantity': quantity,
                            'pnl': pnl,
                            'pnl_percent': pnl_percent,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'related_entry': entry_trade.get('order_id')
                        })

                        # Log trade
                        self.last_trade = {
                            'action': side,
                            'type': 'exit',
                            'price': current_price,
                            'quantity': quantity,
                            'pnl': pnl,
                            'pnl_percent': pnl_percent,
                            'timestamp': datetime.datetime.now().isoformat()
                        }

                        # Reset strategy position
                        self.strategy.update_position(position_type=None, position_data={})

                        # Send notification
                        if self.notification:
                            emoji = "✅" if pnl > 0 else "❌"
                            message = f"{emoji} <b>POSITION CLOSED</b>\n\n" \
                                      f"<b>Symbol:</b> {self.symbol}\n" \
                                      f"<b>Type:</b> {position_side}\n" \
                                      f"<b>Exit Price:</b> ${current_price:.2f}\n" \
                                      f"<b>P&L:</b> ${pnl:.2f} ({pnl_percent:.2%})\n"

                            await self.notification.send_message(message)

                    else:
                        logging.warning("No matching entry trade found for exit")

                    return True
                else:
                    logging.error(f"Position closing failed: {result['message']}")

                    # Send error notification
                    if self.notification:
                        await self.notification.send_error_alert(
                            "Close Position Error",
                            f"Failed to close {position_side} position: {result['message']}",
                            critical=False
                        )

                    return False

        except Exception as e:
            logging.error(f"Error executing trades: {str(e)}")
            traceback.print_exc()

            if self.notification:
                await self.notification.send_error_alert(
                    "Trade Execution Error",
                    str(e),
                    critical=False
                )

            return False

    async def _get_account_balance(self):
        """Get current account balance."""
        if self.trading_mode != "real":
            # Return simulated balance
            return {
                "total_balance": 10000.0,
                "available_balance": 10000.0,
                "unrealized_pnl": 0.0,
                "margin_balance": 10000.0
            }

        try:
            account_info = await self.api_client.get_account_info()

            if isinstance(account_info, dict) and 'list' in account_info:
                account_data = account_info['list'][0]
                coin_data = account_data.get('coin', [])

                # Find USDT data
                usdt_data = next((coin for coin in coin_data if coin.get('coin') == 'USDT'), None)

                if usdt_data:
                    return {
                        "total_balance": safe_float(usdt_data.get('walletBalance', 0)),
                        "available_balance": safe_float(usdt_data.get('availableToWithdraw', 0)),
                        "unrealized_pnl": safe_float(usdt_data.get('unrealisedPnl', 0)),
                        "margin_balance": safe_float(usdt_data.get('marginBalance', 0))
                    }

            # Fallback to default
            return {
                "total_balance": 0.0,
                "available_balance": 0.0,
                "unrealized_pnl": 0.0,
                "margin_balance": 0.0
            }

        except Exception as e:
            logging.error(f"Error getting account balance: {str(e)}")
            return {
                "total_balance": 0.0,
                "available_balance": 0.0,
                "unrealized_pnl": 0.0,
                "margin_balance": 0.0
            }


async def _update_account_history(self):
    """Update account balance history."""
    try:
        # Get current account balance
        balance = await self._get_account_balance()

        # Create a new history entry
        timestamp = datetime.datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'equity': balance.get('equity', 0),
            'available_balance': balance.get('available_balance', 0),
            'position_value': balance.get('position_value', 0),
        }

        # Add to our account history
        self.account_history.append(entry)

        # Limit history to the latest 1000 entries
        self.account_history = self.account_history[-1000:]

        return True

    except Exception as e:
        logging.error(f"Error updating account history: {str(e)}")
        return False


async def get_status(self):
    """Get current system status."""
    try:
        # Get current account balance
        balance = await self._get_account_balance()

        # Calculate performance metrics
        metrics = self.get_performance_metrics()

        # Get current positions (if order_manager is implemented)
        positions = []
        if self.order_manager:
            positions = await self.order_manager.update_positions()

        # Build status report
        status = {
            "running": self.running,
            "trading_enabled": self.trading_enabled,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_hours": (
                    (datetime.datetime.now() - self.start_time).total_seconds() / 3600
            ) if self.start_time else 0,
            "symbol": self.symbol,
            "mode": self.trading_mode,
            "leverage": self.leverage,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "last_prediction": (
                self.last_prediction.get("overall_action", "unknown")
                if self.last_prediction else None
            ),
            "last_trade": self.last_trade,
            "balance": balance.get("total_balance", 0),
            "equity": balance.get("total_balance", 0) + balance.get("unrealized_pnl", 0),
            "unrealized_pnl": balance.get("unrealized_pnl", 0),
            "available_balance": balance.get("available_balance", 0),
            "positions": positions,
            "trades_count": len(self.trades),
            "win_rate": metrics.get("win_rate", 0),
            "pnl": metrics.get("net_pnl", 0),
            "pnl_percent": metrics.get("pnl_percent", 0)
        }

        return status

    except Exception as e:
        logging.error(f"Error getting system status: {str(e)}")
        return {
            "running": self.running,
            "error": str(e)
        }


def get_performance_metrics(self):
    """Calculate performance metrics (e.g., P&L, win rate, drawdown)."""
    try:
        # Gather only exit trades
        completed_trades = [t for t in self.trades if t.get("type") == "exit"]

        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.get("pnl", 0) > 0])
        losing_trades = total_trades - winning_trades

        # Basic stats
        win_rate = winning_trades / total_trades if total_trades else 0
        net_pnl = sum(t.get("pnl", 0) for t in completed_trades)

        # P&L percentage from first to last recorded equity
        if len(self.account_history) > 1:
            initial_equity = self.account_history[0].get("equity", 0)
            current_equity = self.account_history[-1].get("equity", 0)
            pnl_percent = (
                (current_equity - initial_equity) / initial_equity
                if initial_equity > 0 else 0
            )
        else:
            pnl_percent = 0

        # Simplified Sharpe ratio
        if total_trades > 1:
            returns = [t.get("pnl_percent", 0) for t in completed_trades]
            avg_return = sum(returns) / len(returns) if returns else 0
            std_dev = np.std(returns) if len(returns) > 1 else 1
            sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        if len(self.account_history) > 1:
            equity_values = [h.get("equity", 0) for h in self.account_history]
            peak = equity_values[0]
            max_drawdown = 0
            for eq in equity_values:
                if eq > peak:
                    peak = eq
                drawdown = (peak - eq) / peak if peak else 0
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0

        # Holding times
        holding_times = []
        for exit_trade in completed_trades:
            related_entry_id = exit_trade.get("related_entry")
            if related_entry_id:
                entry_trade = next(
                    (t for t in self.trades if t.get("order_id") == related_entry_id),
                    None
                )
                if entry_trade:
                    entry_time_str = entry_trade.get("timestamp", "")
                    exit_time_str = exit_trade.get("timestamp", "")
                    if entry_time_str and exit_time_str:
                        entry_time = datetime.datetime.fromisoformat(entry_time_str)
                        exit_time = datetime.datetime.fromisoformat(exit_time_str)
                        holding_times.append(
                            (exit_time - entry_time).total_seconds() / 3600.0
                        )

        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

        # Average profit/loss and profit factor
        avg_profit = (
            sum(t.get("pnl", 0) for t in completed_trades if t.get("pnl", 0) > 0)
            / winning_trades
            if winning_trades > 0 else 0
        )
        avg_loss = (
            sum(t.get("pnl", 0) for t in completed_trades if t.get("pnl", 0) < 0)
            / losing_trades
            if losing_trades > 0 else 0
        )
        gross_profit = sum(t.get("pnl", 0) for t in completed_trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in completed_trades if t.get("pnl", 0) < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "pnl_percent": pnl_percent,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_holding_time": avg_holding_time,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "period": f"{len(self.account_history)} periods"
            if self.account_history else "Unknown"
        }

    except Exception as e:
        logging.error(f"Error calculating performance metrics: {str(e)}")
        return {
            "total_trades": 0,
            "win_rate": 0,
            "net_pnl": 0,
            "error": str(e)
        }


async def send_performance_report(self):
    """Generate and send a performance report."""
    try:
        metrics = self.get_performance_metrics()
        if self.notification:
            await self.notification.send_performance_report(metrics)
        return metrics
    except Exception as e:
        logging.error(f"Error sending performance report: {str(e)}")
        return None


async def get_status(self):
    """Get current system status."""
    try:
        # Get account balance
        balance = await self._get_account_balance()

        # Calculate performance metrics
        metrics = self.get_performance_metrics()

        # Get current positions
        positions = await self.order_manager.update_positions()

        # Create status report
        status = {
            "running": self.running,
            "trading_enabled": self.trading_enabled,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_hours": (
                        (datetime.datetime.now() - self.start_time).total_seconds() / 3600) if self.start_time else 0,
            "symbol": self.symbol,
            "mode": self.trading_mode,
            "leverage": self.leverage,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "last_prediction": self.last_prediction.get("overall_action", "unknown") if self.last_prediction else None,
            "last_trade": self.last_trade,
            "balance": balance.get("total_balance", 0),
            "equity": balance.get("total_balance", 0) + balance.get("unrealized_pnl", 0),
            "unrealized_pnl": balance.get("unrealized_pnl", 0),
            "available_balance": balance.get("available_balance", 0),
            "positions": positions,
            "trades_count": len(self.trades),
            "win_rate": metrics.get("win_rate", 0),
            "pnl": metrics.get("net_pnl", 0),
            "pnl_percent": metrics.get("pnl_percent", 0)
        }

        return status

    except Exception as e:
        logging.error(f"Error getting system status: {str(e)}")
        return {
            "running": self.running,
            "error": str(e)
        }


def get_performance_metrics(self):
    """Calculate performance metrics."""
    try:
        # Get completed trades
        completed_trades = [t for t in self.trades if t.get("type") == "exit"]

        # Calculate basic metrics
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.get("pnl", 0) > 0])
        losing_trades = total_trades - winning_trades

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Net P&L
        net_pnl = sum(t.get("pnl", 0) for t in completed_trades)

        # Calculate P&L percentage based on account history
        if self.account_history and len(self.account_history) > 1:
            initial_equity = self.account_history[0].get("equity", 0)
            current_equity = self.account_history[-1].get("equity", 0)

            if initial_equity > 0:
                pnl_percent = (current_equity - initial_equity) / initial_equity
            else:
                pnl_percent = 0
        else:
            pnl_percent = 0

        # Calculate Sharpe ratio (simplified)
        if total_trades > 1:
            returns = [t.get("pnl_percent", 0) for t in completed_trades]
            avg_return = sum(returns) / len(returns) if returns else 0
            std_dev = np.std(returns) if len(returns) > 1 else 1
            sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        if self.account_history and len(self.account_history) > 1:
            equity_values = [h.get("equity", 0) for h in self.account_history]
            peak = equity_values[0]
            max_drawdown = 0

            for equity in equity_values:
                if equity > peak:
                    peak = equity

                drawdown = (peak - equity) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0

        # Calculate holding times
        holding_times = []

        for exit_trade in completed_trades:
            related_entry_id = exit_trade.get("related_entry")

            if related_entry_id:
                entry_trade = next((t for t in self.trades if t.get("order_id") == related_entry_id), None)

                if entry_trade:
                    entry_time = datetime.datetime.fromisoformat(entry_trade.get("timestamp", ""))
                    exit_time = datetime.datetime.fromisoformat(exit_trade.get("timestamp", ""))

                    holding_time = (exit_time - entry_time).total_seconds() / 3600  # hours
                    holding_times.append(holding_time)

        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

        # Calculate average profit and loss
        avg_profit = sum(t.get("pnl", 0) for t in completed_trades if
                         t.get("pnl", 0) > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.get("pnl", 0) for t in completed_trades if
                       t.get("pnl", 0) < 0) / losing_trades if losing_trades > 0 else 0

        # Calculate profit factor
        gross_profit = sum(t.get("pnl", 0) for t in completed_trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in completed_trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Return all metrics
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "pnl_percent": pnl_percent,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_holding_time": avg_holding_time,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "period": f"{len(self.account_history)} periods" if self.account_history else "Unknown"
        }

    except Exception as e:
        logging.error(f"Error calculating performance metrics: {str(e)}")
        return {
            "total_trades": 0,
            "win_rate": 0,
            "net_pnl": 0,
            "error": str(e)
        }


async def send_performance_report(self):
    """Generate and send a performance report."""
    try:
        metrics = self.get_performance_metrics()

        if self.notification:
            await self.notification.send_performance_report(metrics)

        return metrics

    except Exception as e:
        logging.error(f"Error sending performance report: {str(e)}")
        return None


# ------------------------------------------------------------------------------
# Main Function and Entry Point
# ------------------------------------------------------------------------------
async def run_trading_bot(config=None):
    """Run the trading bot asynchronously."""
    # Initialize the trading system
    system = EnhancedTradingSystem(config)

    try:
        # Initialize components
        if not await system.initialize():
            logging.error("Failed to initialize trading system")
            return False

        # Start the system
        if not await system.start():
            logging.error("Failed to start trading system")
            return False

        # Keep running until interrupted
        while system.running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt detected, stopping...")
                break

        # Stop the system
        await system.stop()
        return True

    except Exception as e:
        logging.error(f"Unhandled error in trading bot: {str(e)}")
        traceback.print_exc()

        # Try to stop the system
        try:
            if system.running:
                await system.stop()
        except:
            pass

        return False


def main():
    """Main entry point for the trading bot."""
    print("=" * 80)
    print("   ENHANCED AI-DRIVEN AUTOMATED FUTURES TRADING BOT FOR BYBIT EXCHANGE")
    print("=" * 80)

    # Load configuration
    config = ConfigManager()

    # Print current configuration
    print(f"Symbol: {config.get('symbol', DEFAULT_SYMBOL)}")
    print(f"Trading Mode: {config.get('trading_mode', DEFAULT_TRADING_MODE).upper()}")
    print(f"Leverage: {config.get('leverage', DEFAULT_LEVERAGE)}x")
    print(f"Max Risk Per Trade: {config.get('max_risk_per_trade', DEFAULT_MAX_RISK_PER_TRADE) * 100:.1f}%")
    print(f"Using Testnet: {'Yes' if config.get('use_testnet', False) else 'No'}")
    print("-" * 80)

    # Check for API credentials
    if config.get('trading_mode') == 'real':
        api_key = config.get('bybit_api_key', '')
        api_secret = config.get('bybit_api_secret', '')

        if not api_key or not api_secret:
            print("WARNING: API credentials are missing, switching to simulation mode")
            config.set('trading_mode', 'simulation')

    # Menu
    print("\nSelect operating mode:")
    print("1. Start trading bot")
    print("2. Run in simulation mode")
    print("3. Test system configuration and API connection")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ")

    if choice == "1":
        # Confirm real trading
        if config.get('trading_mode') == 'real':
            print("\n⚠️  WARNING: You are about to start the bot in REAL TRADING mode ⚠️")
            print("This will use real funds and execute actual trades on BYBIT!")
            confirm = input("\nType 'CONFIRM' to proceed with real trading or anything else to use simulation mode: ")

            if confirm != "CONFIRM":
                print("Switching to simulation mode for safety")
                config.set('trading_mode', 'simulation')

        # Start the bot
        print("\nStarting trading bot...")
        asyncio.run(run_trading_bot(config))

    elif choice == "2":
        # Set simulation mode
        config.set('trading_mode', 'simulation')

        # Ask for duration
        try:
            duration = int(input("\nEnter simulation duration in minutes (default: 60): ") or "60")
        except ValueError:
            duration = 60

        # Set shorter update interval for simulation
        config.set('update_interval', 30)  # 30 seconds

        print(f"\nRunning simulation for {duration} minutes...")

        # Create and start the system
        async def run_simulation():
            system = EnhancedTradingSystem(config)
            await system.initialize()
            await system.start()

            # Run for specified duration
            end_time = datetime.datetime.now() + datetime.timedelta(minutes=duration)

            try:
                while datetime.datetime.now() < end_time and system.running:
                    # Print status every minute
                    status = await system.get_status()
                    print(f"\nTime: {datetime.datetime.now().strftime('%H:%M:%S')}")
                    print(f"Last prediction: {status.get('last_prediction', 'None')}")
                    print(f"Balance: ${status.get('balance', 0):.2f}")
                    print(f"P&L: ${status.get('pnl', 0):.2f} ({status.get('pnl_percent', 0) * 100:.2f}%)")
                    print(f"Trades: {status.get('trades_count', 0)}")

                    await asyncio.sleep(60)  # Status update every minute

            except KeyboardInterrupt:
                print("\nSimulation interrupted...")

            finally:
                # Print final performance report
                metrics = system.get_performance_metrics()

                print("\n" + "=" * 40)
                print("SIMULATION RESULTS:")
                print(f"Total trades: {metrics.get('total_trades', 0)}")
                print(f"Win rate: {metrics.get('win_rate', 0) * 100:.1f}%")
                print(f"Net P&L: ${metrics.get('net_pnl', 0):.2f}")
                print(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")
                print(f"Max drawdown: {metrics.get('max_drawdown', 0) * 100:.1f}%")
                print("=" * 40)

                # Stop the system
                await system.stop()

        asyncio.run(run_simulation())

    elif choice == "3":
        # Test configuration
        print("\nTesting system configuration and API connection...")

        async def run_test():
            # Create system but don't start trading
            system = EnhancedTradingSystem(config)

            # Initialize components
            print("Initializing components...")
            if await system.initialize():
                print("✅ System initialization successful")

                # Test API connection
                print("\nTesting API connection...")
                try:
                    balance = await system._get_account_balance()
                    print(f"✅ API connection successful")
                    print(f"Account balance: ${balance.get('total_balance', 0):.2f}")
                except Exception as e:
                    print(f"❌ API connection failed: {str(e)}")

                # Test market data
                print("\nTesting market data...")
                try:
                    await system.market_data.update_all()
                    price = system.market_data.get_latest_price()
                    print(f"✅ Market data successful")
                    print(f"Current {system.symbol} price: ${price:.2f}")
                except Exception as e:
                    print(f"❌ Market data failed: {str(e)}")

                # Test AI model
                print("\nTesting AI model...")
                try:
                    df_15m = system.market_data.get_data("15")
                    if df_15m is not None and not df_15m.empty:
                        prediction = system.ai_model.predict(df_15m, "15")
                        print(f"✅ AI prediction successful")
                        print(
                            f"Prediction: {prediction.get('action', 'unknown')} with {prediction.get('confidence', 0):.2%} confidence")
                    else:
                        print("❌ AI test failed: No data available")
                except Exception as e:
                    print(f"❌ AI test failed: {str(e)}")

                # Test strategy
                print("\nTesting trading strategy...")
                try:
                    analysis = system.strategy.analyze_market(system.market_data,
                                                              {"overall_action": "hold", "overall_confidence": 0.5})
                    print(f"✅ Strategy analysis successful")
                    print(f"Signal: {analysis.get('entry') or analysis.get('exit') or 'hold'}")
                except Exception as e:
                    print(f"❌ Strategy test failed: {str(e)}")

                # Cleanup
                await system.api_client.close()
            else:
                print("❌ System initialization failed")

            print("\nSystem test completed")

        asyncio.run(run_test())

    elif choice == "4":
        print("Exiting...")

    else:
        print("Invalid choice, exiting...")

    print("\nTrading bot finished. Press Enter to exit...")
    input()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTrading bot interrupted by user")
    except Exception as e:
        print(f"\nUnhandled error: {str(e)}")
        traceback.print_exc()