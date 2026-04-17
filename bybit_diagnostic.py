#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bybit Trading Bot Diagnostic Script
This script helps diagnose issues with the main trading bot
"""

import sys
import os
import time

print("=== BYBIT TRADING BOT DIAGNOSTIC ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script path: {__file__}")
print("=" * 40)


def check_basic_imports():
    """Test if basic imports are working"""
    print("\n[1] Testing basic Python imports...")
    try:
        import datetime
        print(f"✓ datetime - Current time: {datetime.datetime.now()}")

        import json
        print("✓ json")

        import logging
        print("✓ logging")

        import threading
        print("✓ threading")

        import traceback
        print("✓ traceback")

        print("All basic imports successful.")
        return True
    except Exception as e:
        print(f"✗ Error importing basic modules: {str(e)}")
        return False


def check_numerical_libraries():
    """Test if numerical libraries are working"""
    print("\n[2] Testing numerical libraries...")

    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")

        test_array = numpy.array([1, 2, 3, 4, 5])
        print(f"  Test array: {test_array.mean()}")
    except Exception as e:
        print(f"✗ Error with numpy: {str(e)}")

    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")

        df = pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"  Test dataframe shape: {df.shape}")
    except Exception as e:
        print(f"✗ Error with pandas: {str(e)}")

    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except Exception as e:
        print(f"✗ Error with matplotlib: {str(e)}")

    try:
        import talib
        print(f"✓ talib - Functions: {', '.join(dir(talib)[:5])}...")
    except Exception as e:
        print(f"✗ Error with talib: {str(e)}")
        print("  Note: TA-Lib is difficult to install and requires special steps.")
        print("  See: https://github.com/mrjbq7/ta-lib#dependencies")


def check_tensorflow():
    """Test if TensorFlow is working"""
    print("\n[3] Testing TensorFlow and Keras...")

    try:
        import tensorflow as tf
        print(f"✓ tensorflow {tf.__version__}")

        # Test TensorFlow computation
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[1, 1], [1, 1]])
        print(f"  TF test computation: {tf.reduce_sum(a + b).numpy()}")

        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  GPUs available: {len(gpus)}")
        if gpus:
            for gpu in gpus:
                print(f"    - {gpu}")
        else:
            print("    No GPUs found, using CPU only")

        # Test Keras
        from tensorflow import keras
        print(f"✓ keras {keras.__version__}")

        print("TensorFlow and Keras working correctly.")
    except Exception as e:
        print(f"✗ Error with TensorFlow/Keras: {str(e)}")


def check_crypto_libraries():
    """Test if crypto-specific libraries are working"""
    print("\n[4] Testing crypto libraries...")

    try:
        import pybit
        print(f"✓ pybit - Available modules: {', '.join(dir(pybit)[:5])}...")

        from pybit.unified_trading import HTTP
        print("✓ pybit.unified_trading.HTTP")
    except Exception as e:
        print(f"✗ Error with pybit: {str(e)}")

    try:
        import websocket
        print(f"✓ websocket-client")
    except Exception as e:
        print(f"✗ Error with websocket-client: {str(e)}")


def check_telegram():
    """Test if telegram libraries are working"""
    print("\n[5] Testing Telegram libraries...")

    try:
        import telegram
        print(f"✓ telegram")

        from telegram.ext import ApplicationBuilder
        print("✓ telegram.ext.ApplicationBuilder")
    except Exception as e:
        print(f"✗ Error with python-telegram-bot: {str(e)}")


def test_bybit_connection():
    """Test connection to Bybit API"""
    print("\n[6] Testing Bybit API connection...")

    try:
        from pybit.unified_trading import HTTP

        # Test public API (no credentials needed)
        session = HTTP("https://api.bybit.com")

        # Test simple public endpoint
        response = session.get_tickers(category="spot", symbol="BTCUSDT")

        if response and response.get("retCode") == 0:
            print("✓ Successfully connected to Bybit API")
            print(f"  BTC/USDT price: ${float(response['result']['list'][0]['lastPrice']):.2f}")
            return True
        else:
            print(f"✗ API connection failed: {response}")
            return False
    except Exception as e:
        print(f"✗ Error connecting to Bybit: {str(e)}")
        return False


def main():
    """Main diagnostic function"""
    print("Starting diagnostics...\n")

    # Run all checks
    basic_imports_ok = check_basic_imports()
    if basic_imports_ok:
        check_numerical_libraries()
        check_tensorflow()
        check_crypto_libraries()
        check_telegram()
        api_ok = test_bybit_connection()

    print("\n" + "=" * 40)
    print("Diagnostic complete!")

    if not basic_imports_ok:
        print("\n⚠️ Basic imports failed - fix Python environment first")
    else:
        print("\nRecommendations:")
        print("1. Make sure all required packages are installed with correct versions:")
        print("   - tensorflow==2.16.2 (or closest available for your system)")
        print("   - keras==3.9.0")
        print("   - pybit==5.9.0")
        print("   - pandas==2.2.3")
        print("   - numpy==1.26.4")
        print("   - matplotlib==3.10.1")
        print("   - TA-Lib==0.6.3 (requires special installation steps)")
        print("   - websocket-client==1.8.0")
        print("   - python-telegram-bot==21.9")
        print("   - requests==2.32.3")

        print("\n2. Check errors in your main script:")
        print("   - Add explicit print statements at the beginning")
        print("   - Add try/except blocks around major components")
        print("   - Start with a minimal version of the script")

    print("\nUse this output to resolve issues with your trading bot.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Diagnostic script failed: {str(e)}")
        import traceback

        traceback.print_exc()

    # Keep console open to see results
    input("\nPress Enter to exit...")
