# bybit-ml-trading-lab

A research lab of Bybit futures trading bots exploring different ML + TA combinations. Three bot variants (`1.py`, `enhanced_trading_bot.py`, `minimal_trading_bot.py`), a diagnostic script and a training script, built to compare approaches on BTCUSDT across multiple timeframes (1m / 5m / 15m / 1h).

## Scripts

| File | Role |
|------|------|
| `1.py` | Comprehensive TensorFlow-driven futures bot (full-featured reference implementation) |
| `enhanced_trading_bot.py` | Enhanced bot with additional TA + risk features |
| `minimal_trading_bot.py` | Minimal bot: trimmed logic for readability and rapid iteration |
| `bybit_diagnostic.py` | API connectivity / account diagnostic utility |
| `train_model.py` | Model training entry point |

## Data shape

- **Symbol:** BTCUSDT
- **Leverage:** 3x (configurable)
- **Max risk per trade:** 1%
- **Timeframes:** 1m, 5m, 15m, 1h (models per TF stored in `models/`)

## Usage

```bash
pip install -r requirements.txt

# Set credentials via env vars (never hardcode)
export BYBIT_API_KEY=your_key
export BYBIT_API_SECRET=your_secret
export TELEGRAM_BOT_TOKEN=your_bot_token   # optional
export TELEGRAM_CHAT_ID=your_chat_id       # optional

# Train
python train_model.py

# Run a bot variant
python minimal_trading_bot.py
# or
python enhanced_trading_bot.py
```

## Outputs

- `models/` — metadata for per-TF trained models (weights excluded from repo; train locally)
- `charts/` — sample backtest/analysis plots

## ⚠️ Disclaimer

This repository is a research lab intended **for educational and experimentation purposes only** and is **not** financial advice. All three bots default to simulation mode (`TRADING_MODE` / `DEFAULT_TRADING_MODE = "simulation"`); switching to `"real"` places live orders. Leveraged futures trading can result in rapid liquidation. The author assumes no liability for losses.
