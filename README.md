# Polaris AI Trading Bot (CNN-BiGRU + Attention)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00)

## ENGLISH
This AI-based trading system predicts price movements in the Forex market using a deep neural network with Conv1D, BiGRU, and Multi-Head Attention. It selects the top 3 performing models based on ret/DD and uses majority voting for signal generation. Developed by **Roxana Khaleghi**.

### Features
- Deep learning-based time series prediction
- Feature selection (XGBoost)
- Confidence filtering
- Voting logic
- Backtest with real win rate/profit tracking

## فارسی
این ربات معاملاتی با استفاده از یادگیری عمیق، حرکات قیمت در بازار فارکس را پیش‌بینی می‌کند. معماری آن شامل کانولوشن، GRU دوطرفه و توجه چندسری (Multi-Head Attention) است. با تحلیل نسبت سود به افت سرمایه (ret/DD) بهترین مدل‌ها انتخاب می‌شوند و با رأی‌گیری بین آن‌ها سیگنال تولید می‌شود. توسعه‌دهنده: **رکسانا خالقی**

### امکانات
- پیش‌بینی سری‌های زمانی با شبکه عصبی عمیق
- انتخاب ویژگی‌ها با XGBoost
- فیلتر اطمینان سیگنال
- رأی‌گیری بین مدل‌ها
- بک‌تست و ارزیابی وین‌ریت و سود

---

### How to Run / نحوه اجرا:
```bash
python train_and_save_models.py
python voting_and_backtest.py
