import json
import datetime
import os
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import time
import matplotlib.dates as mdates
from datetime import datetime
import telebot
from telebot import types

BOT_TOKEN = "7659905271:AAFq8q30_9rPAopf_dteVqH07ctDGqijSNY"
CHAT_ID = "1056357383"

SYMBOLS = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "TON": "TONUSDT",
    "SOL": "SOLUSDT"
    "XRP": "XRPUSDT"	
  
}
INTERVALS = {
    "15 минут": "15",
    "30 минут": "30",
    "1 час": "60",
    "4 часа": "240".	
    
}
LIMIT = 500

bot = telebot.TeleBot(BOT_TOKEN)
user_state = {}

def exponential_moving_average(data, period):
    return pd.Series(data).ewm(span=period, 
    adjust=False).mean()

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def compute_macd(prices):
    ema12 = exponential_moving_average(prices, 12)
    ema26 = exponential_moving_average(prices, 26)
    macd_line = ema12 - ema26
    signal_line = exponential_moving_average(macd_line, 9)
    return macd_line, signal_line

def build_candle_chart(df, symbol):
    from mplfinance.original_flavor import candlestick_ohlc
    df['date'] = mdates.date2num(df['timestamp'])
    ohlc = df[['date', 'open', 'high', 'low', 'close']]

    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    candlestick_ohlc(ax, ohlc.values, width=0.01, colorup='green', colordown='red')

    ax.plot(df['date'], df['ema14'], label='EMA-14')
    ax.plot(df['date'], df['ema50'], label='EMA-50')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    plt.title(f"Свечной график {symbol}")
    plt.xlabel("Время")
    plt.ylabel("Цена USDT")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    plt.clf()
    plt.cla()
    buf.seek(0)
    return buf

def get_candles(symbol, interval):
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={LIMIT}"
    response = requests.get(url)
    data = response.json()
    candles = data["result"]["list"]
    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df

def get_live_price(symbol):
    url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["result"]["list"][0]["lastPrice"])
    except Exception as e:
        print(f"Ошибка получения live-цены: {e}")
        return None

def send_crypto_menu(chat_id):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    for name in SYMBOLS:
        markup.add(types.KeyboardButton(name))
    bot.send_message(chat_id, "Выберите криптовалюту для анализа:", reply_markup=markup)

def log_request_entry(user_id, username, symbol_code, interval_code):
    stats_file = "user_requests.json"
    request = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "username": username,
        "symbol": symbol_code,
        "interval": interval_code
    }

    try:
        with open(stats_file, "a") as f:  # режим append
            f.write(json.dumps(request) + "\n")  # каждая строка — отдельный JSON
    except Exception as e:
        print(f"Ошибка записи в лог: {e}")
def analyze_and_send(chat_id, symbol_name, symbol_code, interval_code):
    try:
        bot.send_message(chat_id, f"🔍 Анализ {symbol_code} начался...")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                df = get_candles(symbol_code, interval_code)
                df = df[::-1]
                closes = df["close"]
                df["ema14"] = exponential_moving_average(closes, 14)
                df["ema50"] = exponential_moving_average(closes, 50)
                ema14 = df["ema14"].iloc[-1]
                ema50 = df["ema50"].iloc[-1]
                # Поиск последнего пересечения EMA-14 и EMA-50
                df["ema_cross"] = df["ema14"] > df["ema50"]
                df["cross_change"] = df["ema_cross"] != df["ema_cross"].shift()

                last_cross_row = df[df["cross_change"]].iloc[-1] if df["cross_change"].any() else None

               
                   
                   
# Добавляем в сообщение
                
                rsi = compute_rsi(closes)
                macd, signal = compute_macd(closes)
                break  # Успешно, выходим из цикла
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Попытка {attempt + 1} не удалась (пользователь: {chat_id}). Повтор через 5 секунд...")
                    time.sleep(5)
                else:
                    print(f"❌ Не удалось получить данные: {e}")
                    bot.send_message(chat_id, "⚠️ Потеряна связь с сервером (Bybit/Telegram). Повторная попытка...")
                    send_crypto_menu(chat_id)
                    return
      
        # 🛡️ Проверка на пустоту
        if len(macd) == 0 or len(signal) == 0:
            bot.send_message(chat_id, "⚠️ Ошибка анализа: недостаточно данных для расчета MACD.")
            send_crypto_menu(chat_id)
            return
        macd_val, signal_val = macd.iloc[-1], signal.iloc[-1]
        price = get_live_price(symbol_code) or closes.iloc[-1]

        trend = "⬆️ Лонг" if ema14 > ema50 else "⬇️ Шорт"
        direction = "бычий" if macd_val > signal_val else "медвежий"
        # Надёжность сигнала
        strong_signal = False
        if trend == "📈 Лонг" and direction == "бычий" and rsi.iloc[-1] < 70:
            strong_signal = True
        elif trend == "📉 Шорт" and direction == "медвежий" and rsi.iloc[-1] > 30:
            strong_signal = True
        # Определим текст надёжности сигнала
        signal_strength = "✅ Надёжный сигнал" if strong_signal else "⚠️ Слабый сигнал — нужна осторожность"
        if ema14 > ema50:
            # Лонг: TP выше, SL ниже
            take_profit = price * 1.005
            stop_loss = price * 0.995
        else:
            # Шорт: TP ниже, SL выше
            take_profit = price * 0.995
            stop_loss = price * 1.005
        try:
            if last_cross_row is not None:
                       cross_type = "📈 Golden Cross" if last_cross_row["ema14"] > last_cross_row["ema50"] else "📉 Death Cross"
                       cross_msg = (
                       "📊 Пересечение EMA:\n"
                       f"🔹 Тип: {cross_type}\n"
                       f"💰 Цена закрытия: {last_cross_row['close']:.2f} USDT\n"
                       f"🕒 Время: {last_cross_row['timestamp']}\n\n"
                       f"📈 Индикаторы:\n"
                       f"🔹 RSI: {rsi.iloc[-1]:.2f}\n"
                       f"🔹 EMA-14: {ema14:.5f}\n"
                       f"🔹 EMA-50: {ema50:.5f}\n"
                       f"🔹 MACD направление: {direction}\n"
                       f"🔹 MACD значение: {macd_val:.5f}\n"
                       f"🔹 Сигнал MACD: {signal_val:.5f}"

                       )
                       
            else:
                       cross_msg = "❔ Пересечение EMA не обнаружено."
                   
        except Exception as e:
                   cross_msg = f"⚠️ Ошибка при поиске пересечения EMA:\n{e}"
        buf = io.BytesIO()
        

        from mplfinance.original_flavor import candlestick_ohlc
        df['date'] = mdates.date2num(df['timestamp'])
        ohlc = df[['date', 'open', 'high', 'low', 'close']]
        plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        candlestick_ohlc(ax, ohlc.values, width=0.01, colorup='green', colordown='red')
        ax.plot(df['date'], df['ema14'], label='EMA-14')
        ax.plot(df['date'], df['ema50'], label='EMA-50')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
        plt.title(f"Свечной график {symbol_code}")
        plt.xlabel("Время")
        plt.ylabel("Цена USDT")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        bot.send_photo(chat_id, buf, caption=f"📉 Свечной график {symbol_code}")
        bot.send_message(chat_id, cross_msg)
        trend_emoji = "📈" if trend == "Лонг" else "📉"
        analysis_msg = (
        f"📊 Анализ {symbol_code} – {trend_emoji} {trend}\n"
        f"🎯 Точка входа: {price:.5f} USDT\n\n"
    	f"📌 Take-profit: {take_profit:.5f} USDT\n"
    	f"🛑 Stop-loss: {'выше' if trend == 'Шорт' else 'ниже'} {stop_loss:.5f} USDT\n\n"

    	f"{signal_strength}"
	)
        bot.send_message(chat_id, analysis_msg)
        send_crypto_menu(chat_id,)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)  # Вывод в консоль
        bot.send_message(chat_id, f"❌ Ошибка анализа:\n{err_msg}")

        send_crypto_menu(chat_id)

    

@bot.message_handler(commands=["start"])
def start(message):
    send_crypto_menu(message.chat.id)

@bot.message_handler(func=lambda message: message.text in SYMBOLS)
def choose_symbol(message):
    user_state[message.chat.id] = {"symbol": message.text}
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    for interval in INTERVALS:
        markup.add(types.KeyboardButton(interval))
    bot.send_message(message.chat.id, "Теперь выберите таймфрейм:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text in INTERVALS)
def choose_interval(message):
    chat_id = message.chat.id
    interval = INTERVALS[message.text]
    symbol_name = user_state.get(chat_id, {}).get("symbol")
    if not symbol_name:
        bot.send_message(chat_id, "Сначала выберите криптовалюту командой /start")
        return
    symbol_code = SYMBOLS[symbol_name]

    log_request_entry(
        user_id=chat_id,
        username=message.from_user.username or  
    "unknown",
        symbol_code=symbol_code,
        interval_code=interval
    )

    analyze_and_send(chat_id, symbol_name, symbol_code, interval)

if __name__ == '__main__':
    print("Бот запущен. Ожидаем команды /start...")

    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=30)
        except Exception as e:
            print(f"[ОШИБКА poll]: {e}")
            import traceback
            traceback.print_exc()

            # Попробуем отправить уведомление в Telegram
            try:
                bot.send_message(CHAT_ID, "⚠️ Потеряна связь с сервером (Bybit/Telegram). Повторная попытка через 10 секунд...")
            except:
                print("❗️ Невозможно отправить сообщение: бот сам отключён.")

            print("⏳ Повторный запуск через 10 секунд...")
            time.sleep(10)
