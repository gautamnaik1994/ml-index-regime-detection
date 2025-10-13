from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import talib as ta
import pickle
import os
import boto3
import logging
import yfinance as yf
from dotenv import load_dotenv
from curl_cffi import requests
from telegram import TelegramService
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

s3 = boto3.client('s3')
bucket_name = os.getenv('S3_BUCKET_NAME')
session = requests.Session(impersonate="chrome")

telegram = TelegramService(
    bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""))

if not bucket_name:
    raise ValueError("S3_BUCKET_NAME environment variable not set")


def prepare_features(df, scaler):
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    df['momentum_10d'] = ta.MOM(df['Close'], timeperiod=10)
    df['ma_ratio'] = df['Close'] / ta.SMA(df['Close'], timeperiod=50)

    # Risk Features
    df['vol_10d'] = df['return_1d'].rolling(10).std()
    df['vol_20d'] = df['return_1d'].rolling(20).std()
    df['vol_ratio'] = df['vol_10d'] / df['vol_20d']
    df['atr_14'] = ta.ATR(df['High'], df['Low'],
                          df['Close'], timeperiod=14) / df['Close']

    # Stress Features
    df['drawdown'] = df['Close'] / df['Close'].cummax() - 1

    # Exhaustion Features
    df['rsi_14'] = ta.RSI(df['Close'], timeperiod=14)

    # Collect feature set
    features_orig = df[['return_1d', 'return_5d', 'momentum_10d', 'ma_ratio',
                        'vol_10d', 'vol_20d', 'vol_ratio', 'atr_14',
                        'drawdown', 'rsi_14']].dropna()

    X_scaled = scaler.transform(features_orig)

    return X_scaled


def predict_regime(model, df):
    return model.predict(df[-1].reshape(1, -1))[0]


def download_model():
    logger.info("Downloading model and scaler from S3...")
    s3.download_file(bucket_name, 'nifty_regime_model.pkl',
                     '/tmp/nifty_regime_model.pkl')
    with open('/tmp/nifty_regime_model.pkl', 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    logger.info("Model and scaler downloaded from S3.")
    return model, scaler


def download_recent_nifty_data():
    ticker = "^NSEI"
    df_obj = yf.Ticker(ticker, session=session)
    data = df_obj.history(period="3mo", interval="1d")
    return data


def handler(event, context):
    # track execution time
    # start_time = time.time()
    logger.info("Starting Nifty Regime Detection inference process...")
    telegram.send_message(
        "Starting Nifty Regime Detection inference process...")

    model, scaler = download_model()
    df = download_recent_nifty_data()
    processed_features = prepare_features(df, scaler)
    regime = predict_regime(model, processed_features)

    telegram.send_message(f"Latest detected regime: {regime}")
    logger.info(f"Latest detected regime: {regime}")
    # log execution time
    # end_time = time.time()
    # execution_time = end_time - start_time
    # logger.info(f"Execution time: {execution_time} seconds")
    # telegram.send_message(f"Execution time: {execution_time} seconds")

    return {
        'statusCode': 200,
        'body': {
            'regime': regime
        }
    }


if __name__ == "__main__":
    handler(None, None)
