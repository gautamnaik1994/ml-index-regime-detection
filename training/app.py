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


def get_nifty_data():
    logger.info("Fetching NIFTY data...")
    s3.download_file(
        Bucket=bucket_name, Key='nifty_data.pkl', Filename='/tmp/nifty_data.pkl')
    with open('/tmp/nifty_data.pkl', 'rb') as f:
        obj = pickle.load(f)
        df = obj["data"]
        metadata = obj["metadata"]
    return df, metadata


def update_nifty_data(df, metadata):
    logger.info("Updating NIFTY data...")
    last_date = pd.to_datetime(metadata['last_date']).replace(
        tzinfo=None) + pd.Timedelta(days=1)
    today = pd.to_datetime('today').normalize()
    if last_date < today:
        logger.info("Updating NIFTY data...")
        try:
            last_date = pd.to_datetime(df.index.max()) + pd.Timedelta(days=1)
            ticker = "^NSEI"
            df_obj = yf.Ticker(ticker, session=session)
            data = df_obj.history(start=last_date.strftime(
                '%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), interval='1d')
            df = pd.concat([df, data])
            metadata['last_date'] = today
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            telegram.send_error(
                f"Error fetching data from Yahoo Finance: {e}")
    return df, metadata


def hyperparameter_tuning(X):
    logger.info("Tuning hyperparameters...")
    best_bic = np.inf
    best_model = None
    clusters_count = None
    for K in range(2, 8):
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            model = GaussianHMM(
                n_components=K, covariance_type=cov_type, n_iter=2000)
            model.fit(X)
            logL = model.score(X)
            n_params = K*(K-1) + K*X.shape[1]*2  # rough estimate
            bic = n_params * np.log(len(X)) - 2*logL
            if bic < best_bic:
                best_bic = bic
                best_model = model
                clusters_count = K
    logger.info(
        f"Best model has {clusters_count} clusters with BIC: {best_bic}")
    telegram.send_message(
        f"Best model has {clusters_count} clusters with BIC: {best_bic}")
    return best_model, clusters_count


def feature_engineering(df):
    logger.info("Performing feature engineering...")
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
    features = df[['return_1d', 'return_5d', 'momentum_10d', 'ma_ratio',
                   'vol_10d', 'vol_20d', 'vol_ratio', 'atr_14',
                   'drawdown', 'rsi_14']].dropna()

    # print first rows of features for debugging
    logger.info(f"Features head:\n{features.head()}")
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    return X, scaler


def save_model(model, scaler, clusters_count):
    logger.info("Saving model and scaler to S3...")
    with open('/tmp/hmm_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'clusters_count': clusters_count
        }, f)
    s3.upload_file('/tmp/hmm_model.pkl', bucket_name, 'nifty_regime_model.pkl')
    logger.info("Model and scaler saved to S3.")


def save_nifty_data(df, metadata):
    logger.info("Saving updated NIFTY data to S3...")
    with open('/tmp/nifty_data.pkl', 'wb') as f:
        pickle.dump({
            'data': df,
            'metadata': metadata
        }, f)
    s3.upload_file('/tmp/nifty_data.pkl', bucket_name, 'nifty_data.pkl')
    logger.info("NIFTY data saved to S3.")


def handler(event, context):
    logger.info("Starting model training process...")
    telegram.send_message("Starting Nifty Regime Detection Model training...")
    df, metadata = get_nifty_data()
    df, metadata = update_nifty_data(df, metadata)
    save_nifty_data(df, metadata)
    X, scaler = feature_engineering(df)
    model, clusters_count = hyperparameter_tuning(X)
    save_model(model, scaler, clusters_count)
    telegram.send_success(
        "Nifty Regime Detection Model training completed successfully!")
    return {
        'statusCode': 200,
        'body': 'Model training completed successfully!'
    }


if __name__ == "__main__":
    handler(None, None)
