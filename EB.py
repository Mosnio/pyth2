from datetime import datetime
import logging
import requests
import re
import time
import random
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, GRU, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from urllib3.exceptions import InsecureRequestWarning

start_time = datetime.now()

# Suppress warnings
warnings.simplefilter('ignore', InsecureRequestWarning)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
rprofit = 0

# Configuration
headers = {
    'Host': 'earnbitmoon.club',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7',
    'cache-control': 'max-age=0',
    'referer': 'https://earnbitmoon.club/',
    'sec-ch-ua': '"Not-A.Brand";v="99", "Chromium";v="124"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
}

cookies = {
    '_ga': 'GA1.1.29984595.1700918987',
    'bitmedia_fid': 'eyJmaWQiOiIwZjI3MTQzMTdhMjVjZDk4MmZhMzIwNDFkN2YxMjgyZSIsImZpZG5vdWEiOiIxMmJjZGZjZWFmYWY3OGNkZjhiYTRjODlmZTcyMGJmYSJ9',
    'uuid': 'P84fe6598-06e0-4de6-93ed-6bf223a9b4f0',
    '_fbp': 'fb.1.1709306956970.1744135557',
    'PHPSESSID': '4h4u8se48tqe1kfct0i6mvh22k',
    '_gcl_au': '1.1.155664092.1717503155.2090946952.1718803768.1718803800',
    'SesHashKey': 'n3cbzqbekci5z7ed',
    'SesToken': 'ses_id%3D484517%26ses_key%3Dn3cbzqbekci5z7ed',
    'AccExist': '484517',
    '_ga_7Z81E54NN3': 'GS1.1.1718803757.12.1.1718804286.0.0.0'
}

# Fetch the token
def fetch_token(url):
    try:
        response = requests.get(url, headers=headers, cookies=cookies, verify=False)
        token_pattern = re.compile(r"var token = '([a-f0-9]+)';")
        match = token_pattern.search(response.text)
        token = match.group(1) if match else None

        if not token:
            raise ValueError("Token not found.")
        
        return token
    except requests.RequestException as e:
        logging.error(f"Network error: {e}")
        sys.exit()
    except ValueError as e:
        logging.error(e)
        sys.exit()

# Initialize variables
def initialize_variables():
    return {
        "historical_results": [],
        "wrong_predictions_count": 0,
        "bet_amount": 5.00,
        "max_bet_amount": 5000000,
        "profit": 0,
        "url": 'https://142.132.197.179/system/ajax.php',
        "max_wrong_predictions": random.randint(1, 3),
        "T_profit": 0,
        "danger_zone": False
    }

vars = initialize_variables()

# Encode results
def encode_result(result):
    return 1 if result == 'BTC' else 0

# Decode prediction
def decode_prediction(prediction):
    return 'BTC' if prediction > 0.5 else 'ETH'

# Base class for Keras models for scikit-learn
class KerasBaseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, epochs=10, batch_size=32, **params):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.params = params
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = self.build_fn(**self.params)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, **kwargs)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def get_params(self, deep=True):
        return {"build_fn": self.build_fn, "epochs": self.epochs, "batch_size": self.batch_size, **self.params}

    def set_params(self, **params):
        self.epochs = params.pop('epochs', self.epochs)
        self.batch_size = params.pop('batch_size', self.batch_size)
        self.params.update(params)
        return self

# Create and compile the LSTM model with dropout layers
def create_model(optimizer='adam', neurons=128, dropout_rate=0.2, reg=0.01):
    model = Sequential()
    model.add(Input(shape=(1, 1)))
    model.add(LSTM(neurons, return_sequences=True, kernel_regularizer=l2(reg)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, kernel_regularizer=l2(reg)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Additional model for ensemble method
def create_gru_model(optimizer='adam', neurons=128, dropout_rate=0.2, reg=0.01):
    model = Sequential()
    model.add(Input(shape=(1, 1)))
    model.add(GRU(neurons, return_sequences=True, kernel_regularizer=l2(reg)))
    model.add(Dropout(dropout_rate))
    model.add(GRU(neurons, kernel_regularizer=l2(reg)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning function
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'epochs': [10, 20, 30],
        'batch_size': [10, 20, 40],
        'optimizer': ['SGD', 'Adam', 'Adagrad'],
        'neurons': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4],
        'reg': [0.01, 0.001, 0.0001]
    }

    model = KerasBaseModel(build_fn=create_model)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    return grid_result.best_params_

# Train the model with historical data
def train_model(historical_results):
    if len(historical_results) < 50:  # Increased the required amount of data
        logging.info(f"Not enough data to train the model. Data size: {len(historical_results)}")
        return None, None

    scaler = MinMaxScaler()
    historical_results_scaled = scaler.fit_transform(np.array(historical_results).reshape(-1, 1))

    X_train, X_val, y_train, y_val = train_test_split(
        historical_results_scaled[:-1].reshape(-1, 1, 1), 
        historical_results_scaled[1:], 
        test_size=0.2, 
        random_state=42
    )

    logging.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    best_params = tune_hyperparameters(X_train, y_train)
    logging.info(f"Best params: {best_params}")
    
    lstm_model = create_model(optimizer=best_params['optimizer'], neurons=best_params['neurons'], dropout_rate=best_params['dropout_rate'], reg=best_params['reg'])
    gru_model = create_gru_model(optimizer=best_params['optimizer'], neurons=best_params['neurons'], dropout_rate=best_params['dropout_rate'], reg=best_params['reg'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                   epochs=best_params['epochs'], batch_size=best_params['batch_size'], 
                   verbose=1, callbacks=[early_stopping])
    
    gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                  epochs=best_params['epochs'], batch_size=best_params['batch_size'], 
                  verbose=1, callbacks=[early_stopping])
    
    return lstm_model, gru_model

# Predict the next outcome based on historical results
def predict_next_outcome(models, historical_results):
    if len(historical_results) < 50:  # Increased the required amount of data
        return random.choice(['BTC', 'ETH'])

    scaler = MinMaxScaler()
    historical_results_scaled = scaler.fit_transform(np.array(historical_results).reshape(-1, 1))

    X_pred = historical_results_scaled[-1].reshape(-1, 1, 1)
    
    lstm_model, gru_model = models
    lstm_prediction = lstm_model.predict(X_pred)[0][0]
    gru_prediction = gru_model.predict(X_pred)[0][0]
    
    # Simple averaging ensemble
    average_prediction = (lstm_prediction + gru_prediction) / 2

    return decode_prediction(average_prediction)

# Adjust bet amount based on confidence and past performance
def adjust_bet_amount(bet_amount, max_bet_amount, max_wrong_predictions, wrong_predictions_count, confidence):
    if wrong_predictions_count >= max_wrong_predictions:
        if confidence > 0.75:
            bet_amount *= random.uniform(1.5, 1.7)
        elif confidence > 0.5:
            bet_amount *= random.uniform(1.2, 1.4)
        elif confidence > 0.25:
            bet_amount *= random.uniform(0.9, 1.1)
        else:
            bet_amount *= random.uniform(0.5, 0.7)

        bet_amount = min(bet_amount, max_bet_amount)
    bet_amount = max(bet_amount, 5)
    return bet_amount

# Main loop execution
def main_loop():
    token = fetch_token('https://142.132.197.179/flip.html')
    time.sleep(2)
    global vars
    global rprofit
    danger_zone = False
    models = (None, None)
    
    while True:
        # Ensure models are trained only when there are sufficient historical results
        if len(vars['historical_results']) >= 50:  # Increased the required amount of data
            models = train_model(vars['historical_results'])
        
        if models and models[0] and models[1]:
            next_bet = predict_next_outcome(models, vars['historical_results'])
        else:
            next_bet = random.choice(['BTC', 'ETH'])

        data = {
            'a': 'coinFlip',
            'token': token,
            'betAmount': str(vars['bet_amount']),
            'coin': next_bet
        }

        try:
            response = requests.post(vars['url'], headers=headers, data=data, verify=False, cookies=cookies)
            data = response.json()
        except requests.RequestException as e:
            logging.error(f"Network error: {e}")
            time.sleep(random.randint(5, 7))
            continue
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            time.sleep(random.randint(5, 7))
            continue

        result = data['result']
        prft = data['profit']
        rprofit += float(prft)
        vars['profit'] += float(prft)
        vars['T_profit'] += float(prft)
        vars['historical_results'].append(encode_result(result))

        # Update wrong predictions count
        if len(vars['historical_results']) > 1 and encode_result(result) != vars['historical_results'][-2]:
            vars['wrong_predictions_count'] += 1
        else:
            vars['wrong_predictions_count'] = 0

        # Adjust the bet amount based on the model's confidence
        if models and models[0]:
            confidence = models[0].predict(np.array([encode_result(result)]).reshape(-1, 1, 1))[0][0]
        else:
            confidence = 0.5  # Default confidence if no model is available

        vars['bet_amount'] = adjust_bet_amount(vars['bet_amount'], vars['max_bet_amount'], vars['max_wrong_predictions'], vars['wrong_predictions_count'], confidence)

        # Reset conditions
        if vars['profit'] >= 0:
            vars['profit'] = 0
            vars['bet_amount'] = 5
            vars['max_wrong_predictions'] = random.randint(1, 2)

        if vars['T_profit'] <= -100 and float(vars['bet_amount']) > 50 and not danger_zone:
            danger_zone = True
            logging.warning("Entered Danger Zone")
        elif danger_zone and (vars['T_profit'] >= -50 or vars['T_profit'] >= 0):
            vars['bet_amount'] = 5
            danger_zone = False
            logging.info("Exited Danger Zone")

        if vars['T_profit'] >= random.randint(10, 30):
            time.sleep(10)
            vars = initialize_variables()
            token = fetch_token('https://142.132.197.179/flip.html')
            logging.info('Reset done')
            logging.info('***' * 24)
            time.sleep(random.randint(10, 20))
            continue

        logging.info(f'Overall profit: {rprofit}')
        logging.info(f'T.PROFIT: {vars["T_profit"]}')
        logging.info('-#-' * 24)
        logging.info(f'NEXT bet amount: {vars["bet_amount"]}')
        logging.info(f'Max wrong predictions for this round: {vars["max_wrong_predictions"]}')
        if float(rprofit) >= 50:
            current_time = datetime.now()
            formatted_time = current_time.strftime("%H:%M:%S")
            logging.info(f"Ending Time: {formatted_time}")

            end_time = datetime.now()
            total_time = end_time - start_time
            logging.info(f"Total Time Consumed: {total_time}")

            sys.exit('time to sleep')
        else:
            pass
        # Sleep for a random interval
        time.sleep(random.randint(5, 7))

if __name__ == "__main__":
    main_loop()
    
