import keras._tf_keras.keras as keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, LSTM, Conv1D, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import os
from keras._tf_keras.keras.optimizers.schedules import ExponentialDecay
from keras._tf_keras.keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_preprocess_data(filename):
    # Load data, make first column the index.
    df = pd.read_csv(filename, header=0, index_col=0)
    
    # Store original index name for debugging if needed.
    original_index_name_for_debug = df.index.name if df.index.name else "Unnamed"

    try:
        # Attempt specific format first if known, otherwise general conversion
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
    except (TypeError, ValueError): # Catches if index is not string-like or format is wrong
        df.index = pd.to_datetime(df.index, errors='coerce') # General fallback

    # Drop rows where date index parsing failed (index became NaT)
    df = df[df.index.notna()] 
    
    if df.empty:
        raise ValueError(
            f"DataFrame is empty after reading '{filename}' and removing rows with invalid dates from the index. "
            f"Original index name (header of first CSV column) was '{original_index_name_for_debug}'. Check the CSV's date column for parsable dates."
        )

    # Ensure the index is now a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Index of DataFrame from '{filename}' is not a DatetimeIndex after conversion and NaT removal. "
            f"Type: {type(df.index)}. Original index header: '{original_index_name_for_debug}'."
        )

    # Standardize the index name to 'Date' before reset_index
    df.index.name = 'Date'
    
    df = df.reset_index() # 'Date' index becomes a column named 'Date'
    
    # The first column (df.columns[0]) should now be 'Date'.
    # A robust check and rename if needed, though ideally not necessary if above logic is sound.
    if df.columns[0] != 'Date':
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            
    # Ensure the 'Date' column (now definitely a column) is in datetime format and drop any remaining NaT.
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True) 
    
    if df.empty:
        raise ValueError(
            f"DataFrame is empty after all date processing for '{filename}'. "
            f"Original index header: '{original_index_name_for_debug}'. Check data integrity."
        )
        
    df = df.sort_values('Date')
    
    # Convert key financial columns to numeric, coercing errors to NaN
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # This case should ideally be caught by later feature checks, but good to be aware
            print(f"Warning: Column '{col}' not found for numeric conversion.") 

    # Drop rows that might have NaNs in critical numeric columns after conversion
    # This is important before calculations like .diff() or rolling means
    df.dropna(subset=numeric_cols, inplace=True) 

    if df.empty:
        raise ValueError(
            f"DataFrame is empty after converting financial columns to numeric and dropping NaNs for '{filename}'. "
            f"Check for non-numeric data in 'Open', 'High', 'Low', 'Close', 'Volume' columns. "
            f"Original index header: '{original_index_name_for_debug}'."
        )

    if 'Close' not in df.columns:
        raise ValueError(f"Required 'Close' column not found in the DataFrame from '{filename}' after date processing.")

    # Calculate daily returns and direction (1 if price goes up, 0 if down)
    df['Next_Close'] = df['Close'].shift(-1)
    df['Direction'] = (df['Next_Close'] > df['Close']).astype(int)
    
    # Technical indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) 
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20DaySTD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['20DaySTD'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['20DaySTD'] * 2)
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    
    # ADVANCED INDICATORS - Adding more sophisticated features
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Average Directional Index (ADX)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].shift(1) - df['Low']
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    atr = df['ATR']
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(window=14).mean()
    
    # Price Rate of Change
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
    df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
    
    # Volume Rate of Change
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5) * 100
    
    # Chaikin Money Flow
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
    mfv = mfm * df['Volume']
    df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    
    # Fibonacci retracement levels based on recent high-low range
    high_low_diff = df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min()
    recent_high = df['High'].rolling(window=20).max()
    df['Fib_0.382'] = recent_high - 0.382 * high_low_diff
    df['Fib_0.5'] = recent_high - 0.5 * high_low_diff
    df['Fib_0.618'] = recent_high - 0.618 * high_low_diff
    
    # Volatility - Coefficient of variation
    df['Price_Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
    
    # Trend strength - Directional Movement Index components
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    
    # Drop NaN values created by rolling windows and Next_Close
    df = df.dropna()
    
    if df.empty:
        raise ValueError(
            f"DataFrame is empty after calculating indicators and dropping NaNs for '{filename}'. "
            "Insufficient data or too many NaNs generated. "
            f"Original index header: '{original_index_name_for_debug}'."
        )

    # Extended features list including new advanced indicators
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10',
            'RSI', 'MACD', 'Signal_Line',
            'Upper_Band', 'Lower_Band',
            'Momentum', 'OBV',
            '%K', '%D', 'ATR', 'ADX',
            'ROC_5', 'ROC_10', 'ROC_20',
            'Volume_ROC', 'CMF',
            'Fib_0.382', 'Fib_0.5', 'Fib_0.618',
            'Price_Volatility', 'Plus_DI', 'Minus_DI']

    
    # Check if all feature columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing feature columns for '{filename}': {missing_features}. "
            "Check CSV content and indicator calculations. "
            f"Original index header: '{original_index_name_for_debug}'."
        )
        
    # Scale the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    y = df['Direction'].values
    
    return X, y, df, features

def create_train_test_splits(X, y, test_size=0.2):
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

# Hyperparameters for Stock Market Prediction
EPOCHS = 1000
BATCH_SIZE = 32              # smaller batches often work better for time-series
DROPOUT_RATE = 0.2           # lower dropout helps retain signal in volatile data
L2_REG = 1e-5                # gentle regularization to avoid underfitting
INIT_LR = 1e-3               # good starting point for learning
MIN_LR = 1e-6                # a safe floor for LR decay


def build_model(input_shape):
    # Define sequence length for LSTM - using last 10 time steps
    timesteps = 10
    num_features = input_shape
    
    # Reshape data for LSTM if there's enough data
    if input_shape > 0:  # Ensure we have features
        model = Sequential([
            # 1D Convolutional layer to extract local patterns
            Dense(128, activation='swish', kernel_regularizer=l2(L2_REG), input_shape=(num_features,)),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            
            # Hidden layers with increasing regularization
            Dense(64, activation='swish', kernel_regularizer=l2(L2_REG*1.5)),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            
            Dense(32, activation='swish', kernel_regularizer=l2(L2_REG*2)),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])

        # Use a learning rate schedule for better convergence
        lr_schedule = ExponentialDecay(
            initial_learning_rate=INIT_LR,
            decay_steps=10000,
            decay_rate=0.9
        )
        
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        return model
    else:
        raise ValueError("Input shape must be positive to build model")

def build_lstm_model(X_train):
    # Reshape data for LSTM [samples, timesteps, features]
    timesteps = 10  # Look back 10 days
    num_features = X_train.shape[1]
    
    # We need to reshape the data for LSTM
    # First ensure we have enough data
    if X_train.shape[0] <= timesteps:
        print("Not enough data for LSTM model")
        return None
        
    # Reshape the data - create sequences
    X_lstm = []
    y_lstm = []
    
    for i in range(timesteps, len(X_train)):
        X_lstm.append(X_train[i-timesteps:i])
        y_lstm.append(X_train[i])  # Next day's features
        
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, kernel_regularizer=l2(L2_REG), 
             input_shape=(timesteps, num_features)),
        Dropout(DROPOUT_RATE),
        
        LSTM(32, kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        
        Dense(16, activation='swish', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE/2),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=INIT_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model, X_lstm, y_lstm

def build_random_forest(X_train, y_train):
    """Build and train a Random Forest model for comparison"""
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    return rf_model

def plot_history(history):
    plt.figure(figsize=(16, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    
    # Plot AUC if available
    if 'auc' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Training & Validation AUC')
        plt.legend()
    
    # Plot Precision/Recall if available
    if 'precision' in history.history and 'recall' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['precision'], label='Train Precision')
        plt.plot(history.history['val_precision'], label='Val Precision')
        plt.plot(history.history['recall'], label='Train Recall')
        plt.plot(history.history['val_recall'], label='Val Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision & Recall')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Get raw predictions (probabilities)
    raw_predictions = model.predict(X_test)
    
    # Print raw prediction stats to see distribution
    print("\nRaw prediction statistics:")
    print(f"Min: {np.min(raw_predictions):.4f}")
    print(f"Max: {np.max(raw_predictions):.4f}")
    print(f"Mean: {np.mean(raw_predictions):.4f}")
    print(f"Std: {np.std(raw_predictions):.4f}")
    
    # Adjust threshold dynamically based on prediction distribution
    # Use the median as a threshold if there's enough variation
    if np.std(raw_predictions) > 0.05:
        threshold = np.median(raw_predictions)
    else:
        # If little variation, use a threshold that gives balanced class distribution
        threshold = np.percentile(raw_predictions, 50)  # Start with median
        
        # Adjust if too imbalanced
        pred_test = (raw_predictions > threshold).astype(int)
        ones_ratio = np.mean(pred_test)
        
        # Target a ratio between 0.4 and 0.6
        if ones_ratio < 0.4:
            threshold = np.percentile(raw_predictions, 60)  # Lower threshold to get more 1s
        elif ones_ratio > 0.6:
            threshold = np.percentile(raw_predictions, 40)  # Raise threshold to get more 0s
    
    print(f"Using threshold: {threshold:.4f}")
    
    # Apply the threshold
    y_pred = (raw_predictions > threshold).astype(int)
    
    # Print prediction stats to verify variety
    pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
    print("\nPrediction distribution:")
    for val, count in zip(pred_unique, pred_counts):
        print(f"Class {val}: {count} ({count/len(y_pred)*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

def evaluate_rf_model(model, X_test, y_test):
    """Evaluate Random Forest model"""
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Dynamically adjust threshold
    threshold = np.median(y_pred_proba)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    print("\n--- Random Forest Model Evaluation ---")
    print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importances = model.feature_importances_
    
    return y_pred, y_pred_proba, feature_importances

def create_train_test_splits(X, y, test_size=0.15):  # Reduced test size
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

def plot_predictions(df, y_test, y_pred, test_size=0.15):
    # Get the dates for the test set
    split_idx = int(len(df) * (1 - test_size))
    test_dates = df['Date'].iloc[split_idx:split_idx + len(y_test)]
    
    # Ensure dimensions match
    if len(test_dates) != len(y_test):
        print(f"Warning: Dimension mismatch in plot_predictions - test_dates: {len(test_dates)}, y_test: {len(y_test)}")
        # Use the smaller length to avoid dimension errors
        min_len = min(len(test_dates), len(y_test))
        test_dates = test_dates[:min_len]
        y_test = y_test[:min_len]
        y_pred = y_pred[:min_len]
    
    # Plot actual vs predicted directions
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, y_test, label='Actual Direction', alpha=0.7)
    plt.plot(test_dates, y_pred, label='Predicted Direction', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.title('Actual vs Predicted Stock Price Directions')
    plt.legend()
    plt.show()

def display_trading_signals(test_df_slice, predicted_directions):
    """
    Displays trading signals based on model predictions.

    Args:
        test_df_slice (pd.DataFrame): Slice of the DataFrame corresponding to the test set. 
                                      Must contain 'Date' and 'Close' columns.
        predicted_directions (np.array): Array of predicted directions (0 or 1) for the test set.
    """
    if len(test_df_slice) != len(predicted_directions):
        print("Warning: Length mismatch between test_df_slice and predicted_directions.")
        min_len = min(len(test_df_slice), len(predicted_directions))
        test_df_slice = test_df_slice.iloc[:min_len]
        predicted_directions = predicted_directions[:min_len]
    
    # Create the signals DataFrame - directly use the DataFrame instead of .values
    signals_df = pd.DataFrame()
    signals_df['Date'] = test_df_slice['Date']
    signals_df['Close'] = test_df_slice['Close']
    signals_df['Predicted_Direction'] = predicted_directions
    
    # Define trading signal
    # 1 -> Buy, 0 -> Sell
    signals_df['Signal'] = signals_df['Predicted_Direction'].apply(lambda x: 'Buy' if x == 1 else 'Sell')

    print("\n--- Trading Signals ---")
    print("Based on the model's prediction for the next day's price movement:")
    
    # Format date for better display
    if pd.api.types.is_datetime64_any_dtype(signals_df['Date']):
        signals_df['Date'] = signals_df['Date'].dt.strftime('%Y-%m-%d')
    
    print(signals_df[['Date', 'Close', 'Predicted_Direction', 'Signal']].head(10))
    print(f"Total signals: {len(signals_df)} ({len(signals_df[signals_df['Signal'] == 'Buy'])} Buy, {len(signals_df[signals_df['Signal'] == 'Sell'])} Sell)")
    print("-----------------------")

    # Plot the signals
    plt.figure(figsize=(15, 6))
    plt.plot(test_df_slice['Date'], test_df_slice['Close'], label='Close Price', alpha=0.7)

    buy_signals = signals_df[signals_df['Signal'] == 'Buy']
    sell_signals = signals_df[signals_df['Signal'] == 'Sell']

    # Convert back to datetime for plotting if needed
    if not pd.api.types.is_datetime64_any_dtype(buy_signals['Date']):
        buy_signals['Date'] = pd.to_datetime(buy_signals['Date'])
        sell_signals['Date'] = pd.to_datetime(sell_signals['Date'])

    plt.scatter(buy_signals['Date'], buy_signals['Close'], label='Buy Signal', marker='^', color='green')
    plt.scatter(sell_signals['Date'], sell_signals['Close'], label='Sell Signal', marker='v', color='red')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Trading Signals Based on Predictions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return signals_df

def plot_feature_importance(model, X_test, y_test, feature_names):
    """
    Plot feature importance by measuring prediction changes when features are zeroed out.
    This helps identify which features are most important for predictions.
    """
    # Make a copy of the test data
    X_baseline = X_test.copy()
    
    # Get baseline predictions
    baseline_pred = model.predict(X_baseline)
    
    # Store importance scores
    importance_scores = []
    
    # For each feature, zero it out and measure impact
    for i in range(X_test.shape[1]):
        # Make a copy of the data
        X_modified = X_test.copy()
        
        # Zero out the feature
        X_modified[:, i] = 0
        
        # Get new predictions
        new_pred = model.predict(X_modified)
        
        # Calculate mean absolute difference in predictions
        importance = np.mean(np.abs(baseline_pred - new_pred))
        importance_scores.append(importance)
    
    # Convert to numpy array
    importance_scores = np.array(importance_scores)
    
    # Sort features by importance
    sorted_idx = importance_scores.argsort()
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), importance_scores[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Mean absolute difference in predictions when feature is zeroed")
    plt.tight_layout()
    plt.show()
    
    # Print importance values
    print("\nFeature Importance:")
    for i in sorted_idx[::-1]:  # Reversed to show most important first
        print(f"{feature_names[i]}: {importance_scores[i]:.6f}")

def plot_rf_feature_importance(rf_model, feature_names):
    """Plot Random Forest feature importance"""
    # Get feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.title('Random Forest Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Print top 15 features
    print("\nTop 15 features by importance:")
    for i in range(min(15, len(importances))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.6f}")

def ensemble_predictions(nn_pred_proba, rf_pred_proba, weights=(0.5, 0.5)):
    """Combine predictions from multiple models with weighted average"""
    # Weighted average of probabilities
    ensemble_proba = weights[0] * nn_pred_proba + weights[1] * rf_pred_proba
    
    # Use median as threshold for binary prediction
    threshold = np.median(ensemble_proba)
    ensemble_pred = (ensemble_proba > threshold).astype(int)
    
    return ensemble_pred, ensemble_proba

def main(filename):
    X, y, df, features = load_and_preprocess_data(filename)
    
    # Enhanced class weighting
    class_weights = {0: 1., 1: len(y)/(2*np.bincount(y)[1])}
    
    X_train, X_test, y_train, y_test = create_train_test_splits(X, y)
    
    # Build and train DNN model
    nn_model = build_model(X_train.shape[1])
    
    # Enhanced callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_auc', patience=50, mode='max', 
                     restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=20,
                         min_lr=MIN_LR, mode='max')
    ]
    
    # Train with larger batches and more epochs
    history = nn_model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Build and train Random Forest model
    rf_model = build_random_forest(X_train, y_train)
    
    # Save models
    nn_model.save('models/nn_model.h5')
    joblib.dump(rf_model, 'models/rf_model.joblib')
    
    # Evaluation and visualization
    print("\n--- Neural Network Model Evaluation ---")
    plot_history(history)
    nn_pred = evaluate_model(nn_model, X_test, y_test)
    nn_pred_proba = nn_model.predict(X_test).flatten()
    
    print("\n--- Random Forest Model Evaluation ---")
    rf_pred, rf_pred_proba, rf_importances = evaluate_rf_model(rf_model, X_test, y_test)
    
    # Feature importance for both models
    plot_feature_importance(nn_model, X_test, y_test, features)
    plot_rf_feature_importance(rf_model, features)
    
    # Ensemble the models
    print("\n--- Ensemble Model Evaluation ---")
    ensemble_pred, ensemble_proba = ensemble_predictions(nn_pred_proba, rf_pred_proba, weights=(0.6, 0.4))
    
    # Evaluate ensemble
    print("\nEnsemble Model Accuracy:", np.mean(ensemble_pred == y_test))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred))
    
    # Plot predictions and trading signals
    plot_predictions(df, y_test, ensemble_pred)
    
    split_idx = len(X_train)
    test_df_slice = df.iloc[split_idx:].copy()
    display_trading_signals(test_df_slice, ensemble_pred)
    
    return nn_model, rf_model

# Example usage:
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    data_dir = r'C:\Users\Hassan Elkhatib\Desktop\SE-training\stock\StockPredictorML\data'
    
    # Choose one of the stock files to run
    stock_file = 'NVDA.csv'  # Change to 'NVDA.csv' or 'TSLA.csv' to test other stocks
    
    file_path = os.path.join(data_dir, stock_file)
    print(f"\nRunning prediction model for {stock_file}\n")
    nn_model, rf_model = main(file_path)
    
    # Uncomment these lines to run all stocks
    """
    models = {}
    for stock_file in ['AAPL.csv', 'NVDA.csv', 'TSLA.csv']:
        file_path = os.path.join(data_dir, stock_file)
        print(f"\n\n{'='*50}")
        print(f"Running prediction model for {stock_file}")
        print(f"{'='*50}\n")
        nn_model, rf_model = main(file_path)
        models[stock_file] = {'nn': nn_model, 'rf': rf_model}
    """