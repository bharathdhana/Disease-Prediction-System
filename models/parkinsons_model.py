
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input, Dropout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class ParkinsonsModel:
    def __init__(self):

        self.scaler = MinMaxScaler()
        self.feature_names = ['mdvp_fo', 'mdvp_fhi', 'mdvp_flo', 'mdvp_jitter', 'mdvp_shimmer']
        self.model = None
        self._train_model()

    def check_predictions(self):
        print("\n--- Checking Parkinson's Model Predictions ---")
        test_cases = [
            # Positive Cases (Parkinson's Detected)
            {'features': [119.992, 157.302, 74.997, 0.00784, 0.04374], 'expected': 1, 'desc': 'Positive Case 1'},
            {'features': [122.400, 148.650, 113.819, 0.00968, 0.06134], 'expected': 1, 'desc': 'Positive Case 2'},
            {'features': [116.682, 131.111, 111.555, 0.01050, 0.05233], 'expected': 1, 'desc': 'Positive Case 3'},
            {'features': [116.676, 137.871, 111.366, 0.00997, 0.05492], 'expected': 1, 'desc': 'Positive Case 4'},
            {'features': [116.014, 141.781, 110.655, 0.01284, 0.06425], 'expected': 1, 'desc': 'Positive Case 5'},
            # Negative Cases (No Parkinson's Detected)
            {'features': [197.076, 206.896, 192.055, 0.00289, 0.01098], 'expected': 0, 'desc': 'Negative Case 1'},
            {'features': [199.228, 209.512, 192.091, 0.00241, 0.01015], 'expected': 0, 'desc': 'Negative Case 2'},
            {'features': [198.383, 215.203, 193.104, 0.00212, 0.01263], 'expected': 0, 'desc': 'Negative Case 3'},
            {'features': [202.266, 211.604, 197.079, 0.00180, 0.00954], 'expected': 0, 'desc': 'Negative Case 4'},
            {'features': [203.184, 211.526, 196.160, 0.00178, 0.00958], 'expected': 0, 'desc': 'Negative Case 5'}
        ]
        
        passes = 0
        total = len(test_cases)
        
        for case in test_cases:
            prediction, probability = self.predict(case['features'])
            result_str = "Parkinson's Detected" if prediction == 1 else "No Parkinson's Detected"
            expected_str = "Parkinson's Detected" if case['expected'] == 1 else "No Parkinson's Detected"
            
            status = "PASS" if prediction == case['expected'] else "FAIL"
            if status == "PASS":
                passes += 1
                
            print(f"{status} | {case['desc']} | Expected: {expected_str} | Predicted: {result_str} ({probability[1]:.2%} confidence)")
            
        print(f"\nResult: {passes}/{total} passed")
        return passes == total

    def _build_cnn_model(self):
        model = Sequential()
        model.add(Input(shape=(5, 1)))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train_model(self):
        try:
            df = pd.read_csv('data/parkinsons.csv')
        except FileNotFoundError:
            print("Error: data/parkinsons.csv not found. Please run download_data.py first.")
            return

        features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer']
        X = df[features].values
        y = df['status'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        self.model = self._build_cnn_model()
        history = self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
        
        if not os.path.exists('static/images/plots/parkinsons_viz.png'):
            plt.figure(figsize=(6, 4))
            plt.plot(history.history['accuracy'], label='Train Acc', color='blue')
            plt.plot(history.history['val_accuracy'], label='Val Acc', color='green', linestyle='--')
            plt.title("CNN Training Accuracy (Parkinsons)")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('static/images/plots/parkinsons_viz.png')
            plt.close()

    def predict(self, input_data):
        data = np.array(input_data).reshape(1, -1)
        data_scaled = self.scaler.transform(data)
        data_reshaped = data_scaled.reshape(1, 5, 1)
        prob = self.model.predict(data_reshaped, verbose=0)[0][0]
        prediction = 1 if prob > 0.5 else 0
        return prediction, np.array([1-prob, prob])

    def get_metrics(self):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return {'accuracy': round(acc * 100, 2), 'viz_path': '/static/images/plots/parkinsons_viz.png'}
