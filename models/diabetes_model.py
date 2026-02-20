
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

class DiabetesModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        self.model = None
        self._train_model()

    def check_predictions(self):
        print("\n--- Checking Diabetes Model Predictions ---")
        test_cases = [
            # Positive Cases (Diabetes Detected)
            {'features': [6, 148, 72, 35, 0, 33.6, 0.627, 50], 'expected': 1, 'desc': 'Positive Case 1'},
            {'features': [8, 183, 64, 0, 0, 23.3, 0.672, 32], 'expected': 1, 'desc': 'Positive Case 2'},
            {'features': [0, 137, 40, 35, 168, 43.1, 2.288, 33], 'expected': 1, 'desc': 'Positive Case 3'},
            {'features': [3, 78, 50, 32, 88, 31.0, 0.248, 26], 'expected': 1, 'desc': 'Positive Case 4'},
            {'features': [2, 197, 70, 45, 543, 30.5, 0.158, 53], 'expected': 1, 'desc': 'Positive Case 5'},
            # Negative Cases (No Diabetes Detected)
            {'features': [1, 85, 66, 29, 0, 26.6, 0.351, 31], 'expected': 0, 'desc': 'Negative Case 1'},
            {'features': [1, 89, 66, 23, 94, 28.1, 0.167, 21], 'expected': 0, 'desc': 'Negative Case 2'},
            {'features': [5, 116, 74, 0, 0, 25.6, 0.201, 30], 'expected': 0, 'desc': 'Negative Case 3'},
            {'features': [10, 115, 0, 0, 0, 35.3, 0.134, 29], 'expected': 0, 'desc': 'Negative Case 4'},
            {'features': [4, 110, 92, 0, 0, 37.6, 0.191, 30], 'expected': 0, 'desc': 'Negative Case 5'}
        ]
        
        passes = 0
        total = len(test_cases)
        
        for case in test_cases:
            prediction, probability = self.predict(case['features'])
            result_str = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"
            expected_str = "Diabetes Detected" if case['expected'] == 1 else "No Diabetes Detected"
            
            status = "PASS" if prediction == case['expected'] else "FAIL"
            if status == "PASS":
                passes += 1
                
            print(f"{status} | {case['desc']} | Expected: {expected_str} | Predicted: {result_str} ({probability[1]:.2%} confidence)")
            
        print(f"\nResult: {passes}/{total} passed")
        return passes == total

    def _build_model(self, input_dim):
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train_model(self):
        try:
            df = pd.read_csv('data/diabetes.csv')
        except FileNotFoundError:
            print("Error: data/diabetes.csv not found. Please run download_data.py first.")
            return

        X = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.model = self._build_model(X.shape[1])
        history = self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        
        if not os.path.exists('static/images/plots/diabetes_viz.png'):
            plt.figure(figsize=(6, 4))
            plt.plot(history.history['loss'], label='Train Loss', color='red', linewidth=2)
            plt.plot(history.history['val_loss'], label='Val Loss', color='orange', linestyle='--')
            plt.title("Deep Neural Network Training Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('static/images/plots/diabetes_viz.png')
            plt.close()

    def predict(self, input_data):
        data = np.array(input_data).reshape(1, -1)
        data_scaled = self.scaler.transform(data)
        prob = self.model.predict(data_scaled, verbose=0)[0][0]
        prediction = 1 if prob > 0.5 else 0
        return prediction, np.array([1-prob, prob])

    def get_metrics(self):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return {"accuracy": round(acc * 100, 2), "viz_path": "/static/images/plots/diabetes_viz.png"}
