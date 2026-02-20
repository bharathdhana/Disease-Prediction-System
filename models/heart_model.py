
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam

class HeartDiseaseModel:
    def __init__(self):

        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        self.input_dim = len(self.feature_names)
        
        self.ann_model = None
        self.cnn_model = None
        
        self._train_model()

    def check_predictions(self):
        print("\n--- Checking Heart Disease Model Predictions ---")
        test_cases = [
            # Positive Cases (Heart Disease)
            {'features': [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1], 'expected': 1, 'desc': 'Positive Case 1'},
            {'features': [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2], 'expected': 1, 'desc': 'Positive Case 2'},
            {'features': [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2], 'expected': 1, 'desc': 'Positive Case 3'},
            {'features': [56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2], 'expected': 1, 'desc': 'Positive Case 4'},
            {'features': [57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2], 'expected': 1, 'desc': 'Positive Case 5'},
            # Negative Cases (No Heart Disease)
            {'features': [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2], 'expected': 0, 'desc': 'Negative Case 1'},
            {'features': [67, 1, 0, 120, 229, 0, 0, 129, 1, 2.6, 1, 2, 3], 'expected': 0, 'desc': 'Negative Case 2'},
            {'features': [62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2], 'expected': 0, 'desc': 'Negative Case 3'},
            {'features': [63, 1, 0, 130, 254, 0, 0, 147, 0, 1.4, 1, 1, 3], 'expected': 0, 'desc': 'Negative Case 4'},
            {'features': [53, 1, 0, 140, 203, 1, 0, 155, 1, 3.1, 0, 0, 3], 'expected': 0, 'desc': 'Negative Case 5'}
        ]
        
        passes = 0
        total = len(test_cases)
        
        for case in test_cases:
            prediction, probability = self.predict(case['features'])
            result_str = "Heart Disease" if prediction == 1 else "No Heart Disease"
            expected_str = "Heart Disease" if case['expected'] == 1 else "No Heart Disease"
            
            status = "PASS" if prediction == case['expected'] else "FAIL"
            if status == "PASS":
                passes += 1
                
            print(f"{status} | {case['desc']} | Expected: {expected_str} | Predicted: {result_str} ({probability[1]:.2%} confidence)")
            
        print(f"\nResult: {passes}/{total} passed")
        return passes == total

    def _build_ann_model(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn_model(self):
        model = Sequential([
            Input(shape=(self.input_dim, 1)),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Dropout(0.2),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train_model(self):

        try:
            df = pd.read_csv('data/heart.csv')
        except FileNotFoundError:
            print("Error: data/heart.csv not found. Please run download_data.py first.")
            return


        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.ann_model = self._build_ann_model()
        history_ann = self.ann_model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
        
        X_train_cnn = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test_cnn = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        
        self.cnn_model = self._build_cnn_model()
        history_cnn = self.cnn_model.fit(X_train_cnn, self.y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
        
        if not os.path.exists('static/images/plots/heart_viz.png'):
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history_ann.history['accuracy'], label='ANN Train')
            plt.plot(history_ann.history['val_accuracy'], label='ANN Val')
            plt.title('ANN Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history_cnn.history['accuracy'], label='CNN Train')
            plt.plot(history_cnn.history['val_accuracy'], label='CNN Val')
            plt.title('CNN Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('static/images/plots/heart_viz.png')
            plt.close()

    def predict(self, input_data):
        data = np.array(input_data).reshape(1, -1)
        
        # Scale the data using the fitted scaler
        data = self.scaler.transform(data)
        
        ann_prob = self.ann_model.predict(data, verbose=0)[0][0]
        
        data_cnn = data.reshape((1, data.shape[1], 1))
        cnn_prob = self.cnn_model.predict(data_cnn, verbose=0)[0][0]
        
        avg_prob = (ann_prob + cnn_prob) / 2
        prediction = 1 if avg_prob > 0.5 else 0
        
        return prediction, np.array([1-avg_prob, avg_prob])

    def get_metrics(self):
        loss, acc_ann = self.ann_model.evaluate(self.X_test, self.y_test, verbose=0)
        
        X_test_cnn = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        loss_cnn, acc_cnn = self.cnn_model.evaluate(X_test_cnn, self.y_test, verbose=0)
        
        avg_acc = (acc_ann + acc_cnn) / 2
        return {"accuracy": round(avg_acc * 100, 2), "viz_path": "/static/images/plots/heart_viz.png"}
