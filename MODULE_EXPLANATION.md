# Disease Prediction System - Module Explanation

This document provides a detailed technical explanation of the first two modules in the Disease Prediction System.

## Module 1: Heart Disease Prediction
**Source File:** [heart_model.py](file:///d:/code3/disease_prediction/models/heart_model.py)

### 1. Neural Network Architecture
The Heart Disease module uses an **Ensemble Learning** approach combining two different deep learning architectures to improve prediction reliability.

*   **Artificial Neural Network (ANN)**:
    - A standard feed-forward architecture.
    - **Layers**: Three hidden layers with 64, 32, and 16 neurons.
    - **Regularization**: Uses `Dropout` layers (30% and 20%) to prevent overfitting.
    - **Output**: A single neuron with a `sigmoid` activation function for binary classification (Heart Disease / No Heart Disease).
*   **Convolutional Neural Network (CNN)**:
    - Utilizes `Conv1D` to treat the medical feature set as a sequence.
    - **Layers**: 1D Convolutional layer (32 filters), Max Pooling (size 2), and a Flattening layer followed by Dense layers.
    - **Purpose**: To capture spatial correlations between different medical indicators.

### 2. Implementation Workflow
- **Data Loading**: Loads the dataset from `data/heart.csv`.
- **Pre-processing**: Uses `StandardScaler` to normalize the 13 clinical features (age, sex, chest pain type, cholesterol, etc.) to a mean of 0 and variance of 1.
- **Ensemble Logic**: 
    - The `predict()` function runs the input through both the ANN and CNN models.
    - The final confidence score is the **average** of the probabilities produced by both models.
- **Reporting**: Generates performance visualizations saved as `heart_viz.png`.

---

## Module 2: Diabetes Prediction
**Source File:** [diabetes_model.py](file:///d:/code3/disease_prediction/models/diabetes_model.py)

### 1. Deep Neural Network (DNN) Architecture
The Diabetes module employs a specialized **Deep Neural Network** designed for tabular medical data.

*   **Structure**: A `Sequential` model built using TensorFlow/Keras.
*   **Hidden Layers**: 
    - Layer 1: 64 neurons (ReLU activation, 20% Dropout).
    - Layer 2: 32 neurons (ReLU activation).
    - Layer 3: 16 neurons (ReLU activation).
*   **Output Layer**: Sigmoid activation providing a probability between 0 and 1.
*   **Optimization**: Uses the `Adam` optimizer with `binary_crossentropy` as the loss function.

### 2. Implementation Workflow
- **Features**: Analyzes 8 key variables: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
- **Training**: The model is trained for 100 epochs, allowing it to converge on complex patterns in the glucose and insulin relationships.
- **Data Scaling**: Similar to the heart model, it uses `StandardScaler` to ensure that features like 'Glucose' (high range) don't dominate features like 'DiabetesPedigreeFunction' (low range).
- **Visualization**: Produces a training history plot (`diabetes_viz.png`) showing how the "Loss" decreased over time, ensuring the model learned effectively.
