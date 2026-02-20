    # Testing Inputs for Disease Prediction Models

Use the following inputs to test the accuracy of the disease prediction models. These examples are taken directly from the training datasets.

## 1. Heart Disease Prediction

| Case Type | Age | Sex | CP | Trestbps | Chol | FBS | RestECG | Thalach | Exang | Oldpeak | Slope | CA | Thal | **Expected Result** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | 63 | 1 | 3 | 145 | 233 | 1 | 0 | 150 | 0 | 2.3 | 0 | 0 | 1 | **Heart Disease** |
| **Positive** | 37 | 1 | 2 | 130 | 250 | 0 | 1 | 187 | 0 | 3.5 | 0 | 0 | 2 | **Heart Disease** |
| **Positive** | 41 | 0 | 1 | 130 | 204 | 0 | 0 | 172 | 0 | 1.4 | 2 | 0 | 2 | **Heart Disease** |
| **Positive** | 56 | 1 | 1 | 120 | 236 | 0 | 1 | 178 | 0 | 0.8 | 2 | 0 | 2 | **Heart Disease** |
| **Positive** | 57 | 0 | 0 | 120 | 354 | 0 | 1 | 163 | 1 | 0.6 | 2 | 0 | 2 | **Heart Disease** |
| **Negative** | 67 | 1 | 0 | 160 | 286 | 0 | 0 | 108 | 1 | 1.5 | 1 | 3 | 2 | **No Heart Disease** |
| **Negative** | 67 | 1 | 0 | 120 | 229 | 0 | 0 | 129 | 1 | 2.6 | 1 | 2 | 3 | **No Heart Disease** |
| **Negative** | 62 | 0 | 0 | 140 | 268 | 0 | 0 | 160 | 0 | 3.6 | 0 | 2 | 2 | **No Heart Disease** |
| **Negative** | 63 | 1 | 0 | 130 | 254 | 0 | 0 | 147 | 0 | 1.4 | 1 | 1 | 3 | **No Heart Disease** |
| **Negative** | 53 | 1 | 0 | 140 | 203 | 1 | 0 | 155 | 1 | 3.1 | 0 | 0 | 3 | **No Heart Disease** |

---

## 2. Diabetes Prediction

| Case Type | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DPF | Age | **Expected Result** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | 6 | 148 | 72 | 35 | 0 | 33.6 | 0.627 | 50 | **Diabetes Detected** |
| **Positive** | 8 | 183 | 64 | 0 | 0 | 23.3 | 0.672 | 32 | **Diabetes Detected** |
| **Positive** | 0 | 137 | 40 | 35 | 168 | 43.1 | 2.288 | 33 | **Diabetes Detected** |
| **Positive** | 3 | 78 | 50 | 32 | 88 | 31.0 | 0.248 | 26 | **Diabetes Detected** |
| **Positive** | 2 | 197 | 70 | 45 | 543 | 30.5 | 0.158 | 53 | **Diabetes Detected** |
| **Negative** | 1 | 85 | 66 | 29 | 0 | 26.6 | 0.351 | 31 | **No Diabetes Detected** |
| **Negative** | 1 | 89 | 66 | 23 | 94 | 28.1 | 0.167 | 21 | **No Diabetes Detected** |
| **Negative** | 5 | 116 | 74 | 0 | 0 | 25.6 | 0.201 | 30 | **No Diabetes Detected** |
| **Negative** | 10 | 115 | 0 | 0 | 0 | 35.3 | 0.134 | 29 | **No Diabetes Detected** |
| **Negative** | 4 | 110 | 92 | 0 | 0 | 37.6 | 0.191 | 30 | **No Diabetes Detected** |

---

## 3. Parkinson's Prediction

| Case Type | MDVP:Fo(Hz) | MDVP:Fhi(Hz) | MDVP:Flo(Hz) | MDVP:Jitter(%) | MDVP:Shimmer | **Expected Result** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | 119.992 | 157.302 | 74.997 | 0.00784 | 0.04374 | **Parkinson's Detected** |
| **Positive** | 122.400 | 148.650 | 113.819 | 0.00968 | 0.06134 | **Parkinson's Detected** |
| **Positive** | 116.682 | 131.111 | 111.555 | 0.01050 | 0.05233 | **Parkinson's Detected** |
| **Positive** | 116.676 | 137.871 | 111.366 | 0.00997 | 0.05492 | **Parkinson's Detected** |
| **Positive** | 116.014 | 141.781 | 110.655 | 0.01284 | 0.06425 | **Parkinson's Detected** |
| **Negative** | 197.076 | 206.896 | 192.055 | 0.00289 | 0.01098 | **No Parkinson's Detected** |
| **Negative** | 199.228 | 209.512 | 192.091 | 0.00241 | 0.01015 | **No Parkinson's Detected** |
| **Negative** | 198.383 | 215.203 | 193.104 | 0.00212 | 0.01263 | **No Parkinson's Detected** |
| **Negative** | 202.266 | 211.604 | 197.079 | 0.00180 | 0.00954 | **No Parkinson's Detected** |
| **Negative** | 203.184 | 211.526 | 196.160 | 0.00178 | 0.00958 | **No Parkinson's Detected** |
