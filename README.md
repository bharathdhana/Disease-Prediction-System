# Multiple Symptom-Based Disease Prediction System

A Deep Learning based web application to predict probability of Heart Disease, Diabetes, and Parkinson's Disease.

## 📋 Prerequisites

1.  **Python 3.10** or higher.
2.  **MySQL Server** installed and running.

## 🚀 Installation & Setup

1.  **Clone/Download the repository** to your local machine.
2.  **Set up Virtual Environment:**
    Open a terminal in the project folder:
    ```bash
    python -m venv .venv
    ```
3.  **Activate Virtual Environment:**
    *   Windows: `.venv\Scripts\activate`
    *   Mac/Linux: `source .venv/bin/activate`
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run


### Option 1: One-Click Start (Windows)
Double-click the `run_app.bat` file.
*   This will open the Web App and the Database Monitor in separate windows.

### Option 2: Manual Start

1.  **Start the Web Application:**
    Open a terminal in the project folder and run:
    ```bash
    .venv\Scripts\python.exe app.py
    ```
    *   *Note: First run takes 2-5 minutes to train models.*
    *   Access the app at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

2.  **Start the Database Viewer (Optional):**
    Open a second terminal window and run:
    ```bash
    .venv\Scripts\python.exe view_db.py
    ```

## 🛠️ Troubleshooting
*   **Database Error:** Ensure MySQL service is running (`sc query MySQL80` or via Services.msc).
*   **Slow Startup:** The app retrains models on every launch. Please wait for "Running on http://127.0.0.1:5000" in the console.
