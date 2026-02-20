import mysql.connector

import json
import os

DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'yourpassword')
DB_NAME = "disease_prediction"

def get_db_connection(database=None):
    """Establish a connection to the MySQL database."""
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=database
    )

def init_db():
    """Initialize the database and create tables if they exist."""

    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    c.close()
    conn.close()

    conn = get_db_connection(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            disease_type VARCHAR(255) NOT NULL,
            input_features TEXT NOT NULL,
            prediction_result VARCHAR(255) NOT NULL,
            probability FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(disease_type, features, result, probability):
    """Log a prediction to the database."""
    conn = get_db_connection(DB_NAME)
    c = conn.cursor()
    
    features_json = json.dumps(features)
    
    sql = '''
        INSERT INTO predictions (disease_type, input_features, prediction_result, probability)
        VALUES (%s, %s, %s, %s)
    '''
    val = (disease_type, features_json, result, probability)
    
    c.execute(sql, val)
    
    conn.commit()
    conn.close()

def get_history(disease_type=None):
    """Retrieve the last 50 predictions with optional disease type filtering."""
    conn = get_db_connection(DB_NAME)

    c = conn.cursor(dictionary=True)
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if disease_type and disease_type != 'All':
        query += " AND disease_type = %s"
        params.append(disease_type)
        
    query += " ORDER BY timestamp DESC LIMIT 50"
    
    c.execute(query, tuple(params))
    rows = c.fetchall()
    conn.close()
    return rows

def clear_history():
    """Clear all records from the predictions table."""
    conn = get_db_connection(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
