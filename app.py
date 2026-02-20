from flask import Flask, render_template, request, make_response
import numpy as np
import database
from models.heart_model import HeartDiseaseModel
from models.diabetes_model import DiabetesModel
from models.parkinsons_model import ParkinsonsModel
import csv
import io
from fpdf import FPDF



app = Flask(__name__)

heart_model = HeartDiseaseModel()
diabetes_model = DiabetesModel()
parkinsons_model = ParkinsonsModel()

database.init_db()

@app.route('/')
def home():
    recent_history = database.get_history()[:5]
    return render_template('index.html', recent_history=recent_history)

@app.route('/history')
def history():
    disease_type = request.args.get('disease_type', '')
    rows = database.get_history(disease_type)
    return render_template('history.html', 
                           history=rows,
                           disease_type=disease_type)

@app.route('/download_history')
def download_history():
    file_format = request.args.get('format', 'csv')
    disease_type = request.args.get('disease_type', '')
    data = database.get_history(disease_type)
    
    if file_format == 'csv':
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['ID', 'Disease Type', 'Input Features', 'Prediction Result', 'Probability', 'Timestamp'])
        for row in data:
            cw.writerow([row['id'], row['disease_type'], row['input_features'], row['prediction_result'], f"{row['probability']}%", row['timestamp']])
        
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=history.csv"
        output.headers["Content-type"] = "text/csv"
        return output
        
    elif file_format == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        
        
        pdf.cell(200, 10, txt=f"Prediction History - {disease_type if disease_type else 'All'}", ln=1, align='C')

        for row in data:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 5, f"{row['timestamp']} - {row['disease_type']}: {row['prediction_result']} ({row['probability']}%)", ln=1)
            pdf.set_font("Arial", size=8)
            pdf.multi_cell(0, 5, f"Features: {row['input_features']}")

        
        response = make_response(pdf.output(dest='S').encode('latin-1'))
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=history.pdf'
        return response
    
    return "Invalid format", 400

@app.route('/predict/heart', methods=['GET', 'POST'])
def predict_heart():
    if request.method == 'POST':
        try:

            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            prediction, probability = heart_model.predict(features)
            metrics = heart_model.get_metrics()
            
            result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
            confidence = float(round(np.max(probability)*100, 2))
            
            database.log_prediction("Heart Disease", features, result_text, confidence)
            
            return render_template('result.html', 
                                   disease_type="Heart Disease",
                                   prediction=result_text, 
                                   probability=confidence,
                                   metrics=metrics)
        except Exception as e:
            return render_template('result.html', error=str(e))
    return render_template('predict_heart.html')

@app.route('/predict/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['pregnancies']),
                float(request.form['glucose']),
                float(request.form['bloodpressure']),
                float(request.form['skinthickness']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['dpf']),
                float(request.form['age'])
            ]
            prediction, probability = diabetes_model.predict(features)
            metrics = diabetes_model.get_metrics()
            
            result_text = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"
            confidence = float(round(np.max(probability)*100, 2))

            database.log_prediction("Diabetes", features, result_text, confidence)
            
            return render_template('result.html', 
                                   disease_type="Diabetes",
                                   prediction=result_text, 
                                   probability=confidence,
                                   metrics=metrics)
        except Exception as e:
            return render_template('result.html', error=str(e))
    return render_template('predict_diabetes.html')

@app.route('/predict/parkinsons', methods=['GET', 'POST'])
def predict_parkinsons():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['mdvp_fo']),
                float(request.form['mdvp_fhi']),
                float(request.form['mdvp_flo']),
                float(request.form['mdvp_jitter']),
                float(request.form['mdvp_shimmer'])
            ]
            prediction, probability = parkinsons_model.predict(features)
            metrics = parkinsons_model.get_metrics()
            
            result_text = "Parkinson's Detected" if prediction == 1 else "No Parkinson's Detected"
            confidence = float(round(np.max(probability)*100, 2))

            database.log_prediction("Parkinson's", features, result_text, confidence)
            
            return render_template('result.html', 
                                   disease_type="Parkinson's Disease",
                                   prediction=result_text, 
                                   probability=confidence,
                                   metrics=metrics)
        except Exception as e:
            return render_template('result.html', error=str(e))
    return render_template('predict_parkinsons.html')



if __name__ == '__main__':
    app.run(debug=True)
