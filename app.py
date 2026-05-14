from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# تحميل الموديل
model = joblib.load("fraud_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():

    # استقبال البيانات
    data = request.get_json()

    # تحويلها DataFrame
    df = pd.DataFrame([data])

    # التوقع
    prediction = model.predict(df)

    # النتيجة
    result = "Fraud" if prediction[0] == 1 else "Normal"

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)