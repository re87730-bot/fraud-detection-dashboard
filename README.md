# fraud-detection-dashboard
# 💳 AI Fraud Detection Dashboard

# An AI-powered Fraud Detection System built using Machine Learning and Streamlit to analyze financial transactions, detect suspicious activities, visualize fraud patterns, and generate professional PDF reports.

---

# Project Features

✅ Fraud Detection using Machine Learning  
✅ Interactive Streamlit Dashboard  
✅ Upload CSV Transactions File  
✅ Fraud Probability Prediction  
✅ Fraud vs Normal Visualization  
✅ Transaction Amount Analysis  
✅ Correlation Heatmap  
✅ Confusion Matrix & ROC Curve  
✅ Fraud Analysis by Country  
✅ Fraud Analysis by ATM Machine  
✅ Live Fraud Location Map  
✅ Download Full PDF Report  

---

# Machine Learning Models

✅ Random Forest Classifier

---

# 📊 Dataset

Dataset used:

- Credit Card Fraud Detection Dataset

Features include:

- Time
- Amount
- V1 → V28
- Class (Fraud / Normal)

---

# Technologies Used

## Programming Language
- Python
  
## Libraries
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- ReportLab
- Joblib

---

# 📈 Dashboard Visualizations

The dashboard includes:

- Pie Chart
- Histogram
- Scatter Plot
- Heatmap
- Confusion Matrix
- ROC Curve
- Fraud by Country Chart
- Fraud by ATM Chart
- Live Fraud Map

---

# 📄 PDF Report

The system automatically generates a professional PDF report containing:

- Fraud Statistics
- AUC Score
- Visual Charts
- Fraud Analysis

---

# Fraud Analysis Features

The project simulates:

- Countries
- Cities
- ATM Machines
- Fraud Locations

This helps visualize fraud activities geographically.

---

# 📂 Project Structure

```bash
Fraud_Detection_Project/
│
├── app_streamlit.py
├── fraud_model.pkl
├── requirements.txt
├── README.md
├── data/
│   └── creditcard.csv
│
└── images/


## Run the Project
1/ Clone Repository
git clone YOUR_GITHUB_LINK

2/ Install Requirements
pip install -r requirements.txt

3/ Run Streamlit App
streamlit run app_streamlit.py

## Deployment

The project is deployed using:
# Streamlit Community Cloud
