# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =========================
# 2. Load Dataset
# =========================
data = pd.read_csv("data/creditcard.csv")
data = data.sample(20000)

print(data.head())
# Data Cleaning
# =========================

# Check Missing Values
print("Missing Values:")
print(data.isnull().sum())

# Remove Duplicates
data = data.drop_duplicates()

print("\nDuplicates Removed")

# Check Data Types
print("\nData Types:")
print(data.dtypes)

# Statistical Summary
print("\nStatistical Summary:")
print(data.describe())

print("Shape of dataset:", data.shape)
print(data.info())
print(data['Class'].value_counts())


# =========================
# 3. Data Analysis (Location & Machine)
# =========================
# إنشاء Location و Machine (Simulation)
data['Location'] = np.random.choice(['Cairo', 'Alex', 'Giza'], size=len(data))
data['Machine'] = np.random.choice(['ATM1', 'ATM2', 'ATM3'], size=len(data))

# تحليل Location
fraud_by_location = data.groupby('Location')['Class'].mean() * 100
print("\nFraud Percentage by Location:")
print(fraud_by_location)

# تحليل Machine
fraud_by_machine = data.groupby('Machine')['Class'].mean() * 100
print("\nFraud Percentage by Machine:")
print(fraud_by_machine)

#______________heatmap ____________________

plt.figure(figsize=(10,6))
numeric_data = data.select_dtypes(include=np.number)
sns.heatmap(numeric_data.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 4. Visualization
# =========================
# رسم Machine
plt.figure(figsize=(6,4))
fraud_by_machine.plot(kind='bar')
plt.title("Fraud Rate by Machine")
plt.xlabel("Machine")
plt.ylabel("Fraud Percentage")
plt.show()

# رسم توزيع العمليات
counts = data['Class'].value_counts()
plt.bar(['Normal', 'Fraud'], counts)
plt.title("Transaction Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.savefig("transaction_chart.png")
plt.show()


# =========================
# 5. Prepare Data for ML
# =========================
# حذف الأعمدة الإضافية
data = data.drop(['Location', 'Machine'], axis=1)

X = data.drop('Class', axis=1)
y = data['Class']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 6. Train Model using rondomforest
# =========================
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# =========scaling============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
# =========================

lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))

# =========================
# 7. Prediction & Evaluation
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# Model Comparison
# =========================

# Random Forest Report
print("Random Forest Report:")
print(classification_report(y_test, y_pred))

# Logistic Regression Report
print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))

# Accuracy Comparison
models = ['Random Forest', 'Logistic Regression']

scores = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, lr_pred)
]

plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

# Zoom to show small differences
plt.ylim(0.95, 1.0)
plt.show()

 #_________ROC______________

y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
print("AUC Score:", roc_auc_score(y_test, y_prob))

# =========================
# 8. Confusion Matrix Visualization
# =========================
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()


# =========================
# 9. Fraud Rate Analysis
# =========================
fraud_rate = (data['Class'].sum() / len(data)) * 100
print("\nFraud Percentage:", fraud_rate)

fraud_detected = cm[1][1]
total_fraud = cm[1][0] + cm[1][1]

fraud_detection_rate = (fraud_detected / total_fraud) * 100
print("Fraud Detection Rate:", fraud_detection_rate)


# =========================
# 10. Feature Importance
# =========================
importance = model.feature_importances_
features = X.columns

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

print("\nTop Important Features:")
print(feature_importance.sort_values(by='Importance', ascending=False).head(10))

#_________Feature Importance__________
top_features = feature_importance.sort_values( by='Importance', ascending=False).head(10)
plt.figure(figsize=(8,5))
plt.bar(top_features['Feature'], top_features['Importance'])
plt.xticks(rotation=45)
plt.title("Top Important Features")
plt.show()

# =========================
# 11. Test New Transaction
# =========================
sample = X_test.iloc[[0]]
prediction = model.predict(sample)
print("Fraud" if prediction[0] else "Normal")

##___savemodel__________

joblib.dump(model, "fraud_model.pkl")

##_______batch simulation _______
model = joblib.load("fraud_model.pkl")

def predict_fraud(input_data):
    df = pd.DataFrame([input_data])
    
    prediction = model.predict(df)

    if prediction[0] == 1:
        return "⚠ Fraud Transaction"
    else:
        return "✅ Normal Transaction"

for i in range(5):
    sample = X_test.iloc[i].to_dict()
    print(predict_fraud(sample))
