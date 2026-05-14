import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    layout="wide",
    page_icon="💳"
)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    border-radius: 20px;
    background-color: #161B22;
    border: 2px solid #00E5FF;
}

h1 {
    color: #00E5FF;
    text-align: center;
}

h2, h3 {
    color: white;
}

.stButton > button {
    background-color: #00E5FF;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================

model = joblib.load("fraud_model.pkl")

# =========================
# Convert Figure to Image
# =========================

def fig_to_img(fig):

    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        bbox_inches='tight'
    )

    buf.seek(0)

    return buf

# =========================
# Title
# =========================

st.title("💳 AI Fraud Detection Dashboard")

st.write(
    "Upload CSV file to analyze transactions and detect fraudulent activities in real time."
)

# =========================
# Upload CSV
# =========================

file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

# =========================
# MAIN LOGIC
# =========================

if file is not None:

    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    required_cols = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    # =========================
    # Check Columns
    # =========================

    if all(col in df.columns for col in required_cols):

        # =========================
        # Predict Button
        # =========================

        if st.button("🔍 Predict Transactions"):

            # =========================
            # Prediction
            # =========================

            with st.spinner("⏳ Analyzing transactions... Please wait..."):

                X = df[required_cols]

                predictions = model.predict(X)

                probabilities = model.predict_proba(X)[:, 1]

                df["Prediction"] = predictions

                df["Fraud Probability %"] = (
                    probabilities * 100
                ).round(2)

                df["Prediction"] = df["Prediction"].map({
                    0: "Normal",
                    1: "Fraud"
                })

            st.success(
                "✅ Prediction Completed Successfully"
            )

            # =========================
            # Metrics
            # =========================

            fraud_count = (
                df["Prediction"] == "Fraud"
            ).sum()

            normal_count = (
                df["Prediction"] == "Normal"
            ).sum()

            total_transactions = len(df)

            # =========================
            # KPI Cards
            # =========================

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "🚨 Fraud Transactions",
                fraud_count
            )

            col2.metric(
                "✅ Normal Transactions",
                normal_count
            )

            col3.metric(
                "📦 Total Transactions",
                total_transactions
            )

            st.divider()

            # =========================
            # Results Table
            # =========================

            st.subheader("📋 Prediction Results")

            st.dataframe(df)

            st.divider()

            # =========================
            # Top Suspicious
            # =========================

            st.subheader(
                "🚨 Top Suspicious Transactions"
            )

            top_fraud = df.sort_values(
                by="Fraud Probability %",
                ascending=False
            ).head(10)

            st.dataframe(top_fraud)

            st.divider()

            # =========================
            # PIE CHART
            # =========================

            st.subheader(
                "📈 Fraud vs Normal Distribution"
            )

            fig1, ax1 = plt.subplots(
                figsize=(4, 4)
            )

            ax1.pie(
                [fraud_count, normal_count],
                labels=["Fraud", "Normal"],
                autopct="%1.1f%%",
                colors=["red", "green"]
            )

            ax1.set_title("Fraud vs Normal")

            st.pyplot(fig1)

            st.info(f"""
📌 Analysis:
- Fraud Transactions: {fraud_count}
- Normal Transactions: {normal_count}
- Most transactions are {'Fraudulent' if fraud_count > normal_count else 'Normal'}.
""")

            st.divider()

            # =========================
            # HISTOGRAM
            # =========================

            st.subheader(
                "💰 Transaction Amount Distribution"
            )

            fig2, ax2 = plt.subplots(
                figsize=(7, 4)
            )

            ax2.hist(
                df["Amount"],
                bins=30,
                color="cyan"
            )

            ax2.set_xlabel("Amount")
            ax2.set_ylabel("Frequency")

            st.pyplot(fig2)

            st.info("""
📌 Amount Analysis:
- This chart shows the distribution of transaction amounts.
- Higher bars indicate frequently occurring values.
- Outliers may indicate suspicious activity.
""")

            st.divider()

            # =========================
            # SCATTER PLOT
            # =========================

            st.subheader(
                "📉 Relationship Between Time and Amount"
            )

            fig3, ax3 = plt.subplots(
                figsize=(7, 4)
            )

            colors = df["Prediction"].map({
                "Fraud": "red",
                "Normal": "green"
            })

            ax3.scatter(
                df["Time"],
                df["Amount"],
                c=colors,
                alpha=0.6
            )

            ax3.set_xlabel("Time")
            ax3.set_ylabel("Amount")

            st.pyplot(fig3)

            st.info("""
📌 Relationship Analysis:
- Red points represent Fraud transactions.
- Green points represent Normal transactions.
- Fraud activities may cluster in unusual regions.
""")

            st.divider()

            # =========================
            # HEATMAP
            # =========================

            st.subheader(
                "🔥 Feature Correlation Heatmap"
            )

            fig4, ax4 = plt.subplots(
                figsize=(8, 5)
            )

            corr = df[
                ["Amount", "Time", "V1", "V2", "V3", "V4", "V5"]
            ].corr()

            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                ax=ax4
            )

            st.pyplot(fig4)

            st.info("""
📌 Heatmap Analysis:
- Warm colors indicate positive correlation.
- Cool colors indicate negative correlation.
- Strong correlations may help identify fraud patterns.
""")

            st.divider()

            # =========================
            # FRAUD TABLE
            # =========================

            fraud_df = df[
                df["Prediction"] == "Fraud"
            ]

            st.subheader(
                "🚨 Fraud Transactions Only"
            )

            st.dataframe(fraud_df)

            st.divider()

            # =========================
            # CONFUSION MATRIX
            # =========================

            roc_auc = 0

            if "Class" in df.columns:

                st.subheader(
                    "🧠 Confusion Matrix"
                )

                y_true = df["Class"]

                y_pred = predictions

                cm = confusion_matrix(
                    y_true,
                    y_pred
                )

                fig5, ax5 = plt.subplots(
                    figsize=(5, 5)
                )

                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm
                )

                disp.plot(ax=ax5)

                st.pyplot(fig5)

                st.info("""
📌 Confusion Matrix Analysis:
- True Positives = Correct fraud detection.
- True Negatives = Correct normal detection.
- False predictions represent model errors.
""")

                st.divider()

                # =========================
                # ROC CURVE
                # =========================

                st.subheader(
                    "📈 ROC Curve"
                )

                fpr, tpr, thresholds = roc_curve(
                    y_true,
                    probabilities
                )

                roc_auc = auc(fpr, tpr)

                fig6, ax6 = plt.subplots(
                    figsize=(6, 4)
                )

                ax6.plot(
                    fpr,
                    tpr,
                    color="orange",
                    label=f"AUC = {roc_auc:.2f}"
                )

                ax6.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    color="white"
                )

                ax6.set_xlabel(
                    "False Positive Rate"
                )

                ax6.set_ylabel(
                    "True Positive Rate"
                )

                ax6.legend()

                st.pyplot(fig6)

                st.info(f"""
📌 ROC Analysis:
- AUC Score = {roc_auc:.2f}
- Higher AUC means better fraud detection performance.
- The closer the curve to the top-left corner, the stronger the model.
""")

            # =========================
            # PDF REPORT
            # =========================

            doc = SimpleDocTemplate(
                "fraud_report.pdf"
            )

            styles = getSampleStyleSheet()

            elements = []

            # =========================
            # Title
            # =========================

            elements.append(
                Paragraph(
                    "Fraud Detection Full Report",
                    styles['Title']
                )
            )

            elements.append(
                Spacer(1, 12)
            )

            # =========================
            # Summary
            # =========================

            summary = Paragraph(
                f"""
                <b>Total Transactions:</b> {total_transactions}<br/>
                <b>Fraud Transactions:</b> {fraud_count}<br/>
                <b>Normal Transactions:</b> {normal_count}<br/>
                <b>AUC Score:</b> {roc_auc:.2f}
                """,
                styles['BodyText']
            )

            elements.append(summary)

            elements.append(
                Spacer(1, 20)
            )

            # =========================
            # Add Charts to PDF
            # =========================

            charts = [
                ("Fraud vs Normal Distribution", fig1),
                ("Transaction Amount Distribution", fig2),
                ("Time vs Amount Relationship", fig3),
                ("Feature Correlation Heatmap", fig4)
            ]

            if "Class" in df.columns:

                charts.extend([
                    ("Confusion Matrix", fig5),
                    ("ROC Curve", fig6)
                ])

            for title, fig in charts:

                elements.append(
                    Paragraph(
                        title,
                        styles['Heading2']
                    )
                )

                elements.append(
                    Spacer(1, 10)
                )

                img = Image(
                    fig_to_img(fig),
                    width=450,
                    height=250
                )

                elements.append(img)

                elements.append(
                    Spacer(1, 20)
                )

            # =========================
            # Build PDF
            # =========================

            doc.build(elements)

            # =========================
            # Download Button
            # =========================

            with open(
                "fraud_report.pdf",
                "rb"
            ) as pdf_file:

                st.download_button(
                    label="📥 Download Full PDF Report",
                    data=pdf_file,
                    file_name="fraud_report.pdf",
                    mime="application/pdf"
                )

    else:

        st.error(
            "❌ CSV must contain Time, V1-V28, Amount"
        )
