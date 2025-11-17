import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report,precision_score,recall_score,confusion_matrix,f1_score

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")
st.write("""
This app uses **Isolation Forest** to detect fraudulent transactions from the Credit Card dataset.
You can upload your own dataset, visualize patterns, and run anomaly detection.
""")


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    df.dropna(subset=['Class'],inplace=True)
    st.write(df.head())
    sc = StandardScaler()
    amount = df['Amount'].values
    df['Amount'] = sc.fit_transform(amount.reshape(-1,1))

    df = df.drop(['Time'], axis=1)
    X = df.drop('Class', axis=1)
    y = df['Class'].values

    st.sidebar.header("Select Parameters")
    contaminations = st.sidebar.slider("Contamination Fraud(%):",0.001,0.02,0.003,step=0.001)

    n_estimators = st.sidebar.selectbox("Number of Estimators:",[100,200,300,400])



    # Isolation Forest
    IF = IsolationForest(bootstrap=True,
                        n_estimators=n_estimators,
                        max_features=0.8,
                        max_samples=0.8,
                        random_state=42,
                        contamination=contaminations)

    predictions = IF.fit_predict(X)
    mapped_predictions = np.array([0 if p == 1 else 1 for p in predictions])

    precision = precision_score(y, mapped_predictions)
    recall = recall_score(y, mapped_predictions)
    f1 = f1_score(y, mapped_predictions)

    st.subheader("Model Performance")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, mapped_predictions)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)





    st.subheader("PCA Visualization")


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X) 

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        X_pca[mapped_predictions == 0, 0],
        X_pca[mapped_predictions == 0, 1],
        s=4,
        c="Blue",
        label="Normal",
        alpha=0.5
    )
    ax.scatter(
        X_pca[mapped_predictions == 1,0],
        X_pca[mapped_predictions == 1,1],
        s=4,
        c='Red',
        label='Fraud',
        alpha=0.5
    )
    st.pyplot(fig)

    st.subheader("Detected Fraudulent Transactions")

    frauds = df[mapped_predictions== 1]
    st.write(f"Total Fraudulent Transactions: {len(frauds)}")

    st.dataframe(frauds.head(50))

    csv = frauds.to_csv(index=False).encode()
    st.download_button("Download Fraud Transactions CSV", csv, "fraud_results.csv")

else:
    st.warning("Please upload a CSV file to continue.")


