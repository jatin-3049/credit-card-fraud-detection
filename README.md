# ⭐ **Credit Card Fraud Detection — Unsupervised Machine Learning (Isolation Forest)**

This project detects fraudulent credit card transactions using **unsupervised anomaly detection** with the **Isolation Forest** algorithm.  
It includes an interactive **Streamlit web app** where users can upload a dataset, visualize anomalies, and analyze performance.

---

## 🚀 **Project Features**

- ✔ **Upload any credit card transaction CSV**  
- ✔ **Automatic data cleaning & preprocessing**  
- ✔ **Adjustable contamination (fraud %) slider**  
- ✔ **PCA 2D scatter-plot visualization**  
- ✔ **Confusion Matrix heatmap**  
- ✔ **Precision, Recall, and F1-Score**  
- ✔ **Fully interactive Streamlit Dashboard**

---

## 🧠 **Model Used: Isolation Forest**

- Works great for **anomaly detection**  
- Detects fraud **without requiring labels**  
- Handles highly **imbalanced datasets** effectively  
- Fast and scalable for large datasets

---

## 📊 **How the App Works**

1. **Upload the dataset**  
   - Expected format: Kaggle’s **creditcard.csv**  
2. **Preprocessing**  
   - Scales **Amount**  
   - Removes **Time**  
3. **Prediction**  
   - Model predicts anomalies using **Isolation Forest**  
4. **Outputs shown in the app**  
   - **Confusion Matrix**  
   - **PCA visualization**  
   - **Precision, Recall, F1-Score**

---

## ▶️ **How to Run Locally**

### **1️⃣ Install dependencies**

pip install -r requirements.txt


### **2️⃣ Run the Streamlit App**
streamlit run app.py


Streamlit will open the dashboard in your browser.


**🧪 Dataset**

Publicly available on Kaggle:
Credit Card Fraud Detection Dataset
(284,807 transactions + 492 frauds)

**🎨 Visual Outputs**

PCA scatter plot of normal vs fraud transactions

Confusion Matrix

Performance metrics
