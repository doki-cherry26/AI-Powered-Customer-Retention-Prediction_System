# 📊 AI-Powered-Customer Churn Prediction  
## End-to-End Machine Learning Project

This project is a complete **end-to-end Machine Learning system** that predicts whether a telecom customer is likely to churn (leave the service).  
It includes **EDA, feature engineering, preprocessing, model training, evaluation, and deployment** using a Flask web application with a modern user interface.

---

## 📁 Project Workflow

1. Data Collection  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Feature Selection  
5. Categorical Encoding  
6. Data Balancing (SMOTE)  
7. Feature Scaling  
8. Model Training & Evaluation  
9. Hyperparameter Tuning  
10. Best Model Selection  
11. Flask Web Deployment  

---

## 1. Exploratory Data Analysis (EDA)

### Tools Used
- **Matplotlib** – Bar, Pie Charts  
- **Seaborn** – Histogram, Count, Bar, Box Plots  
- **Pandas** – Data manipulation  
- **NumPy** – Numerical operations  

---

### Key Insights

| Metric | Observation |
|--------|-------------|
| Churn Rate | 26.5% |
| Retention Rate | 73.5% |
| Highest Churn | New customers (0–14 months) |
| Lowest Churn | Long-term customers (58–72 months) |
| Top Internet Service | Fiber Optic |
| Best Loyalty Group | High Total Charges |

---

## 2. Feature Engineering

### Handling Missing Values
| Method | Result |
|--------|--------|
| Mean Imputation | Reduced variance |
| Median Imputation | Robust to outliers |
| **KNN Imputation (Best)** | Preserved relationships |

---

## 3. Variable Transformation

| Technique | Purpose |
|-----------|---------|
| Log Transform | Reduce skewness |
| IQR Method | Detect outliers |
| Box-Cox | Normalize data |
| **Clipping (Best)** | Handle extreme values |

---

## 4. Feature Selection

### Filter Methods
1. Constant Feature Removal  
2. Quasi-Constant Removal  
3. Chi-Square Test  
4. **Pearson Correlation (Best)**  

---

## 5. Categorical Encoding
- Ordinal Encoding  
- Target Encoding  
- Binary Encoding  
- Hashing Encoding  

---

## 6. Data Balancing

Class imbalance handled using **SMOTE**:

\[
x_{new} = x_i + (x_{zi} - x_i) \times \delta
\]

---

## 7. Feature Scaling

| Scaler | Description |
|--------|-------------|
| **StandardScaler (Used)** | Mean = 0, Std = 1 |
| MinMaxScaler | 0 to 1 |
| RobustScaler | Uses median |
| MaxAbsScaler | Sparse data |

---

## 8. Model Training

Algorithms Tested:
- Logistic Regression
- KNN
- Naive Bayes
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- SVM

Evaluation Metric:
- **ROC Curve**
- **AUC Score**

---

## 9. Best Model

**Logistic Regression**
- Accuracy: ~75%
- Best ROC-AUC Score

The model is saved using **Pickle** and deployed using Flask.

---

## 10. Deployment

A Flask web app allows users to:
- Enter customer details
- Select SIM card using logos
- Get churn prediction in real time

---

## ✅ Conclusion

This project demonstrates a complete machine learning pipeline for telecom churn prediction.  
Logistic Regression achieved the best ROC-AUC and is deployed in a user-friendly Flask web application for real-time business decision support.

---

## 🚀 Future Enhancements
- Deep Learning models (ANN, LSTM, CNN)  
- Real-time streaming data  
- Interactive dashboards  
- Cloud deployment  
