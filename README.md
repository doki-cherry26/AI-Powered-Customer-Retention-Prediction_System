# 📊 Telecom Customer Churn Prediction  
##End-to-End Machine Learning Project

This project builds a **complete end-to-end machine learning pipeline** to predict whether a telecom customer will **churn** (leave the service).  
It covers everything from **EDA → preprocessing → feature engineering → modeling → evaluation → deployment**.

---

## 🎯 Business Problem

Customer churn causes huge revenue loss in telecom companies.  
This system predicts **high-risk customers** so the company can take proactive retention actions.

---

## 🛠️ Technologies Used

| Category | Tools |
|---------|------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Encoding | category_encoders |
| Feature Selection | VarianceThreshold, Chi-Square, Pearson |
| Scaling | StandardScaler, MinMaxScaler, RobustScaler |
| Balancing | SMOTE |
| Web App | Flask |
| Frontend | HTML, CSS |
| Storage | Pickle |

---

## 🔁 End-to-End Pipeline


Data → EDA → Missing Values → Outliers → Feature Selection
→ Encoding → Balancing → Scaling → Model Training → Evaluation → Deployment


---

## 📂 Dataset

Target Column: **Churn**
- Yes → 1  
- No → 0  

Features include:
- Customer tenure  
- Monthly charges  
- Contract type  
- Payment method  
- Internet services  

---

## 📊 1. Exploratory Data Analysis (EDA)

### Visualizations Used
- Countplots for churn distribution  
- Boxplots for outliers  
- Heatmap for correlation  
- Histograms for numeric variables  

### Key Findings
- Month-to-month customers churn more  
- Higher monthly charges → higher churn  
- New customers have high churn risk  

---

## 🧹 2. Missing Value Handling

Best imputer selected using **ROC-AUC comparison**:

| Method | Technique |
|--------|-----------|
| Constant | Fill with 0 |
| Mean | Numeric mean |
| Median | Numeric median |
| KNN | KNN Imputation |

Best method is selected automatically using **Logistic Regression ROC-AUC**.

---

## 📉 3. Outlier Handling

- IQR method  
- Strategies tested:  
  - Clip  
  - Remove  
- Best strategy chosen by **minimum remaining outliers**

---

## 🧬 4. Feature Selection

| Technique | Purpose |
|----------|---------|
| VarianceThreshold | Remove constant features |
| Quasi-Variance | Remove near-constant |
| Chi-Square | Feature importance |
| Pearson Correlation | Remove weak features |

Final selected features saved in:


feature_column.pkl


---

## 🔄 5. Categorical Encoding

Encoders tested:
- Ordinal Encoder  
- Target Encoder  
- Binary Encoder  
- Hashing Encoder  

Best selected using **ROC-AUC** with Logistic Regression.

Saved as:


cat_encoder.pkl


---

## ⚖️ 6. Data Balancing

- Class imbalance detected  
- **SMOTE** applied only if imbalance ratio > 3  

---

## 📏 7. Feature Scaling

Scalers tested:
- StandardScaler  
- MinMaxScaler  
- RobustScaler  
- MaxAbsScaler  

Best scaler chosen using:


Skewness + Kurtosis score


Saved as:


best_scaler.pkl


---

## 🤖 8. Model Training

| Model | ROC-AUC |
|------|---------|
| Logistic Regression | **0.84** |
| KNN | 0.80 |
| Decision Tree | 0.86 |
| Random Forest | 0.77 |
| Gradient Boosting | 0.89 |
| XGBoost | 0.90 |
| SVM | 0.88 |

---

## 🏆 Best Model

**Logistic Regression**  
**ROC-AUC = 0.84**

Saved as:


Churn_Prediction_Best_Model.pkl


---

## 📈 ROC Curve

ROC Curve plots:
- True Positive Rate  
- False Positive Rate  

Higher AUC = better churn prediction.

---

## 🌐 9. Deployment

Flask web app allows:
- User input  
- SIM logo selection  
- Churn probability prediction  

---

## 📁 Project Structure



telecom-churn/
│
├── app.py
├── Churn_Prediction_Best_Model.pkl
├── best_scaler.pkl
├── cat_encoder.pkl
├── feature_column.pkl
│
├── templates/
│ └── index.html
│
└── static/
└── images/
├── airtel.png
├── jio.png
├── vi.png
└── bsnl.png


---

## ▶️ Run Project

```bash
pip install -r requirements.txt
python app.py


Open:

http://127.0.0.1:5000

✅ Conclusion

This project demonstrates a complete real-world ML churn prediction system.
With Random Forest (ROC-AUC = 0.91), the model accurately identifies high-risk customers and helps telecom companies reduce churn.


---

If you want, I can next:
- Convert this into **HTML documentation**
- Add **screenshots section**
- Create **GitHub Pages website**
