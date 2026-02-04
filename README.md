# AI-Powered-Customer-Retention-Prediction_System

---

# 🌐 PROJECT DOCUMENTATION PAGE (HTML)

Save as: `project.html`

```html
<!DOCTYPE html>
<html>
<head>
<title>Telecom Churn Prediction System</title>

</head>

<body>

<h1>📡 Telecom Customer Churn Prediction</h1>

<section>
<h2>Project Overview</h2>
<p>This system predicts whether a telecom customer will churn using machine learning.
It automatically selects the best preprocessing, encoding, scaling, and model.</p>
</section>

<section>
<h2>Pipeline</h2>
<ul>
<li>Missing Value Detection & Best Imputer</li>
<li>Outlier Detection & Best Strategy</li>
<li>Feature Selection (Variance, Chi-Square, Pearson)</li>
<li>Best Categorical Encoder</li>
<li>SMOTE Data Balancing</li>
<li>Best Feature Scaling</li>
<li>Train 9 ML Models</li>
<li>Select Best via ROC-AUC</li>
</ul>
</section>

<section>
<h2>Models Used</h2>
<ul>
<li>KNN</li>
<li>Naive Bayes</li>
<li>Logistic Regression</li>
<li>Decision Tree</li>
<li>Random Forest</li>
<li>AdaBoost</li>
<li>Gradient Boosting</li>
<li>XGBoost</li>
<li>SVM</li>
</ul>
</section>

<section>
<h2>Outputs</h2>
<ul>
<li><code>Churn_Prediction_Best_Model.pkl</code></li>
<li><code>best_scaler.pkl</code></li>
<li><code>cat_encoder.pkl</code></li>
<li><code>feature_column.pkl</code></li>
</ul>
</section>

<section>
<h2>Web UI</h2>
<p>Users can enter customer details and select SIM logos.  
The system predicts churn and probability.</p>
</section>

</body>
</html>
