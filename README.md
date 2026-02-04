<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Telecom Churn Prediction - Documentation</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">


<body>

<header>
    <h1>📊 Telecom Customer Churn Prediction</h1>
    <p>End-to-End Machine Learning System Documentation</p>
</header>

<section>

<div class="card">
<h2>1. Project Overview</h2>
<p>
This system predicts whether a telecom customer will churn using Machine Learning.
It includes a full automated pipeline from raw data to deployment.
</p>
</div>

<div class="card">
<h2>2. Business Problem</h2>
<p>
Customer churn reduces revenue. This system helps identify customers likely to leave so
retention strategies can be applied.
</p>
</div>

<div class="card">
<h2>3. Dataset Features</h2>
<table>
<tr><th>Category</th><th>Examples</th></tr>
<tr><td>Demographic</td><td>Gender, SeniorCitizen, Partner</td></tr>
<tr><td>Services</td><td>InternetService, StreamingTV</td></tr>
<tr><td>Billing</td><td>MonthlyCharges, TotalCharges</td></tr>
<tr><td>Contract</td><td>Contract, PaymentMethod</td></tr>
<tr><td>Network</td><td>SIM, Region, DeviceType</td></tr>
<tr><td>Target</td><td>Churn (Yes/No)</td></tr>
</table>
</div>

<div class="card">
<h2>4. ML Pipeline</h2>
<ol>
<li>Data Loading</li>
<li>Missing Value Handling</li>
<li>Outlier Treatment</li>
<li>Feature Selection</li>
<li>Categorical Encoding</li>
<li>Class Balancing</li>
<li>Feature Scaling</li>
<li>Model Training</li>
<li>Best Model Saving</li>
</ol>
</div>

<div class="card">
<h2>5. Models Used</h2>
<table>
<tr><th>Model</th><th>Description</th></tr>
<tr><td>KNN</td><td>Distance-based</td></tr>
<tr><td>Naive Bayes</td><td>Probabilistic</td></tr>
<tr><td>Logistic Regression</td><td>Linear</td></tr>
<tr><td>Decision Tree</td><td>Rule-based</td></tr>
<tr><td>Random Forest</td><td>Ensemble</td></tr>
<tr><td>AdaBoost</td><td>Boosting</td></tr>
<tr><td>Gradient Boosting</td><td>Sequential Boosting</td></tr>
<tr><td>XGBoost</td><td>Optimized Boosting</td></tr>
<tr><td>SVM</td><td>Support Vector Machine</td></tr>
</table>
</div>

<div class="card">
<h2>6. Saved Artifacts</h2>
<pre>
best_scaler.pkl
cat_encoder.pkl
feature_column.pkl
Churn_Prediction_Best_Model.pkl
</pre>
</div>

<div class="card">
<h2>7. Web Application</h2>
<p>
Users enter customer data, select SIM using logos, and receive churn prediction with probability.
</p>
</div>

<div class="card">
<h2>8. Project Structure</h2>
<pre>
telecom-churn-prediction/
│
├── app.py
├── Churn_Prediction_Best_Model.pkl
├── best_scaler.pkl
├── cat_encoder.pkl
├── feature_column.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── images/
        ├── airtel.png
        ├── jio.png
        ├── vi.png
        └── bsnl.png
</pre>
</div>

<div class="card">
<h2>9. SIM Providers</h2>
<img src="static/images/airtel.png" class="logo">
<img src="static/images/jio.png" class="logo">
<img src="static/images/vi.png" class="logo">
<img src="static/images/bsnl.png" class="logo">
</div>

<div class="card">
<h2>10. How to Run</h2>
<pre>
pip install -r requirements.txt
python app.py
</pre>
<p>Open: <b>http://127.0.0.1:5000</b></p>
</div>

<div class="card">
<h2>11. Sample Output</h2>
<pre>
Prediction: Churn
Probability: 86.75%
</pre>
</div>

</section>

<div class="footer">
© 2026 Telecom Churn Prediction | Developed by You
</div>

</body>
</html>
