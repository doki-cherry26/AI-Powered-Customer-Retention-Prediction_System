# AI-Powered-Customer-Retention-Prediction_System
 A complete end-to-end Machine Learning pipeline to predict whether a telecom customer will churn (leave the service) based on demographic, service usage, and billing information.
The project includes automated preprocessing, feature engineering, model comparison, and a web interface with SIM provider logos.

<header>
    <h1>📊 Telecom Customer Churn Prediction System</h1>
    <p>End-to-End Machine Learning Project with Web Interface</p>
</header>

<section>

<div class="card">
<h2>🚀 Project Overview</h2>
<p>
This project predicts whether a telecom customer will churn using advanced Machine Learning techniques.
It includes automated preprocessing, feature engineering, model comparison, and a web interface with SIM provider logos.
</p>
</div>

<div class="card">
<h2>✨ Key Features</h2>
<ul>
<li>Automated Missing Value Handling</li>
<li>Outlier Detection & Transformation</li>
<li>Feature Selection (Variance + Statistical)</li>
<li>Best Categorical Encoder Auto-Selection</li>
<li>Class Balancing using SMOTE</li>
<li>Best Feature Scaling Selection</li>
<li>9 Machine Learning Models</li>
<li>Web UI with SIM Logos</li>
</ul>
</div>

<div class="card">
<h2>🧠 Models Used</h2>
<table>
<tr><th>Model</th><th>Description</th></tr>
<tr><td>KNN</td><td>Distance-based classifier</td></tr>
<tr><td>Naive Bayes</td><td>Probabilistic classifier</td></tr>
<tr><td>Logistic Regression</td><td>Linear classifier</td></tr>
<tr><td>Decision Tree</td><td>Rule-based model</td></tr>
<tr><td>Random Forest</td><td>Ensemble trees</td></tr>
<tr><td>AdaBoost</td><td>Boosting model</td></tr>
<tr><td>Gradient Boosting</td><td>Sequential boosting</td></tr>
<tr><td>XGBoost</td><td>Optimized boosting</td></tr>
<tr><td>SVM</td><td>Support Vector Machine</td></tr>
</table>
</div>

<div class="card">
<h2>🔁 Machine Learning Pipeline</h2>
<ol>
<li>Data Loading</li>
<li>Missing Value Imputation</li>
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
<h2>📂 Project Structure</h2>
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
<h2>📱 SIM Providers</h2>
<img sc="static/images/airtel.png" class="logo">
<img sc="static/images/jio.png" class="logo">
<img sc="static/images/vi.png" class="logo">
<img sc="static/images/bsnl.png" class="logo">
🖥️ Web Application
Users can:
Enter customer details
Select SIM provider using logos
Click Predict Churn
Receive churn probability instantly
</div>

<div class="card">
<h2>⚙️ How to Run</h2>
<pre>
pip install -r requirements.txt
python app.py
</pre>
<p>Open browser: <b>http://127.0.0.1:5000</b></p>
</div>

<div class="card">
<h2>📈 Output Example</h2>
<pre>
Prediction: Churn
Probability: 87.32%
</pre>
</div>

</section>

<div class="footer">
© 2026 Telecom Churn Prediction | Developed by You
</div>

</body>
</html>
