# AI-Powered-Customer-Retention-Prediction_System
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Telecom Customer Churn Prediction System</title>

section{
    padding:30px;
    max-width:1000px;
    margin:auto;
}

.card{
    background:#020617;
    border:1px solid #334155;
    border-radius:12px;
    padding:20px;
    margin-bottom:25px;
    box-shadow:0 0 20px rgba(0,0,0,0.6);
}

h2{color:#38bdf8;border-bottom:1px solid #334155;padding-bottom:8px;}
ul{line-height:1.8;}
code{
    background:#1e293b;
    padding:4px 8px;
    border-radius:6px;
    color:#22c55e;
}

.grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:20px;
}

.sim-card{
    text-align:center;
    background:#020617;
    border:1px solid #334155;
    border-radius:10px;
    padding:15px;
}
.sim-card img{width:80px;}

footer{
    text-align:center;
    padding:20px;
    background:#020617;
    border-top:1px solid #334155;
    color:#94a3b8;
}
</style>
</head>

<body>

<header>
    <h1>📡 Telecom Customer Churn Prediction System</h1>
    <p>End-to-End Machine Learning Pipeline with Auto Optimization</p>
</header>

<section>

<div class="card">
<h2>📘 Project Overview</h2>
<p>
This project predicts whether a telecom customer will churn using machine learning.  
It automatically selects the best preprocessing methods, encoders, scalers, and classification models.
</p>
</div>

<div class="card">
<h2>⚙️ System Pipeline</h2>
<ul>
<li>Data Loading & Cleaning</li>
<li>Best Missing Value Imputer (Constant, Mean, Median, KNN)</li>
<li>Outlier Detection & Handling (Clip vs Remove)</li>
<li>Feature Selection (Variance, Chi-Square, Pearson)</li>
<li>Best Categorical Encoding (Ordinal, Target, Binary, Hashing)</li>
<li>SMOTE Data Balancing</li>
<li>Automatic Feature Scaling</li>
<li>Model Training & Selection</li>
</ul>
</div>

<div class="card">
<h2>🤖 Machine Learning Models</h2>
<div class="grid">
<div>KNN</div>
<div>Naive Bayes</div>
<div>Logistic Regression</div>
<div>Decision Tree</div>
<div>Random Forest</div>
<div>AdaBoost</div>
<div>Gradient Boosting</div>
<div>XGBoost</div>
<div>SVM</div>
</div>
</div>

<div class="card">
<h2>📂 Project Files</h2>
<ul>
<li><code>main.py</code> → Pipeline Controller</li>
<li><code>missing_values.py</code> → Best Imputer</li>
<li><code>variable_outliers.py</code> → Outlier Handling</li>
<li><code>feature_selection.py</code> → Feature Selection</li>
<li><code>cat_to_num.py</code> → Encoding</li>
<li><code>data_balancing.py</code> → SMOTE</li>
<li><code>all_models.py</code> → Model Training</li>
<li><code>Churn_Prediction_Best_Model.pkl</code> → Best Model</li>
</ul>
</div>

<div class="card">
<h2>🖥 Web Interface</h2>
<p>Users enter customer data and select SIM cards using logos.</p>
<div class="grid">
<div class="sim-card">
<img src="static/images/airtel.png"><p>Airtel</p>
</div>
<div class="sim-card">
<img src="static/images/jio.png"><p>Jio</p>
</div>
<div class="sim-card">
<img src="static/images/vi.png"><p>Vodafone</p>
</div>
<div class="sim-card">
<img src="static/images/bsnl.png"><p>BSNL</p>
</div>
</div>
</div>

<div class="card">
<h2>📊 Outputs</h2>
<ul>
<li><code>Churn_Prediction_Best_Model.pkl</code></li>
<li><code>best_scaler.pkl</code></li>
<li><code>cat_encoder.pkl</code></li>
<li><code>feature_column.pkl</code></li>
</ul>
</div>

</section>

<footer>
<p>© Telecom Churn Prediction | Machine Learning Project</p>
</footer>

</body>
</html>
