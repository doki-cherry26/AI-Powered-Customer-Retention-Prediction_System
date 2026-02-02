# AI-Powered-Customer-Retention-Prediction_System
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Telecom Customer Churn Prediction</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f8f9fa;
            line-height: 1.6;
        }

        h1, h2, h3 {
            color: #2c3e50;
        }

        .section {
            background: #ffffff;
            padding: 25px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }

        pre {
            background: #272822;
            color: #f8f8f2;
            padding: 15px;
            overflow-x: auto;
            border-radius: 6px;
        }

        img {
            max-width: 100%;
            margin-top: 15px;
            border-radius: 6px;
        }

        .highlight {
            background: #eef6ff;
            padding: 10px;
            border-left: 5px solid #007bff;
            margin-top: 10px;
        }
    </style>
</head>

<body>

<h1>📊 Telecom Customer Churn Prediction</h1>
<p>
    This project builds a complete end-to-end Machine Learning pipeline
    to predict customer churn using advanced preprocessing, feature
    engineering, and multiple ML models.
</p>

<!-- ================= PIPELINE OVERVIEW ================= -->
<div class="section">
    <h2>🔄 Project Pipeline Overview</h2>
    <p>
        The following diagram represents the complete workflow of the churn
        prediction system.
    </p>

    <img src="images/pipeline_flow.png" alt="ML Pipeline Flow Diagram">
</div>

<!-- ================= DATA LOADING ================= -->
<div class="section">
    <h2>📂 Data Loading & Splitting</h2>
    <p>
        The dataset is loaded from CSV, the target variable <b>Churn</b>
        is encoded, and data is split into training and testing sets.
    </p>

    <pre>
class TELE_CUSTOMER:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.y = self.df['Churn'].map({'Yes':1,'No':0})
        self.X = self.df.drop(['Churn','customerID'], axis=1)
    </pre>
</div>

<!-- ================= MISSING VALUES ================= -->
<div class="section">
    <h2>🧩 Missing Value Handling</h2>
    <p>
        Multiple imputation techniques are evaluated using a
        RandomForest model, and the best one is selected based on F1-score.
    </p>

    <ul>
        <li>Mean Imputation</li>
        <li>Median Imputation</li>
        <li>Constant Value Imputation</li>
        <li>KNN Imputation</li>
    </ul>

    <img src="images/missing_values.png" alt="Missing Values Handling">
</div>

<!-- ================= TRANSFORMATION ================= -->
<div class="section">
    <h2>🔁 Variable Transformation & Scaling</h2>
    <p>
        Each numerical feature is tested with transformations to reduce
        skewness and improve model performance.
    </p>

    <ul>
        <li>Log Transformation</li>
        <li>Square Root Transformation</li>
        <li>Yeo-Johnson Transformation</li>
    </ul>

    <div class="highlight">
        ✔ Best transformation is selected <b>per column</b> using F1-score
    </div>
</div>

<!-- ================= OUTLIERS ================= -->
<div class="section">
    <h2>📦 Outlier Detection & Treatment</h2>
    <p>
        Several statistical techniques are compared to handle extreme values.
    </p>

    <ul>
        <li>IQR Capping</li>
        <li>Z-Score Capping</li>
        <li>Modified Z-Score</li>
        <li>Percentile Capping</li>
        <li>Isolation Forest</li>
    </ul>

    <h3>Before Outlier Handling</h3>
    <img src="images/outliers_before.png">

    <h3>After Outlier Handling</h3>
    <img src="images/outliers_after.png">
</div>

<!-- ================= FEATURE SELECTION ================= -->
<div class="section">
    <h2>🧠 Feature Selection</h2>
    <p>
        Constant and quasi-constant features are removed using
        Variance Threshold to reduce noise.
    </p>

    <pre>
VarianceThreshold(threshold=0.001)
    </pre>
</div>

<!-- ================= ENCODING ================= -->
<div class="section">
    <h2>🔤 Categorical Encoding</h2>
    <p>
        Categorical features are converted into numerical form using:
    </p>

    <ul>
        <li>Ordinal Encoding</li>
        <li>One-Hot Encoding</li>
        <li>Target Encoding</li>
        <li>Binary Encoding</li>
        <li>Hash Encoding</li>
    </ul>
</div>

<!-- ================= BALANCING ================= -->
<div class="section">
    <h2>⚖️ Data Balancing</h2>
    <p>
        Since churn data is imbalanced, SMOTE is applied to improve
        minority class learning.
    </p>

    <div class="highlight">
        ✔ SMOTE dynamically adjusts neighbors based on minority samples
    </div>
</div>

<!-- ================= MODELS ================= -->
<div class="section">
    <h2>🤖 Model Training</h2>
    <p>The following models are trained and evaluated:</p>

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
</div>

<!-- ================= ROC ================= -->
<div class="section">
    <h2>📈 Model Evaluation (ROC Curve)</h2>
    <p>
        ROC-AUC is used to select the best performing model.
    </p>

    <img src="images/roc_curve.png">
</div>

<!-- ================= FINAL ================= -->
<div class="section">
    <h2>🏆 Final Outcome</h2>
    <p>
        The model with the highest ROC-AUC score is saved and used for
        churn prediction in production.
    </p>

    <div class="highlight">
        ✔ Best model saved as <b>Churn_Prediction_Best_Model.pkl</b>
    </div>
</div>

</body>
</html>
