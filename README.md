💳 MSME Credit Scoring & Loan Recommendation Pipeline

A synthetic-data-driven AI/ML pipeline to estimate credit scores and recommend loan amounts for MSMEs using alternative financial and transactional signals.

This system integrates data engineering, machine learning, explainable AI (SHAP), and LLM-based humanization, along with an API and dashboard for real-time insights.

📌 Overview

Traditional credit scoring systems fail for MSMEs due to lack of formal credit history.

This project solves that by:

Leveraging GST, transaction, and behavioral data
Applying ML + clustering + explainability
Generating interpretable, business-friendly insights
🏗️ Pipeline Architecture
🟢 Data Generation
Generates synthetic MSME data:
GSTIN, invoices
Cash inflows/outflows
Transactions & delays
Business types:
High / Medium / Low activity profiles
🟢 Data Cleaning
Remove duplicates
Sort by GSTIN & Date
Handle missing values:
Forward fill / Backward fill
Median per GSTIN
🟢 Feature Engineering
Rolling averages:
Inflow, Outflow, Transactions
Financial ratios:
Inflow/Outflow ratio
Behavioral metrics:
Filing velocity
Activity score
Variance & growth features
💡 Transformation Logic
df[col] = np.log1p(np.abs(df[col])) * np.sign(df[col])
Reduces skewness
Preserves sign of values
Stabilizes ML performance
🟢 Normalization & Scaling
StandardScaler:
Amount, growth, variance, ratio, velocity columns
RobustScaler:
Count & average columns (outlier resistant)
MinMaxScaler:
Filing delay
🔻 Dimensionality Reduction (PCA)
Reduces correlated features
Retains 95% variance
Improves clustering + model efficiency
🧩 Clustering (KMeans)
Groups MSMEs into 5 behavioral clusters
Assigns cluster per day
Aggregates to GSTIN-level cluster
Converts unsupervised patterns → supervised learning signal
🚀 Supervised Learning
Model: XGBoost
Cross-validation:
GroupKFold (GSTIN-wise)
Prevents data leakage
Predicts credit score
🔍 Explainability (SHAP)
Identifies top contributing features
Measures feature impact on predictions
Adjusts base credit score
🧠 LLM Humanization
Converts SHAP + features → natural language

Example:

“GSTIN B1 shows high cash inflow and low filing delay, indicating stable operations and low risk.”

Makes system usable for non-technical stakeholders
⚖️ Risk & Loan Recommendation
Credit Score Mapping
Range: 300 – 900
Risk Bands
High Risk
Moderate Risk
Neutral
Low Risk
Very Low Risk
Loan Recommendation Formula
recommended_loan = avg_daily_cash * loan_multiplier * (credit_score_final / 900) * tenure_months
🛡️ Fraud Detection (Optional)
Model: Isolation Forest
Detects anomalous MSMEs
Flags unusual transaction behavior
🌐 API Layer (FastAPI)

Endpoint:

/gstin_info

Returns:

Credit score
Risk band
Top contributing features
SHAP-based insights
LLM explanation
Daily activity snapshot
🖥️ Dashboard
Built using HTML + JavaScript

Features:

GSTIN-level insights
Credit score visualization
Risk categorization
Daily trends
🧠 Tech Stack
Backend: FastAPI
ML: XGBoost, Scikit-learn
Explainability: SHAP
Data Processing: Pandas, NumPy
Frontend: HTML, CSS, JavaScript
Visualization: Chart.js
📊 Key Highlights

✔ Real-time credit scoring
✔ Alternative data-based risk analysis
✔ Behavioral clustering (KMeans)
✔ Explainable AI (SHAP)
✔ LLM-powered business insights
✔ Fraud detection capability
✔ API + Dashboard integration

🚀 Use Cases
MSME loan underwriting
FinTech credit risk engines
NBFC decision support systems
Financial inclusion platforms
📈 Future Scope
Integration with real GST / banking APIs
Real-time streaming pipelines
Advanced time-series forecasting
Multi-region behavioral modeling
🤝 Contributing

Contributions are welcome!
Fork the repo, raise issues, or submit PRs.
