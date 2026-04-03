import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
from openai import OpenAI
import os

# --- setup ---
n_gstin = 50
days = 60

gstins = [f"B{i}" for i in range(n_gstin)]
dates = pd.date_range(start="2026-01-01", periods=days)

# assign business types
types = np.random.choice(["high", "medium", "low"], size=n_gstin, p=[0.3, 0.5, 0.2])
business_type = dict(zip(gstins, types))

rows = []

for g in gstins:
    t = business_type[g]
    
    # base values by type
    if t == "high":
        base_inv = np.random.randint(80, 150)
        base_inflow = np.random.randint(5000, 10000)
        delay_range = (0, 2)
    elif t == "medium":
        base_inv = np.random.randint(30, 80)
        base_inflow = np.random.randint(2000, 5000)
        delay_range = (1, 4)
    else:
        base_inv = np.random.randint(5, 30)
        base_inflow = np.random.randint(500, 2000)
        delay_range = (3, 8)

    for d in dates:
        
        # variation
        invoices = int(base_inv * np.random.uniform(0.8, 1.2))
        invoice_amount = invoices * np.random.randint(50, 100)

        upi_inflow = int(base_inflow * np.random.uniform(0.8, 1.2))
        upi_outflow = int(upi_inflow * np.random.uniform(0.6, 0.9))

        txn_count = int(invoices * np.random.uniform(0.5, 1.5))
        eway_count = int(invoices * np.random.uniform(0.1, 0.3))

        filing_delay_days = np.random.randint(*delay_range)

        rows.append([
            g, d, invoices, invoice_amount,
            upi_inflow, upi_outflow, txn_count,
            eway_count, filing_delay_days
        ])

# final dataframe
df = pd.DataFrame(rows, columns=[
    "GSTIN",
    "Date",
    "invoices_count",
    "invoice_amount",
    "upi_inflow",
    "upi_outflow",
    "txn_count",
    "eway_count",
    "filing_delay_days"
])

#Example:You buy stock (₹10,000 outflow),Sell later (₹5,000 inflow today)👉 So:,outflow > inflow ✅ (valid case)
df = df.drop_duplicates(subset=['GSTIN', 'Date'])

df = df.sort_values(["GSTIN", "Date"])

mask = np.random.rand(len(df)) < 0.02
df.loc[mask, "upi_inflow"] = np.nan

mask2 = np.random.rand(len(df)) < 0.02
df.loc[mask2, "txn_count"] = np.nan

df["upi_inflow"] = df.groupby("GSTIN")["upi_inflow"].ffill().bfill()
df["upi_outflow"] = df.groupby("GSTIN")["upi_outflow"].ffill().bfill()

cols = [
    "invoice_amount",
    "txn_count",
    "eway_count",
    "filing_delay_days"
]

for col in cols:
    df[col] = df[col].fillna(df.groupby("GSTIN")[col].transform("median"))

# --- FEATURE ENGINEERING ---

df['rolling_avg_inflow'] = (
    df.groupby('GSTIN')['upi_inflow']
      .rolling(window=7, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

df['rolling_avg_outflow'] = (
    df.groupby('GSTIN')['upi_outflow']
      .rolling(window=7, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

df['rolling_txn_count'] = (
    df.groupby('GSTIN')['txn_count']
      .rolling(window=7, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

df['inflow_outflow_ratio'] = df['upi_inflow'] / (df['upi_outflow'] + 1e-5)
df['filing_velocity'] = 1 / (1 + df['filing_delay_days'])

df['avg_delay'] = df.groupby('GSTIN')['filing_delay_days'].transform('mean')

df['net_cash'] = df['upi_inflow'] - df['upi_outflow']

df['rolling_cashflow_std'] = (
    df.groupby('GSTIN')['net_cash']
      .rolling(window=7, min_periods=1)
      .std()
      .reset_index(level=0, drop=True)
)

df['invoice_growth_rate'] = df.groupby('GSTIN')['invoice_amount'].pct_change().fillna(0)
df['inflow_growth_rate'] = df.groupby('GSTIN')['upi_inflow'].pct_change().fillna(0)
df['shipping_growth'] = df.groupby('GSTIN')['eway_count'].pct_change().fillna(0)

df['activity_score'] = df['invoices_count'] + df['txn_count'] + df['eway_count']

df['txn_variance'] = df.groupby('GSTIN')['txn_count'].transform('var').fillna(0)

df.drop(columns=['net_cash'], inplace=True)

df.replace([np.inf, -np.inf], 0, inplace=True)
df['shipping_growth'] = df.groupby('GSTIN')['eway_count'].transform(
    lambda x: x.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
)
df[col] = np.log1p(df[col].abs()) * np.sign(df[col])

minmax_scaler = MinMaxScaler()
df['filing_delay_days'] = minmax_scaler.fit_transform(df[['filing_delay_days']])

# --- GROUP FEATURES BY TYPE ---
# --- Correct feature groups ---
amount_cols = ['invoice_amount', 'upi_inflow', 'upi_outflow']
count_cols = ['invoices_count', 'txn_count', 'eway_count', 'activity_score',
              'rolling_txn_count', 'rolling_avg_inflow', 'rolling_avg_outflow']
ratio_cols = ['inflow_outflow_ratio']
velocity_cols = ['filing_velocity']
avg_cols = ['avg_delay']  # rolling or avg features that are not counts
growth_cols = ['invoice_growth_rate', 'inflow_growth_rate', 'shipping_growth']
variance_cols = ['txn_variance', 'rolling_cashflow_std']

# --- 1️⃣ LOG TRANSFORM AMOUNTS, GROWTH, VARIANCE (reduce skew) ---
# --- 1️⃣ LOG transform skewed numeric features ---
# --- handle inf / NaN globally ---
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# --- 1️⃣ LOG transform skewed numeric features ---
for col in amount_cols + growth_cols + variance_cols:
    df[col] = np.log1p(df[col].abs()) * np.sign(df[col])

log_std_cols = amount_cols + growth_cols + variance_cols
scaler = StandardScaler()
df[log_std_cols] = scaler.fit_transform(df[log_std_cols])


robust_cols = count_cols + avg_cols + ['filing_delay_days']
df[robust_cols] = RobustScaler().fit_transform(df[robust_cols])

# --- 3️⃣ StandardScaler for ratios & velocities ---
std_scaler = StandardScaler()
df[ratio_cols + velocity_cols] = std_scaler.fit_transform(df[ratio_cols + velocity_cols])

for col in variance_cols:
    df[col] = df[col] + np.random.normal(0, 1e-6, size=len(df))

# --- Check first row after scaling ---
row_idx = 0
print(df.loc[row_idx])



# Features for clustering
feature_cols = (
    amount_cols + count_cols + ratio_cols + velocity_cols +
    avg_cols + growth_cols + variance_cols
)
for col in feature_cols:
    if col not in df.columns:
        print(f"Missing column: {col}") 
X = df[feature_cols]

print(df[feature_cols].isnull().sum())  # NaNs
print((np.isinf(df[feature_cols])).sum()) 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Fit KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['daily_cluster'] = kmeans.fit_predict(X_pca) # daily cluster for each GSTIN-date point

# Compute overall GSTIN cluster (most frequent daily cluster)
gstin_cluster = df.groupby('GSTIN')['daily_cluster'] \
                  .agg(lambda x: x.value_counts().idxmax()) \
                  .reset_index()

gstin_cluster.rename(columns={'daily_cluster':'GSTIN_cluster'}, inplace=True)

# Merge back to df
df = df.merge(gstin_cluster, on='GSTIN', how='left')

# Now df has both daily and overall GSTIN clusters
print(df[['GSTIN', 'Date', 'daily_cluster', 'GSTIN_cluster']].head(10))
print(df[feature_cols].describe())  # check variance
print(df[feature_cols].head(10))    # see if values are actually diverse

# check unique daily cluster counts
print(df['daily_cluster'].value_counts())
daily_clusters = df.pivot(index='Date', columns='GSTIN', values='daily_cluster')
print(daily_clusters.head(10))
gstin_cluster = df.groupby('GSTIN')['daily_cluster'] \
                  .agg(lambda x: x.value_counts().idxmax()) \
                  .reset_index()
gstin_cluster.rename(columns={'daily_cluster':'GSTIN_cluster'}, inplace=True)

print(gstin_cluster)
print(df[df['GSTIN']=='B1'][feature_cols].describe())
print(df[df['GSTIN']=='B13'][feature_cols].describe())
print(df[df['GSTIN']=='B49'][feature_cols].describe())
print(df[feature_cols].var())


# activity-based cluster score
df['cluster_score'] = (
    df['invoice_amount'] +
    df['upi_inflow'] +
    df['txn_count'] +
    df['eway_count'] +
    df['filing_velocity']
)

# 1️⃣ Split GSTINs
gstins_train, gstins_val = train_test_split(df['GSTIN'].unique(), test_size=0.2, random_state=42)

gstins_train, gstins_val = train_test_split(df['GSTIN'].unique(), test_size=0.2, random_state=42)

X_train = df[df['GSTIN'].isin(gstins_train)][feature_cols]
y_train = df[df['GSTIN'].isin(gstins_train)]['cluster_score']

X_val = df[df['GSTIN'].isin(gstins_val)][feature_cols]
y_val = df[df['GSTIN'].isin(gstins_val)]['cluster_score']

reg = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

df_val = df[df['GSTIN'].isin(gstins_val)].copy()
df_val['predicted_cluster_score'] = y_pred


df_val['predicted_cluster_score'] = y_pred

gstin_credit = df_val.groupby('GSTIN')['predicted_cluster_score'].mean().reset_index()

y_min = gstin_credit['predicted_cluster_score'].min()
y_max = gstin_credit['predicted_cluster_score'].max()
gstin_credit['predicted_credit_score'] = 1 + (gstin_credit['predicted_cluster_score'] - y_min) * 99 / (y_max - y_min)
gstin_credit['predicted_credit_score'] = gstin_credit['predicted_credit_score'].round().astype(int)

print(gstin_credit[['GSTIN','predicted_credit_score']])

# 1️⃣ Create TreeExplainer for your trained XGB model
explainer = shap.Explainer(reg)

# 2️⃣ Compute SHAP values for the validation set
shap_values = explainer(X_val)

shap_df = pd.DataFrame(shap_values.values, columns=X_val.columns)
shap_df['GSTIN'] = df_val['GSTIN'].values
shap_df['predicted_score'] = df_val['predicted_cluster_score'].values

gstin_shap_summary = shap_df.groupby('GSTIN').apply(
    lambda x: pd.DataFrame({
        'mean_abs_shap': x[X_val.columns].abs().mean(),
        'mean_shap': x[X_val.columns].mean(),
    })
)

def compute_percent(df_gstin):
    total = df_gstin['mean_abs_shap'].sum()
    df_gstin['percent_contribution'] = (df_gstin['mean_abs_shap'] / total * 100).round(2)
    return df_gstin.sort_values('percent_contribution', ascending=False)

# Apply to all GSTINs
final_summary = gstin_shap_summary.groupby('GSTIN', group_keys=False).apply(compute_percent)

top_n = 4
top_features_per_gstin = final_summary.groupby('GSTIN', group_keys=False).head(top_n).reset_index()
print(top_features_per_gstin)


api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set!")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

def explain_shap_with_llm_top(gstin, shap_summary, predicted_score, top_n=5):
    """
    Generate LLM explanation of top N SHAP features for a single GSTIN
    """
    top_features = shap_summary.head(top_n)
    
    shap_text = "\n".join([
        f"{feat}: mean_abs_shap={row.mean_abs_shap:.4f}, mean_shap={row.mean_shap:.4f}, percent={row.percent_contribution:.2f}%"
        for feat, row in top_features.iterrows()
    ])
    
    prompt = f"""
You are a financial AI assistant. Given the predicted credit score {predicted_score} 
for GSTIN {gstin} and the following top {top_n} SHAP feature contributions:

{shap_text}

Please generate a concise explanation of:
- Which features most influence the credit score
- Which features are positive/negative contributors
- Overall interpretation in plain English
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

llm_explanations = []

for gstin in top_features_per_gstin['GSTIN'].unique():
    top_feats = top_features_per_gstin[top_features_per_gstin['GSTIN']==gstin]
    predicted_score = gstin_credit[gstin_credit['GSTIN']==gstin]['predicted_credit_score'].values[0]
    
    explanation = explain_shap_with_llm_top(
        gstin=gstin,
        shap_summary=top_feats,
        predicted_score=predicted_score,
        top_n=5
    )
    
    llm_explanations.append({
        'GSTIN': gstin,
        'predicted_credit_score': predicted_score,
        'llm_explanation': explanation
    })

llm_df = pd.DataFrame(llm_explanations)
print(llm_df.head(1))  

for idx, row in llm_df.iterrows():
    print(f"GSTIN: {row['GSTIN']}")
    print(f"Predicted Credit Score: {row['predicted_credit_score']}")
    print("Explanation:")
    print(row['llm_explanation'])  # This will render newlines properly
    print("-" * 80)  # separator# print one GSTIN nicely