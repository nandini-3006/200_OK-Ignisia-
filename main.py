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


#Generate Synthetic Data 

def generate_raw_data(n_gstin,days):
    import pandas as pd
    import numpy as np

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

    return df

def clean_data(df):
    import numpy as np

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

    return df

# --- FEATURE ENGINEERING ---
def feature_engineering(df):
    import numpy as np

    # --- Rolling Features ---
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

    return df


def preprocess_and_reduce(df):
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.decomposition import PCA

    # --- Feature Groups ---
    amount_cols = ['invoice_amount', 'upi_inflow', 'upi_outflow']
    count_cols = ['invoices_count', 'txn_count', 'eway_count', 'activity_score',
                  'rolling_txn_count', 'rolling_avg_inflow', 'rolling_avg_outflow']
    ratio_cols = ['inflow_outflow_ratio']
    velocity_cols = ['filing_velocity']
    avg_cols = ['avg_delay']
    growth_cols = ['invoice_growth_rate', 'inflow_growth_rate', 'shipping_growth']
    variance_cols = ['txn_variance', 'rolling_cashflow_std']

    # --- Clean ---
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # --- Log Transform ---
    for col in amount_cols + growth_cols + variance_cols:
        df[col] = np.log1p(np.abs(df[col])) * np.sign(df[col])

    # --- Scaling ---
    df[amount_cols + growth_cols + variance_cols] = StandardScaler().fit_transform(
        df[amount_cols + growth_cols + variance_cols]
    )

    df[count_cols + avg_cols] = RobustScaler().fit_transform(
        df[count_cols + avg_cols]
    )

    df[ratio_cols + velocity_cols] = StandardScaler().fit_transform(
        df[ratio_cols + velocity_cols]
    )

    df[['filing_delay_days']] = MinMaxScaler().fit_transform(df[['filing_delay_days']])

    # --- Feature List ---
    feature_cols = (
        amount_cols + count_cols + ratio_cols + velocity_cols +
        avg_cols + growth_cols + variance_cols + ['filing_delay_days']
    )

    # --- PCA ---
    pca = PCA(n_components=0.95, random_state=42)
    pca_values = pca.fit_transform(df[feature_cols])

    # --- Create PCA DataFrame ---
    pca_cols = [f'PC{i+1}' for i in range(pca_values.shape[1])]
    df_pca = df[['GSTIN', 'Date']].copy()
    df_pca[pca_cols] = pca_values

    return df, df_pca

def perform_clustering(df, df_pca):
    from sklearn.cluster import KMeans

    # --- Get PCA columns ---
    pca_cols = [col for col in df_pca.columns if col.startswith('PC')]

    # --- Fit KMeans ---
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_pca['daily_cluster'] = kmeans.fit_predict(df_pca[pca_cols])

    # --- Merge clusters back to original df ---
    df = df.merge(
        df_pca[['GSTIN', 'Date', 'daily_cluster']],
        on=['GSTIN', 'Date'],
        how='left'
    )

    # --- GSTIN-level cluster (mode) ---
    gstin_cluster = (
        df.groupby('GSTIN')['daily_cluster']
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={'daily_cluster': 'GSTIN_cluster'})
    )

    # --- Merge GSTIN cluster ---
    df = df.merge(gstin_cluster, on='GSTIN', how='left')

    return df

def assign_credit_score(df):

    # average per cluster
    cluster_quality = df.groupby('GSTIN_cluster')['activity_score'].mean()

    # DIRECT rank → 1 to 5
    cluster_scores = cluster_quality.rank(method='dense').astype(int)

    # map back
    df['credit_score'] = df['GSTIN_cluster'].map(cluster_scores.to_dict())

    return df







def train_and_predict(df, feature_cols, n_splits=5):
    from xgboost import XGBRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd

    X = df[feature_cols]
    y = df['credit_score']
    groups = df['GSTIN']  # ensure GSTINs don't appear in both train and val

    gkf = GroupKFold(n_splits=n_splits)
    fold_errors = []
    gstin_preds_list = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor()
        model.fit(X_train, y_train)

        val_df = df.iloc[val_idx].copy()
        val_df['pred_score'] = model.predict(X_val)

        # GSTIN-level average
        gstin_scores = val_df.groupby('GSTIN')['pred_score'].mean().reset_index()
        gstin_preds_list.append(gstin_scores)

        # Fold error
        mse = mean_squared_error(y_val, val_df['pred_score'])
        fold_errors.append(mse)
        print(f"Fold {fold} MSE: {mse}")

    print(f"Average MSE across {n_splits} folds: {np.mean(fold_errors)}")

    # Combine all GSTIN-level predictions (for reporting)
    gstin_preds_df = pd.concat(gstin_preds_list).groupby('GSTIN')['pred_score'].mean().reset_index()
    gstin_preds_df.rename(columns={'pred_score': 'final_score'}, inplace=True)

    # Return last trained model (or you can return all models if needed) and GSTIN-level predictions
    return model, gstin_preds_df


from sklearn.ensemble import IsolationForest

def detect_fraud_isolation(df, feature_cols, contamination=0.05, random_state=42):
    """
    Detects fraudulent GSTINs using Isolation Forest.
    
    Returns a dataframe of only GSTINs flagged as fraud.
    """
    # Aggregate daily features per GSTIN
    df_gstin = df.groupby('GSTIN')[feature_cols].mean().reset_index()
    
    # Fit Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    df_gstin['fraud_flag'] = iso.fit_predict(df_gstin[feature_cols])
    
    # Keep only fraud GSTINs
    fraud_df = df_gstin[df_gstin['fraud_flag'] == -1].copy()
    
    return fraud_df[['GSTIN', 'fraud_flag']]

fraud_features = [
    'invoice_amount', 'upi_inflow', 'upi_outflow',
    'txn_count', 'eway_count', 'activity_score',
    'rolling_txn_count', 'rolling_avg_inflow', 'rolling_avg_outflow',
    'inflow_outflow_ratio', 'filing_velocity', 'avg_delay',
    'invoice_growth_rate', 'inflow_growth_rate', 'shipping_growth',
    'txn_variance', 'rolling_cashflow_std'
]



def generate_shap_and_llm(df, model, feature_cols, api_key, gstin_input):
    import shap
    import pandas as pd
    import numpy as np
    from openai import OpenAI

    # --- 1. Prepare data ---
    X = df[feature_cols]

    # --- 2. SHAP Explainer ---
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # --- 3. SHAP DataFrame ---
    shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
    shap_df['GSTIN'] = df['GSTIN'].values
    shap_df['predicted_score'] = model.predict(X)

    # --- 4. Aggregate per GSTIN ---
    gstin_shap_summary = shap_df.groupby('GSTIN').apply(
        lambda x: pd.DataFrame({
            'mean_abs_shap': x[feature_cols].abs().mean(),
            'mean_shap': x[feature_cols].mean()
        })
    )

    # --- 5. % Contribution ---
    def compute_percent(df_gstin):
        total = df_gstin['mean_abs_shap'].sum()
        df_gstin['percent_contribution'] = (
            df_gstin['mean_abs_shap'] / total * 100
        ).round(2)
        return df_gstin.sort_values('percent_contribution', ascending=False)

    final_summary = gstin_shap_summary.groupby('GSTIN', group_keys=False).apply(compute_percent)

    # --- 6. Top Features ---
    top_n = 4
    top_features_per_gstin = final_summary.groupby('GSTIN', group_keys=False).head(top_n).reset_index()

    # --- 7. GSTIN-level predicted score (mean of daily) ---
    gstin_scores = shap_df.groupby('GSTIN')['predicted_score'].mean().reset_index()
    gstin_scores.rename(columns={'predicted_score': 'final_score'}, inplace=True)

    # --- 8. LLM Setup ---
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def explain(gstin, shap_data, score):
        text = "\n".join([
            f"{row['level_1']}: contribution={row['percent_contribution']}%, impact={row['mean_shap']:.3f}"
            for _, row in shap_data.iterrows()
        ])

        prompt = f"""
You are a financial AI assistant.

GSTIN: {gstin}
Predicted Credit Score: {round(score,2)}

Top contributing features:
{text}

Explain:
- Key drivers of score
- Positive vs negative signals
- Simple business insight
-also a different explanation of loan recommendation 
"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        return res.choices[0].message.content

    # --- 9. Generate LLM Output ---
    results = []

    # if a specific GSTIN is given, only generate for that GSTIN
    gstins_to_process = [gstin_input] if gstin_input else top_features_per_gstin['GSTIN'].unique()

    for gstin in gstins_to_process:
        shap_data = top_features_per_gstin[top_features_per_gstin['GSTIN'] == gstin]
        score = gstin_scores[gstin_scores['GSTIN'] == gstin]['final_score'].values[0]

        explanation = explain(gstin, shap_data, score)

        results.append({
            "GSTIN": gstin,
            "score": round(score, 2),
            "explanation": explanation
        })

    llm_df = pd.DataFrame(results)

    return top_features_per_gstin, llm_df

df_raw = generate_raw_data(n_gstin=10, days=30)  # 10 GSTINs, 30 days
print(df_raw.head())
df_clean = clean_data(df_raw)
print(df_clean.isna().sum())
df_feat = feature_engineering(df_clean)
print(df_feat.head())
df_preprocessed, df_pca = preprocess_and_reduce(df_feat)
print(df_pca.head())
df_clustered = perform_clustering(df_feat, df_pca)
print(df_clustered[['GSTIN', 'Date', 'daily_cluster', 'GSTIN_cluster']].head())
df_scored = assign_credit_score(df_clustered)
print(df_scored[['GSTIN', 'Date', 'GSTIN_cluster', 'credit_score']].head())

# --- Feature columns for model ---
feature_cols = [col for col in df_preprocessed.columns if col not in ['GSTIN', 'Date', 'daily_cluster', 'GSTIN_cluster', 'credit_score']]

# --- Train model & predict ---
model, gstin_scores = train_and_predict(df_scored, feature_cols)
print(gstin_scores)

# --- SHAP + LLM explanations ---
# You can set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")  # or replace with your key directly

top_features, llm_output = generate_shap_and_llm(df_scored, model, feature_cols, api_key,'B0')

print("Top SHAP Features per GSTIN:")
print(top_features.head())

print("\nLLM Explanations:")
print(llm_output.head())

fraud_gstins = detect_fraud_isolation(df_scored, fraud_features, contamination=0.1)
print(fraud_gstins)



