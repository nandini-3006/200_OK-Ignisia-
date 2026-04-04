import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
import shap

# -----------------------------
# 1️⃣ Generate Synthetic Data
# -----------------------------
def generate_raw_data(n_gstin, days):
    gstins = [f"B{i}" for i in range(n_gstin)]
    dates = pd.date_range(start="2026-01-01", periods=days)
    types = np.random.choice(["high", "medium", "low"], size=n_gstin, p=[0.3, 0.5, 0.2])
    business_type = dict(zip(gstins, types))

    rows = []
    for g in gstins:
        t = business_type[g]
        if t == "high":
            base_inv, base_inflow, delay_range = np.random.randint(80,150), np.random.randint(5000,10000), (0,2)
        elif t == "medium":
            base_inv, base_inflow, delay_range = np.random.randint(30,80), np.random.randint(2000,5000), (1,4)
        else:
            base_inv, base_inflow, delay_range = np.random.randint(5,30), np.random.randint(500,2000), (3,8)

        for d in dates:
            invoices = int(base_inv * np.random.uniform(0.8,1.2))
            invoice_amount = invoices * np.random.randint(50,100)
            upi_inflow = int(base_inflow * np.random.uniform(0.8,1.2))
            upi_outflow = int(upi_inflow * np.random.uniform(0.6,0.9))
            txn_count = int(invoices * np.random.uniform(0.5,1.5))
            eway_count = int(invoices * np.random.uniform(0.1,0.3))
            filing_delay_days = np.random.randint(*delay_range)
            rows.append([g,d,invoices,invoice_amount,upi_inflow,upi_outflow,txn_count,eway_count,filing_delay_days])

    df = pd.DataFrame(rows, columns=[
        "GSTIN","Date","invoices_count","invoice_amount",
        "upi_inflow","upi_outflow","txn_count",
        "eway_count","filing_delay_days"
    ])
    return df

# -----------------------------
# 2️⃣ Data Cleaning
# -----------------------------
def clean_data(df):
    df = df.drop_duplicates(subset=['GSTIN','Date']).sort_values(['GSTIN','Date'])
    # Introduce some NaNs for realism
    df.loc[np.random.rand(len(df)) < 0.02, "upi_inflow"] = np.nan
    df.loc[np.random.rand(len(df)) < 0.02, "txn_count"] = np.nan

    df["upi_inflow"] = df.groupby("GSTIN")["upi_inflow"].ffill().bfill()
    df["upi_outflow"] = df.groupby("GSTIN")["upi_outflow"].ffill().bfill()

    for col in ['invoice_amount','txn_count','eway_count','filing_delay_days']:
        df[col] = df[col].fillna(df.groupby("GSTIN")[col].transform("median"))

    return df

# -----------------------------
# 3️⃣ Feature Engineering
# -----------------------------
def feature_engineering(df):
    df['rolling_avg_inflow'] = df.groupby('GSTIN')['upi_inflow'].rolling(7,min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_avg_outflow'] = df.groupby('GSTIN')['upi_outflow'].rolling(7,min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_txn_count'] = df.groupby('GSTIN')['txn_count'].rolling(7,min_periods=1).mean().reset_index(level=0,drop=True)

    df['inflow_outflow_ratio'] = df['upi_inflow'] / (df['upi_outflow'] + 1e-5)
    df['filing_velocity'] = 1 / (1 + df['filing_delay_days'])
    df['avg_delay'] = df.groupby('GSTIN')['filing_delay_days'].transform('mean')
    df['activity_score'] = df['invoices_count'] + df['txn_count'] + df['eway_count']
    df['txn_variance'] = df.groupby('GSTIN')['txn_count'].transform('var').fillna(0)
    df['shipping_growth'] = df.groupby('GSTIN')['eway_count'].transform(lambda x: x.pct_change().replace([np.inf,-np.inf],0).fillna(0))

    df.replace([np.inf,-np.inf],0,inplace=True)
    return df

# -----------------------------
# 4️⃣ Preprocessing & PCA
# -----------------------------
def preprocess_and_reduce(df):
    amount_cols = ['invoice_amount','upi_inflow','upi_outflow']
    count_cols = ['invoices_count','txn_count','eway_count','activity_score','rolling_txn_count','rolling_avg_inflow','rolling_avg_outflow']
    ratio_cols = ['inflow_outflow_ratio']
    velocity_cols = ['filing_velocity']
    avg_cols = ['avg_delay']
    growth_cols = ['shipping_growth']
    variance_cols = ['txn_variance']

    df.fillna(0, inplace=True)
    for col in amount_cols+growth_cols+variance_cols:
        df[col] = np.log1p(np.abs(df[col])) * np.sign(df[col])

    df[amount_cols+growth_cols+variance_cols] = StandardScaler().fit_transform(df[amount_cols+growth_cols+variance_cols])
    df[count_cols+avg_cols] = RobustScaler().fit_transform(df[count_cols+avg_cols])
    df[ratio_cols+velocity_cols] = StandardScaler().fit_transform(df[ratio_cols+velocity_cols])
    df[['filing_delay_days']] = MinMaxScaler().fit_transform(df[['filing_delay_days']])

    feature_cols = amount_cols + count_cols + ratio_cols + velocity_cols + avg_cols + growth_cols + variance_cols + ['filing_delay_days']

    pca = PCA(n_components=0.95, random_state=42)
    pca_values = pca.fit_transform(df[feature_cols])
    pca_cols = [f'PC{i+1}' for i in range(pca_values.shape[1])]
    df_pca = df[['GSTIN','Date']].copy()
    df_pca[pca_cols] = pca_values

    return df, df_pca

# -----------------------------
# 5️⃣ Clustering
# -----------------------------
def perform_clustering(df, df_pca):
    pca_cols = [c for c in df_pca.columns if c.startswith('PC')]
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_pca['daily_cluster'] = kmeans.fit_predict(df_pca[pca_cols])
    df = df.merge(df_pca[['GSTIN','Date','daily_cluster']], on=['GSTIN','Date'], how='left')

    gstin_cluster = df.groupby('GSTIN')['daily_cluster'].agg(lambda x:x.value_counts().idxmax()).reset_index().rename(columns={'daily_cluster':'GSTIN_cluster'})
    df = df.merge(gstin_cluster, on='GSTIN', how='left')
    return df

# -----------------------------
# 6️⃣ Base Credit Score
# -----------------------------
def assign_credit_score(df):
    cluster_quality = df.groupby('GSTIN_cluster')['activity_score'].mean()
    cluster_scores = cluster_quality.rank(method='dense').astype(int)
    df['credit_score'] = df['GSTIN_cluster'].map(cluster_scores.to_dict())
    return df

# -----------------------------
# 7️⃣ Train XGB Model + CV
# -----------------------------
def train_and_predict(df, feature_cols, n_splits=5):
    X = df[feature_cols]
    y = df['credit_score']
    groups = df['GSTIN']
    gkf = GroupKFold(n_splits=n_splits)

    fold_errors = []
    gstin_preds_list = []

    for train_idx, val_idx in gkf.split(X,y,groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor()
        model.fit(X_train, y_train)

        val_df = df.iloc[val_idx].copy()
        val_df['pred_score'] = model.predict(X_val)

        gstin_scores = val_df.groupby('GSTIN')['pred_score'].mean().reset_index()
        gstin_preds_list.append(gstin_scores)
        fold_errors.append(mean_squared_error(y_val, val_df['pred_score']))

    gstin_preds_df = pd.concat(gstin_preds_list).groupby('GSTIN')['pred_score'].mean().reset_index().rename(columns={'pred_score':'final_score'})
    print(f"Average CV MSE: {np.mean(fold_errors)}")
    return model, gstin_preds_df

# -----------------------------
# 8️⃣ Fraud Detection (Optional)
# -----------------------------
def detect_fraud_isolation(df, feature_cols, contamination=0.05):
    df_gstin = df.groupby('GSTIN')[feature_cols].mean().reset_index()
    iso = IsolationForest(contamination=contamination, random_state=42)
    df_gstin['fraud_flag'] = iso.fit_predict(df_gstin[feature_cols])
    return df_gstin[df_gstin['fraud_flag']==-1][['GSTIN','fraud_flag']]

# -----------------------------
# 9️⃣ Adjust Scores with SHAP
# -----------------------------
def adjust_score_with_top5_shap(df, model, feature_cols, max_adjustment=50, loan_multiplier=30, tenure_months=12):
    explainer = shap.Explainer(model)
    shap_values = explainer(df[feature_cols])
    shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
    shap_df['GSTIN'] = df['GSTIN'].values

    weighted_adjustments = []
    gstin_reasons = []
    for gstin, group in shap_df.groupby('GSTIN'):
        mean_abs_shap = group[feature_cols].abs().mean()
        top5 = mean_abs_shap.sort_values(ascending=False).head(5)
        weights = np.arange(len(top5),0,-1)
        weighted_mean = np.average(top5.values, weights=weights)
        weighted_adjustments.append({'GSTIN':gstin,'weighted_shap':weighted_mean})
        gstin_reasons.append({'GSTIN':gstin,'plain_reasons':', '.join(top5.index)})

    adj_df = pd.DataFrame(weighted_adjustments)
    reasons_df = pd.DataFrame(gstin_reasons)

    scaler = MinMaxScaler()
    adj_df['shap_scaled'] = scaler.fit_transform(adj_df[['weighted_shap']])

    df_base = df[['GSTIN','credit_score']].drop_duplicates()
    df_base['base_score'] = 300 + (df_base['credit_score']-1)*(600/4)
    df_final = df_base.merge(adj_df[['GSTIN','shap_scaled']],on='GSTIN',how='left')
    df_final['credit_score_final'] = df_final['base_score'] + (df_final['shap_scaled']-0.5)*max_adjustment
    df_final['credit_score_final'] = df_final['credit_score_final'].clip(300,900)
    df_final = df_final.merge(reasons_df,on='GSTIN',how='left')

    def risk_level(score):
        if score<450: return 'High Risk'
        elif score<550: return 'Moderate Risk'
        elif score<650: return 'Neutral'
        elif score<750: return 'Low Risk'
        else: return 'Very Low Risk'

    df_final['risk_band'] = df_final['credit_score_final'].apply(risk_level)
    df['net_cash'] = df['upi_inflow'] - df['upi_outflow']
    daily_cash = df.groupby('GSTIN')['net_cash'].mean().reset_index().rename(columns={'net_cash':'avg_daily_cash'})
    df_final = df_final.merge(daily_cash,on='GSTIN',how='left')

    df_final['recommended_loan'] = df_final['avg_daily_cash'] * loan_multiplier * (df_final['credit_score_final']/900) * tenure_months
    df_final['recommended_loan'] = df_final['recommended_loan'].clip(lower=0)
    df_final = df_final[['GSTIN','credit_score_final','plain_reasons','risk_band','avg_daily_cash','recommended_loan']]

    return df_final

# -----------------------------
#  🔟 Full Pipeline Function
# -----------------------------
def full_credit_pipeline(n_gstin=100, days=30):
    df_raw = generate_raw_data(n_gstin, days)
    df_clean = clean_data(df_raw)
    df_feat = feature_engineering(df_clean)
    df_preprocessed, df_pca = preprocess_and_reduce(df_feat)
    df_clustered = perform_clustering(df_feat, df_pca)
    df_scored = assign_credit_score(df_clustered)

    feature_cols = [c for c in df_preprocessed.columns if c not in ['GSTIN','Date','daily_cluster','GSTIN_cluster','credit_score']]
    model, gstin_preds = train_and_predict(df_scored, feature_cols)
    df_final = adjust_score_with_top5_shap(df_scored, model, feature_cols)

    return df_final, df_scored

# -----------------------------
# Example Usage
# -----------------------------
df_master, df_daily = full_credit_pipeline(n_gstin=5, days=15)
print(df_master.head())