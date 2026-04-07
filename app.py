# app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from fastapi.staticfiles import StaticFiles
from main import full_credit_pipeline
from fastapi.responses import HTMLResponse
import pathlib


app = FastAPI(title="MSME Credit Scoring API", version="1.0")

origins = [
    "http://localhost:5000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Import pipeline

# ✅ CORS (VERY IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Run pipeline once ---
DF_MASTER, DF_DAILY = full_credit_pipeline(n_gstin=100, days=60)

# -----------------------------
# Helpers
# -----------------------------
def safe_df(df):
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    return df


# -----------------------------
# 1️⃣ SUMMARY API
# -----------------------------
HERE = pathlib.Path(__file__).parent  # folder where your app.py is


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = HERE / "dashboard.html"
    print(f"[DEBUG] Resolved HTML path: {html_path}")
    print(f"[DEBUG] Exists: {html_path.exists()}")
    print(f"[DEBUG] Size: {html_path.stat().st_size if html_path.exists() else 'N/A'}")
    
    if not html_path.exists() or html_path.stat().st_size == 0:
        return HTMLResponse(content="File missing or empty", status_code=404)
    
    content = html_path.read_text(encoding="utf-8")
    print(f"[DEBUG] Serving HTML, length: {len(content)}")
    return HTMLResponse(content=content)


@app.get("/api/summary")
async def get_summary(n_gstin: int = 100, days: int = 30):
    df = DF_MASTER.copy()

    return {
        "total_gstins": len(df),
        "avg_credit_score": round(float(df["credit_score_final"].mean()), 1),
        "avg_loan_recommended": float(df["recommended_loan"].mean()),
        "fraud_flagged": 0,
        "risk_distribution": df["risk_band"].value_counts().to_dict()
    }


# -----------------------------
# 2️⃣ GSTIN LIST
# -----------------------------
@app.get("/api/gstins")
async def get_gstins(n_gstin: int = 100, days: int = 30):
    df = safe_df(DF_MASTER)
    return df.to_dict(orient="records")


# -----------------------------
# 3️⃣ LEADERBOARD
# -----------------------------
@app.get("/api/leaderboard")
async def get_leaderboard(top: int = 50, n_gstin: int = 100, days: int = 30):
    df = DF_MASTER.copy()

    df = df.sort_values("credit_score_final", ascending=False).head(top)
    df["rank"] = range(1, len(df) + 1)

    df = df.rename(columns={"credit_score_final": "score"})

    return safe_df(df).to_dict(orient="records")


# -----------------------------
# 4️⃣ GSTIN DETAIL
# -----------------------------
@app.get("/api/gstin/{gstin}")
async def get_gstin_detail(gstin: str, n_gstin: int = 100, days: int = 30):
    g = DF_MASTER[DF_MASTER["GSTIN"] == gstin]

    if g.empty:
        return JSONResponse(status_code=404, content={"error": "GSTIN not found"})

    return safe_df(g).iloc[0].to_dict()


# -----------------------------
# 5️⃣ DAILY CHARTS (CRITICAL)
# -----------------------------
@app.get("/api/gstin/{gstin}/daily_charts")
async def get_daily_charts(gstin: str, n_gstin: int = 100, days: int = 30):
    df = DF_DAILY[DF_DAILY["GSTIN"] == gstin]

    if df.empty:
        return JSONResponse(status_code=404, content={"error": "GSTIN not found"})

    df = safe_df(df)

    return {
        "dates": df["Date"].tolist(),
        "upi_inflow": df["upi_inflow"].tolist(),
        "upi_outflow": df["upi_outflow"].tolist(),
        "txn_count": df["txn_count"].tolist(),
        "eway_count": df["eway_count"].tolist(),
        "invoices_count": df["invoices_count"].tolist(),
        "activity_scores": df["activity_score"].tolist(),
        "daily_credit_proxy": df["credit_score"].tolist(),

        # optional (frontend expects it)
        "shap_features": []
    }


# -----------------------------
# 6️⃣ EXISTING POST API (KEEP)
# -----------------------------
class GSTINRequest(BaseModel):
    gstin: str

@app.post("/gstin_info")
async def get_gstin_info(request: GSTINRequest):
    gstin = request.gstin.upper().strip()

    master_info = DF_MASTER[DF_MASTER['GSTIN'] == gstin]
    daily_info = DF_DAILY[DF_DAILY['GSTIN'] == gstin]

    if master_info.empty or daily_info.empty:
        return JSONResponse(
            status_code=404,
            content={"error": f"GSTIN {gstin} not found."}
        )

    master_info = safe_df(master_info)
    daily_info = safe_df(daily_info)

    return {
        "gstin": gstin,
        "gstin_summary": master_info.to_dict(orient="records"),
        "daily_snapshot": daily_info.to_dict(orient="records")
    }


# -----------------------------
# DEBUG
# -----------------------------
if __name__ == "__main__":
    print(DF_MASTER.head())