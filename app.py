# app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

# Import your pipeline functions
from main import full_credit_pipeline

app = FastAPI(title="MSME Credit Scoring API", version="1.0")

# --- Run pipeline once at server startup ---
API_KEY = os.getenv("OPENAI_API_KEY")
DF_MASTER, DF_DAILY = full_credit_pipeline(n_gstin=100, days=60)

# --- Request schema ---
class GSTINRequest(BaseModel):
    gstin: str

@app.post("/gstin_info")
async def get_gstin_info(request: GSTINRequest):
    gstin = request.gstin.upper().strip()
    
    # Filter master and daily data by GSTIN
    master_info = DF_MASTER[DF_MASTER['GSTIN'] == gstin]
    daily_info = DF_DAILY[DF_DAILY['GSTIN'] == gstin]
    
    if master_info.empty or daily_info.empty:
        return JSONResponse(
            status_code=404,
            content={"error": f"GSTIN {gstin} not found."}
        )

    # ✅ Convert all datetime columns to string
    for df in [master_info, daily_info]:
        for col in df.select_dtypes(include=['datetime64[ns]']).columns:
            df[col] = df[col].astype(str)

    # Return JSON
    return JSONResponse(content={
        "gstin": gstin,
        "gstin_summary": master_info.to_dict(orient="records"),
        "daily_snapshot": daily_info.to_dict(orient="records")
    })
# --- Optional: quick print to check DF_MASTER ---
if __name__ == "__main__":
    print(DF_MASTER.head())