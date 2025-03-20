from fastapi import FastAPI
import pandas as pd
import uvicorn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = FastAPI()

def read_data():
    try:
        df = pd.read_csv("aviator_data.csv")
        return df.tail(10).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/live-data")
def get_live_data():
    return {"data": read_data()}

@app.get("/api/trend-analysis")
def trend_analysis():
    try:
        df = pd.read_csv("aviator_data.csv")
        df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors='coerce')
        if df.empty:
            return {"error": "No data available"}
        avg_multiplier = df["Multiplier"].rolling(window=10).mean().tolist()
        trend = "Increasing" if avg_multiplier[-1] > avg_multiplier[-2] else "Decreasing"
        return {"average_multiplier": avg_multiplier[-1], "trend": trend}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/predict/{platform}")
def predict_outcome(platform: str):
    try:
        df = pd.read_csv(f"{platform}_aviator_data.csv")
        df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors='coerce')
        if df.empty:
            return {"error": "No data available for this platform"}
        df = df.dropna()
        df["Index"] = range(len(df))
        X = df["Index"].values.reshape(-1, 1)
        y = df["Multiplier"].values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y.ravel())
        next_index = scaler.transform(np.array([[X[-1][0] + 1]]))
        prediction = model.predict(next_index)[0]
        return {"platform": platform, "predicted_multiplier": round(prediction, 2)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
