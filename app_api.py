from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Загрузка модели и скейлера
with open('PINModel1.2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Счётчик запросов
request_count = 0

# Входные данные
class PredictionInput(BaseModel):
    is_male: int
    have_car: int
    CNT_MEMBERS: int

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    try:
        # Создание DataFrame
        input_df = pd.DataFrame([{
            'is_male': input_data.is_male,
            'have_car': input_data.have_car,
            'CNT_MEMBERS': input_data.CNT_MEMBERS
        }])

        # Масштабирование
        input_scaled = scaler.transform(input_df)

        # Предсказание
        prediction = model.predict(input_scaled)[0]
        result = "Give" if prediction == 1 else "Not Give"

        return {
            "prediction": result,
            "raw_prediction": int(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

