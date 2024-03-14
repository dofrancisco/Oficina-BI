import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File

# Criando o app
app = FastAPI(docs_url="/", title='Oficina-BI')

# Carregando pipeline de preprocessamento e inferencia
pipeline = joblib.load('breast_pipeline.pkl')


# Criar rota pro endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict usando pkl do pipeline já treinado.
    """
    df = pd.read_csv(file.file, index_col=0)
    pred = pipeline.predict(df)
    return {"prediction": pred.tolist()}
