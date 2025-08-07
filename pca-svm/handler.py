from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import StringIO
import pandas as pd
from core import mainlogic  # importa la logica di PCA+SVM
app = FastAPI()
@app.get("//health")
async def health_check():
    return JSONResponse(content={"status": "ok"})
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Legge i dati del CSV e li carica in un DataFrame
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        # Esegue la logica di preprocessing + PCA + SVM
        output = main_logic(df)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))