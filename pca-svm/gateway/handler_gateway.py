from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import httpx
import io


app = FastAPI()

# Abilita CORS (opzionale ma utile per debug)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Formato file non valido. Atteso .csv"}

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        return {"error": f"Errore nella lettura del CSV: {e}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://gateway.openfaas.svc.cluster.local:8080/function/pca-svm",  # nome della funzione OpenFaaS
                json={"data": df.to_dict(orient="records")}
            )
            return response.json()
    except Exception as e:
        return {"error": f"Errore nella richiesta a OpenFaaS: {e}"}
