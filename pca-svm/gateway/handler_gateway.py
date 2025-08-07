from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import httpx
import pandas as pd
from io import StringIO
import base64
import json

app = FastAPI(root_path="/upload")

# URL corretto per OpenFaaS - solo /function/nome-funzione
OPENFAAS_GATEWAY = "http://gateway.openfaas.svc.cluster.local:8080"
FUNCTION_ENDPOINT = "/function/pca-svm"  # Rimosso /predict
BASIC_AUTH = "vncc:vncc"
ENCODED_AUTH = base64.b64encode(BASIC_AUTH.encode()).decode()

@app.get("/_/health")
async def health_check():
    return {"status": "ok"}

@app.post("/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        content = await file.read()
        # Validazione CSV
        df = pd.read_csv(StringIO(content.decode()))
        print(f"CSV validated: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    try:
        headers = {
            "Authorization": f"Basic {ENCODED_AUTH}",
            "Content-Type": "text/plain"  # OpenFaaS si aspetta text/plain per CSV
        }

        # Invia il contenuto CSV direttamente, non come multipart
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENFAAS_GATEWAY}{FUNCTION_ENDPOINT}",
                content=content.decode(),  # Invia CSV come testo
                headers=headers
            )

        print(f"OpenFaaS response status: {response.status_code}")

        if response.status_code == 200:
            try:
                # La funzione dovrebbe restituire JSON
                result = response.json()
                return JSONResponse(status_code=200, content=result)
            except json.JSONDecodeError:
                # Se non è JSON, restituisci il testo
                return JSONResponse(status_code=200, content={"result": response.text})
        else:
            print(f"OpenFaaS error: {response.text}")
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Function error: {response.text}"
            )
    except httpx.RequestError as e:
        print(f"Request error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Gateway error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
