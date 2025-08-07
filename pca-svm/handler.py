import json
import sys
import pandas as pd
from io import StringIO
from core import main_logic
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.get("/_/health")
async def health():
    """Health check endpoint"""
    return {"status": "ready"}

@app.post("/")
async def handle_request(request: Request):
    """
    Handle function per OpenFaaS
    Riceve il contenuto CSV come stringa e restituisce JSON con i risultati PCA+SVM
    """
    try:
        # Leggi il body della richiesta
        body = await request.body()

        # Il body contiene il CSV come stringa
        csv_content = body.decode('utf-8') if isinstance(body, bytes) else body

        # Debug: stampa le prime righe del CSV ricevuto
        print(f"Received CSV content (first 200 chars): {csv_content[:200]}", file=sys.stderr)

        # Carica CSV in DataFrame
        df = pd.read_csv(StringIO(csv_content))
        print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns", file=sys.stderr)
        print(f"Columns: {df.columns.tolist()}", file=sys.stderr)

        # Esegui la logica PCA+SVM
        result = main_logic(df)

        print("PCA+SVM processing completed successfully", file=sys.stderr)

        # Restituisci il risultato come JSON
        return JSONResponse(content=result)

    except Exception as e:
        error_message = str(e)
        print(f"Error in handle function: {error_message}", file=sys.stderr)

        error_response = {
            "error": error_message,
            "status": "failed",
            "type": type(e).__name__
        }
        return JSONResponse(content=error_response, status_code=500)

