FROM python:3.10-slim

# Aggiungi tool di build e dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crea la cartella dell'app
WORKDIR /app

# Copia codice e requirements
COPY handler.py .
COPY core.py .
COPY requirements.txt .

# Installa dipendenze Python (incluso uvicorn)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Esponi la porta per il servizio HTTP
EXPOSE 8080

# Comando di avvio: avvia il server FastAPI
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8080"]

