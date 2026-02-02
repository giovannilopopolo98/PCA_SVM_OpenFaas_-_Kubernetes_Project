# 🚀 PCA-SVM Distributed Classifier on Kubernetes & OpenFaaS

![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)
![OpenFaaS](https://img.shields.io/badge/openfaas-%23377CE3.svg?style=for-the-badge&logo=openfaas&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)

Questo progetto implementa un'architettura **Cloud-Native** per l'addestramento distribuito e l'analisi di dataset tramite algoritmi di Machine Learning (**PCA** per la riduzione dimensionale e **SVM** per la classificazione).

Il sistema è orchestrato su **Kubernetes**, utilizza **OpenFaaS** per l'esecuzione serverless delle funzioni di calcolo e presenta un **Frontend React** moderno accessibile tramite Ingress Controller personalizzato.

Progetto realizzato per il corso di **Virtual Networks and Cloud Computing**.

---

## 🏗️ Architettura del Sistema

Il sistema segue un pattern a microservizi event-driven:

1.  **Frontend:** Interfaccia utente "Dark Mode" per il caricamento dei file CSV e la visualizzazione dei risultati (Matrice di confusione, varianza spiegata).
2.  **API Gateway (Python FastAPI):** Gestisce le richieste in ingresso, valida i dati e invoca le funzioni serverless.
3.  **Ingress Controller (Nginx):** Gestisce il routing del traffico tramite **Fan-out** su un unico dominio locale (`pca-svm.local`), smistando tra UI e API.
4.  **OpenFaaS Function:** Esegue il calcolo pesante (PCA + GridSearch SVM) in container effimeri e scalabili, restituendo i risultati in formato JSON.

### Flusso dei Dati
`User` ➡️ `Ingress (pca-svm.local)` ➡️ `Gateway` ➡️ `OpenFaaS (ML Function)` ➡️ `Response`

---

## ✨ Funzionalità Chiave

* **⚡ Serverless ML:** L'addestramento del modello avviene on-demand su funzioni OpenFaaS, garantendo scalabilità e ottimizzazione delle risorse.
* **🌐 Unified Ingress:** Configurazione avanzata con rewrite-target per esporre l'intera applicazione su un singolo dominio (`http://pcasvm.local`).
* **🧠 Algoritmi Avanzati:** * *PCA (Principal Component Analysis)*: Riduzione del 95% della varianza.
    * *SVM (Support Vector Machine)*: Classificazione con Cross-Validation.
* **🎨 Modern UI:** Interfaccia responsive con feedback in tempo reale.

---

## 🛠️ Tecnologie Utilizzate

* **Orchestrazione:** Kubernetes (Kubespray), Docker.
* **Serverless:** OpenFaaS (Function-as-a-Service).
* **Backend:** Python 3.9, FastAPI, Scikit-learn, Pandas.
* **Frontend:** React.js, CSS3 Custom Properties.
* **Networking:** Nginx Ingress Controller, MetalLB.

---

## 🚀 Installazione e Deploy

### Prerequisiti
* Cluster Kubernetes attivo.
* OpenFaaS installato (`arkade install openfaas`).
* Docker & Kubectl configurati.

### 1. Clona la repository
```bash
git clone [https://github.com/giovannilopopolo98/PCA_SVM_OpenFaas_-_Kubernetes_Project.git](https://github.com/giovannilopopolo98/PCA_SVM_OpenFaas_-_Kubernetes_Project.git)
cd PCA_SVM_OpenFaas_-_Kubernetes_Project
