import pandas as pd
from sklearn.datasets import make_classification

# CONFIGURAZIONE STRESS TEST
N_SAMPLES = 2000   # Numero di righe (Iris ne ha 150)
N_FEATURES = 20    # Numero di colonne (Iris ne ha 4)
N_CLASSES = 2       # Classificazione binaria
W_IMBALANCE = 0.90  # 90% classe 0, 10% classe 1 (Per testare SMOTE)

print(f"Generazione dataset: {N_SAMPLES} righe, {N_FEATURES} features...")

# Genera dati sintetici ma "difficili"
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=int(N_FEATURES * 0.2), # Solo il 20% dei dati è utile, il resto è rumore (ottimo per PCA)
    n_redundant=int(N_FEATURES * 0.1),
    n_classes=N_CLASSES,
    weights=[W_IMBALANCE], # Sbilanciato per forzare l'uso di SMOTE
    random_state=42
)

# Crea DataFrame
col_names = [f"feat_{i}" for i in range(N_FEATURES)]
df = pd.DataFrame(X, columns=col_names)
df['target'] = y  # Il target è l'ultima colonna

# Salva in CSV
filename = "stress_test.csv"
df.to_csv(filename, index=False)

print(f"Dataset salvato come '{filename}'.")
print(f"Dimensioni file: {df.shape}")
