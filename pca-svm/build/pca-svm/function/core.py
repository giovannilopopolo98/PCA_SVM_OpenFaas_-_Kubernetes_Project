# -*- coding: utf-8 -*-
"""
Self-adapting PCA/SVD + SVM  (memory-safe)
-----------------------------------------------------------------------------
- Funziona con qualsiasi mix di variabili numeriche / categoriche
- Oversampling (SMOTE/SMOTENC) solo se il dataset è “gestibile”
- Se tutte le feature diventano dense -> PCA
  altrimenti -> TruncatedSVD con scelta automatica di n_components
- Valutazione k-fold senza cross_val_predict (loop manuale -> RAM)
- Output: JSON con metriche, curva varianza cumulativa, warning
"""

# ---------- 0. IMPORT --------------------------------------------------------
from collections.abc import Mapping, Sequence


import json, logging
import numpy as np
import pandas as pd

# ---------- scikit-learn -----------------------------------------------------
from sklearn.base              import BaseEstimator, TransformerMixin, clone
from sklearn.compose           import ColumnTransformer
from sklearn.decomposition     import PCA, TruncatedSVD
from sklearn.metrics           import (accuracy_score,
                                       precision_recall_fscore_support,
                                       confusion_matrix)
from sklearn.model_selection   import StratifiedKFold
from sklearn.preprocessing     import OneHotEncoder, StandardScaler
from sklearn.svm               import SVC
from sklearn.utils             import compute_class_weight

# ---------- imbalanced-learn -------------------------------------------------
from imblearn.pipeline         import Pipeline as ImbPipeline
from imblearn.over_sampling    import SMOTE, SMOTENC

# ---------- logging ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from logging.handlers import MemoryHandler
_mem = MemoryHandler(1000)
logging.getLogger().addHandler(_mem)

# ---------- helper json-safe -------------------------------------------------
def _sanitize(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, pd.Series)):
        return [_sanitize(x) for x in o.tolist()]
    if isinstance(o, Mapping):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, Sequence) and not isinstance(o, (str, bytes, bytearray)):
        return [_sanitize(x) for x in o]
    return o

# ---------- costanti globali ------------------------------------------------- 
PCA_VAR_TARGET   = 0.95
IMBAL_THRESHOLD  = 5            # rapporto max/min classi
N_FOLDS_DEFAULT  = 5
RANDOM_STATE     = 42
MAX_SVD_INIT     = 512
SMOTE_MAX_CELLS  = 3e7          # se n_samples * n_features > soglia -> niente SMOTE

class BadRequest(Exception): ...

# ---------- 1. VALIDATE ------------------------------------------------------ 

# Load gestito da handler.py

def validate(df: pd.DataFrame):
    if df.shape[1] < 2:
        raise BadRequest("Richieste almeno 2 colonne (features + target).")
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    if y.isna().any():
        mask = ~y.isna()
        X, y = X.loc[mask], y.loc[mask]
        logging.warning("Righe con target NA rimosse.")
    if y.nunique() < 2:
        raise BadRequest("Il target deve avere >=2 classi.")
    return X, y

# ---------- 2.  PREPROCESSOR ----------------------------------------------------------
def build_preprocessor(X: pd.DataFrame):
    """Restituisce ColumnTransformer e indici delle feature categoriche dopo il prep."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # cast leggero a float32 SOLO sulle numeriche
    X[num_cols] = X[num_cols].astype(np.float32, copy=False)

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=False), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    # sparse_threshold = 1 -> restituisce CSR se la matrice è effettivamente sparsa
    ct = ColumnTransformer(transformers, sparse_threshold=1)

    # Indici (dopo il ColumnTransformer) delle feature categoriche -> SMOTENC
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    return ct, cat_indices

# ---------- 3.  RIDUTTORE AUTO-DIMENSIONI -----------------------------------------------
class AutoDimReducer(BaseEstimator, TransformerMixin):
    # PCA per matrici dense o TruncatedSVD per matrici sparse (varianza target).
    def __init__(self, var_target=PCA_VAR_TARGET,
                 max_init=MAX_SVD_INIT, random_state=RANDOM_STATE):
        self.var_target, self.max_init, self.random_state = var_target, max_init, random_state

    def fit(self, X, y=None):
        is_sparse = hasattr(X, "tocsc")  # CSR/CSC check
        if not is_sparse:
            self.model_ = PCA(n_components=self.var_target,
                              svd_solver="full",
                              random_state=self.random_state)
            self.model_.fit(X)
        else:
            init_k = min(self.max_init, X.shape[1] - 1)
            svd = TruncatedSVD(n_components=init_k,
                               random_state=self.random_state)
            svd.fit(X)
            k = np.searchsorted(np.cumsum(svd.explained_variance_ratio_),
                                self.var_target) + 1
            k = max(1, min(k, X.shape[1] - 1))
            if k != init_k:
                svd = TruncatedSVD(n_components=k,
                                   random_state=self.random_state)
                svd.fit(X)
            self.model_ = svd
        self.explained_variance_ratio_ = self.model_.explained_variance_ratio_
        return self

    def transform(self, X):
        return self.model_.transform(X)

# ---------- 4.  CLASS-WEIGHTS ------------------------------------------------------
def compute_weights(y):
    counts = y.value_counts()
    ratio  = counts.max() / counts.min()
    if ratio < IMBAL_THRESHOLD:
        return None
    weights = compute_class_weight(class_weight="balanced",
                                   classes=counts.index.to_numpy(),
                                   y=y.to_numpy())
    logging.info("Squilibrio classi rilevato (ratio=%.1f). Uso class_weight.", ratio)
    return dict(zip(counts.index, weights))

# ---------- 5.  PIPELINE ---------------------------------------------------------- 
def make_pipeline(prep, cat_idx, cls_w, n_samples, n_features):
    use_smote = (n_samples * n_features < SMOTE_MAX_CELLS)
    if use_smote:
        smote = (SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
                 if cat_idx else SMOTE(random_state=RANDOM_STATE))
        logging.info("SMOTE attivo (dataset %.0f M celle).",
                     n_samples * n_features / 1e6)
    else:
        smote = "passthrough"
        logging.warning("Dataset troppo grande (%.0f M celle) -> SMOTE saltato.",
                        n_samples * n_features / 1e6)

    return ImbPipeline([
        ("prep",   prep),
        ("smote",  smote),
        ("dimred", AutoDimReducer()),
        ("svm",    SVC(kernel="rbf",
                       class_weight=cls_w,
                       random_state=RANDOM_STATE)),
    ])

# ---------- 6.  VALUTAZIONE (loop k-fold memory-safe) ------------------------------ 
def evaluate(pipe, X: pd.DataFrame, y: pd.Series):
    cv = StratifiedKFold(
        n_splits=min(N_FOLDS_DEFAULT, y.value_counts().min(), 10),
        shuffle=True, random_state=RANDOM_STATE
    )

    y_pred = np.empty_like(y.to_numpy())
    for train_idx, test_idx in cv.split(X, y):
        fold_pipe = clone(pipe)
        fold_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = fold_pipe.predict(X.iloc[test_idx])

    # Fit finale per salvare la curva di varianza
    pipe.fit(X, y)
    expl = pipe.named_steps["dimred"].explained_variance_ratio_.cumsum()

    acc = float(accuracy_score(y, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "explained_var_cumsum": expl.tolist(),
        "n_folds": cv.get_n_splits(),
    }

def main_logic(df: pd.DataFrame) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X, y        = validate(df)
    prep, cat_i = build_preprocessor(X)
    cls_w       = compute_weights(y)
    pipe        = make_pipeline(prep, cat_i, cls_w, *X.shape)

    min_class_samples = y.value_counts().min()

    if min_class_samples < 2:
        raise BadRequest(f"Troppi pochi campioni in almeno una classe: {min_class_samples} < 2")
    if min_class_samples < 5:
        logging.warning("Dataset troppo piccolo per il k-fold, uso train_test_split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        pipe.fit(X, y)  # fit finale per explained variance
        expl = pipe.named_steps["dimred"].explained_variance_ratio_.cumsum()

        results = {
            "accuracy": float(report["accuracy"]),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "explained_var_cumsum": expl.tolist(),
            "n_folds": "train_test_split",
        }
    else:
        results = evaluate(pipe, X, y)

    results["class_weights"] = cls_w
    results["warnings"] = [rec.getMessage() for rec in _mem.buffer if rec.levelno >= logging.WARNING]
    return _sanitize(results)
