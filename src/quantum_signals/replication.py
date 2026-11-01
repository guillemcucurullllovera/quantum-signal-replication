"""Original first-party statistical CV helpers; no private data or result registry."""
from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from .kernels import FEATURE_MAPS,compute_rdms_for_dataset,qrbf_kernel_matrix


def fit_angle_scaler(training_features):
    """Fit only on the training fold; do not clip or refit using held-out extrema."""
    return MinMaxScaler(feature_range=(-np.pi,np.pi)).fit(training_features)


def run_quantum_cv(X,y,feature_map_name,gamma=0.1,nu=0.1,k=10,seed=42):
    """Raw-feature IID cross-validation, with per-fold angle scaling and encoding."""
    X=np.asarray(X,float);y=np.asarray(y)
    if X.ndim!=2 or X.shape[1]!=20 or len(X)!=len(y) or not np.all(np.isfinite(X)):
        raise ValueError('finite raw features of shape (samples,20) and aligned labels required')
    feature_map=FEATURE_MAPS[feature_map_name]
    folds=StratifiedKFold(n_splits=k,shuffle=True,random_state=seed)
    train_scores=[];test_scores=[]
    for train,test in folds.split(X,y):
        scaler=fit_angle_scaler(X[train])
        r_train=compute_rdms_for_dataset(scaler.transform(X[train]),feature_map)
        r_test=compute_rdms_for_dataset(scaler.transform(X[test]),feature_map)
        K_train=qrbf_kernel_matrix(r_train,r_train,gamma)
        K_test=qrbf_kernel_matrix(r_test,r_train,gamma)
        model=OneClassSVM(kernel='precomputed',nu=nu).fit(K_train)
        train_scores.append(f1_score_anomaly(y[train],ocsvm_predict_anomaly(model,K_train)))
        test_scores.append(f1_score_anomaly(y[test],ocsvm_predict_anomaly(model,K_test)))
    return {'train_mean':float(np.mean(train_scores)),'train_std':float(np.std(train_scores)),
            'test_mean':float(np.mean(test_scores)),'test_std':float(np.std(test_scores)),
            'per_fold_test':test_scores,'scaling':'training fold only','execution':'EXACT_STATEVECTOR'}

def f1_score_anomaly(y_true: np.ndarray, y_pred_anomaly: np.ndarray) -> float:
    """F1 with anomalies as the positive class.

    y_true: 1 for anomaly (fraud), 0 for normal.
    y_pred_anomaly: 1 if predicted anomaly, 0 if predicted normal.
    """
    tp = int(((y_pred_anomaly == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_anomaly == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_anomaly == 0) & (y_true == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def ocsvm_predict_anomaly(model: OneClassSVM, X_or_K: np.ndarray) -> np.ndarray:
    """sklearn OneClassSVM returns +1 = inlier, -1 = outlier. Map outlier->1."""
    raw = model.predict(X_or_K)
    return (raw == -1).astype(int)


def run_classical_cv(
    X: np.ndarray, y: np.ndarray, gamma: float = 0.1, nu: float = 0.1, k: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """k-fold CV of classical rbf-OCSVM."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_f1s, test_f1s = [], []

    for tr_idx, te_idx in skf.split(X, y):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
        model.fit(Xtr_s)

        train_f1s.append(f1_score_anomaly(ytr, ocsvm_predict_anomaly(model, Xtr_s)))
        test_f1s.append(f1_score_anomaly(yte, ocsvm_predict_anomaly(model, Xte_s)))

    return {
        "train_mean": float(np.mean(train_f1s)), "train_std": float(np.std(train_f1s)),
        "test_mean": float(np.mean(test_f1s)), "test_std": float(np.std(test_f1s)),
        "per_fold_test": test_f1s,
    }
