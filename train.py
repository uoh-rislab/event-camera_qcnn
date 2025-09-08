#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Quantum Training for Event-based Datasets (TQR)
- Datasets: N-MNIST, N-CARS, DVS Gesture, e-CK+
- Input: TQR tensors saved as .npy shaped (H, W, BINS, 2) with values in {0,1}
- Pipeline: TQR -> (optional pooling) -> flatten -> PCA(k) -> Quantum Embedding (k qubits)
- Backend: PennyLane (quantum) + PyTorch (classical head)
"""

import os
import sys
import glob
import math
import random
import numpy as np
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --- PCA (scikit-learn) ---
try:
    from sklearn.decomposition import IncrementalPCA
except Exception as e:
    IncrementalPCA = None
    print("[WARN] scikit-learn no disponible. Implementa PCA manual si lo necesitas.", file=sys.stderr)

# --- PennyLane (quantum) ---
import pennylane as qml

# =========================
# Configuración del experimento
# =========================

DATA_ROOT = os.path.expanduser("~/cachefs/datasets/processed_data/recognition")

# Mapea nombre -> (subcarpeta, n_qubits, n_clases)
DATASETS = {
    "nmnist":      ("nmnist_rep_100ms",          8,  10),
    "ncars":       ("ncars_rep_100ms",           8,   2),
    "dvs-gesture": ("dvsgesture_rep_100ms",     12,  11),
    "eck+":        ("eck+_rep_100ms",           16,   7),
}

# Entrenamiento
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
TRAIN_SPLIT = 0.8
NUM_WORKERS = 4

# Preprocesamiento
USE_SPATIAL_POOLING = True     # pooling previo para reducir HxW antes de aplanar
POOL_FACTOR = 4                # reduce H->H/4, W->W/4 (aprox)
USE_TEMPORAL_AGG = False       # si True: colapsa BINS por OR/mean; si False, conserva BINS
BIN_AGG = "mean"               # "or" | "mean" (si USE_TEMPORAL_AGG=True)

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Utilidades
# =========================

def tqr_to_feature_vector(tqr: np.ndarray) -> np.ndarray:
    """
    tqr: (H, W, BINS, 2) con binarios 0/1
    Aplica pooling opcional y colapsos opcionales, luego flatten.
    """
    assert tqr.ndim == 4 and tqr.shape[-1] == 2, f"Esperado (H,W,BINS,2), got {tqr.shape}"

    H, W, B, C = tqr.shape
    vol = tqr.astype(np.float32)

    # Temporal aggregation (opcional)
    if USE_TEMPORAL_AGG:
        if BIN_AGG == "or":
            vol = (vol.sum(axis=2, keepdims=True) > 0).astype(np.float32)  # (H,W,1,2)
        elif BIN_AGG == "mean":
            vol = vol.mean(axis=2, keepdims=True)                           # (H,W,1,2)
        else:
            raise ValueError("BIN_AGG debe ser 'or' o 'mean'.")

    # Spatial pooling (opcional) vía reshape promedio simple
    if USE_SPATIAL_POOLING:
        # Ajuste para que sea divisible
        ph = max(1, H // POOL_FACTOR)
        pw = max(1, W // POOL_FACTOR)
        H2 = (H // ph) * ph
        W2 = (W // pw) * pw
        vol = vol[:H2, :W2]

        # reshape por bloques y promediar
        # vol shape: (H2, W2, B', 2)
        Bp = vol.shape[2]
        vol = vol.reshape(H2 // ph, ph, W2 // pw, pw, Bp, 2).mean(axis=(1,3))  # (H2/ph, W2/pw, B', 2)

    feat = vol.ravel().astype(np.float32)  # flatten
    return feat

def list_classes_from_subdirs(root: str) -> Dict[str, int]:
    """
    Asume estructura: root/class_name/*.npy
    Retorna: dict nombre_clase -> idx
    """
    class_dirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
    classes = [os.path.basename(d) for d in class_dirs]
    return {c: i for i, c in enumerate(classes)}

def is_tqr_array(arr: np.ndarray) -> bool:
    return arr.ndim == 4 and arr.shape[-1] == 2

def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Archivo {path} no contiene un ndarray")
    if not is_tqr_array(arr):
        raise ValueError(f"{path} no parece TQR (H,W,BINS,2): shape={arr.shape}")
    return arr

# =========================
# Dataset
# =========================

class EventTQRDataset(Dataset):
    def __init__(self, root: str, split: str, pca_model=None, max_files=None):
        """
        root: carpeta dataset (p.ej., ~/cachefs/.../nmnist_rep_100ms)
        split: 'train' o 'val'
        pca_model: sklearn IncrementalPCA fitted (para transformar)
        max_files: opcional, limitar nº de archivos para debug
        """
        self.root = root
        self.split = split
        self.pca = pca_model
        self.samples = self._discover_samples(max_files)

    def _discover_samples(self, max_files) -> List[Tuple[str, int]]:
        cls_map = list_classes_from_subdirs(self.root)
        if not cls_map:
            # Alternativa: archivos sueltos con patrón *_labelX.npy
            npys = sorted(glob.glob(os.path.join(self.root, "**", "*.npy"), recursive=True))
            # Inferencia cruda de label por nombre (ajústalo si tu naming es distinto)
            pairs = []
            for p in npys:
                base = os.path.basename(p)
                # intenta hallar "..._label{int}.npy"
                lbl = None
                for tok in base.replace("-", "_").split("_"):
                    if tok.lower().startswith("label"):
                        try:
                            lbl = int(tok.lower().replace("label", "").split(".")[0])
                        except:
                            pass
                if lbl is not None:
                    pairs.append((p, lbl))
            if not pairs:
                raise RuntimeError(f"No encontré estructura de clases en subcarpetas ni sufijo _labelX en {self.root}")
        else:
            pairs = []
            for cname, cid in cls_map.items():
                npys = sorted(glob.glob(os.path.join(self.root, cname, "*.npy")))
                pairs.extend([(p, cid) for p in npys])

        # split por índice estable
        rng = random.Random(SEED)
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(TRAIN_SPLIT * n)
        subset = pairs[:n_train] if self.split == "train" else pairs[n_train:]
        if max_files:
            subset = subset[:max_files]
        return subset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = load_npy(path)              # (H,W,BINS,2) 0/1
        feat = tqr_to_feature_vector(arr) # flatten float32

        if self.pca is not None:
            feat = self.pca.transform(feat[None, :]).squeeze(0).astype(np.float32)

        x = torch.from_numpy(feat)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# =========================
# PCA fitting util
# =========================

def fit_pca_for_dataset(root: str, k: int, max_fit_samples: int = 20000, batch: int = 256) -> IncrementalPCA:
    if IncrementalPCA is None:
        raise RuntimeError("scikit-learn no disponible. Instala scikit-learn o implementa PCA manual.")

    ipca = IncrementalPCA(n_components=k, batch_size=batch)

    # Reutiliza la lógica de splits para obtener training samples
    tmp_ds = EventTQRDataset(root=root, split="train", pca_model=None, max_files=max_fit_samples)
    loader = DataLoader(tmp_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    buf = []
    for i, (x, _) in enumerate(loader):
        # x aquí es feature vector sin PCA (porque pca_model=None)
        buf.append(x.numpy())
        if len(buf) >= batch:
            X = np.concatenate(buf, axis=0)
            ipca.partial_fit(X)
            buf = []
    if buf:
        X = np.concatenate(buf, axis=0)
        ipca.partial_fit(X)
    return ipca

# =========================
# Quantum + Torch modelo híbrido
# =========================

class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes

        # Dispositivo/cuántico
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Pesos del circuito
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # inputs: (n_qubits,) en [0,1] (ya reducido)
            # Embedding como ángulos
            qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # retornamos expectativas PauliZ (n_qubits,)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.qlayer = qml.qnn.TorchLayer(self.circuit, weight_shapes)

        # Cabeza clásica
        self.head = nn.Sequential(
            nn.Linear(n_qubits, max(16, n_qubits)),
            nn.ReLU(),
            nn.Linear(max(16, n_qubits), n_classes)
        )

    def forward(self, x):
        # x: (B, n_qubits) ya PCA
        qout = self.qlayer(x)      # (B, n_qubits)
        logits = self.head(qout)   # (B, n_classes)
        return logits

# =========================
# Entrenamiento
# =========================

def train_one(dataset_key: str):
    subdir, n_qubits, n_classes = DATASETS[dataset_key]
    root = os.path.join(DATA_ROOT, subdir)
    print(f"\n=== Dataset: {dataset_key} | root={root} | qubits={n_qubits} | classes={n_classes} ===")

    # 1) Fit PCA en train
    print("-> Fitting PCA...")
    pca = fit_pca_for_dataset(root, k=n_qubits, max_fit_samples=20000, batch=256)

    # 2) Datasets transformados con PCA
    ds_train = EventTQRDataset(root=root, split="train", pca_model=pca)
    ds_val   = EventTQRDataset(root=root, split="val",   pca_model=pca)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"-> Train samples: {len(ds_train)} | Val samples: {len(ds_val)}")

    # 3) Modelo
    model = QuantumClassifier(n_qubits=n_qubits, n_classes=n_classes, n_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) Optim / Loss
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        # ----- train -----
        model.train()
        tot_loss, tot_correct, tot_seen = 0.0, 0, 0
        for xb, yb in dl_train:
            xb = xb.to(device)  # (B, n_qubits)
            yb = yb.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            tot_loss += float(loss) * xb.size(0)
            tot_correct += (logits.argmax(dim=1) == yb).sum().item()
            tot_seen += xb.size(0)

        tr_loss = tot_loss / max(1, tot_seen)
        tr_acc = tot_correct / max(1, tot_seen)

        # ----- val -----
        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += float(loss) * xb.size(0)
                v_correct += (logits.argmax(dim=1) == yb).sum().item()
                v_seen += xb.size(0)
        val_loss = v_loss / max(1, v_seen)
        val_acc = v_correct / max(1, v_seen)

        if val_acc > best_val:
            best_val = val_acc

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | best {best_val:.3f}")

    print(f"[{dataset_key}] Best Val Acc = {best_val:.3f}")

# =========================
# Main
# =========================

if __name__ == "__main__":
    # Ejecuta en este orden; comenta los que no uses
    for key in ["nmnist", "ncars", "dvs-gesture", "eck+"]:
        try:
            train_one(key)
        except Exception as e:
            print(f"[ERROR] {key}: {e}", file=sys.stderr)

