
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Quantum Training for Event-based Datasets (TQR)
- Matches layout like:
  /app/input/{nmnist_rep_100ms,ncars_rep_100ms,dvsgesture_rep_100ms,eck+_rep_100ms}/tqr_tensor/{train,test}/<class>/*.npy
  (Case-insensitive: Train, Test, Train_Set, Test_Set also supported.)
"""

import os
import re
import sys
import math
import glob
import random
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.decomposition import IncrementalPCA
except Exception:
    IncrementalPCA = None
    print("[WARN] scikit-learn not available. Please install scikit-learn.", file=sys.stderr)

import pennylane as qml

# ----------------------
# Defaults
# ----------------------

DATASETS = {
    "nmnist":      ("nmnist_rep_100ms",          8,  10),
    "ncars":       ("ncars_rep_100ms",           8,   2),
    "dvs-gesture": ("dvsgesture_rep_100ms",     12,  11),
    "eck+":        ("eck+_rep_100ms",           16,   7),
}

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
NUM_WORKERS = 4

USE_SPATIAL_POOLING = True
POOL_FACTOR = 4
USE_TEMPORAL_AGG = False
BIN_AGG = "mean"

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------
# Utils
# ----------------------

def is_tqr_array(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[-1] == 2

def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if not is_tqr_array(arr):
        raise ValueError(f"{path} shape={getattr(arr, 'shape', None)} is not TQR (H,W,BINS,2)")
    return arr

def tqr_to_feature_vector(tqr: np.ndarray) -> np.ndarray:
    assert tqr.ndim == 4 and tqr.shape[-1] == 2, f"Expected (H,W,BINS,2), got {tqr.shape}"
    H, W, B, C = tqr.shape
    vol = tqr.astype(np.float32)

    if USE_TEMPORAL_AGG:
        if BIN_AGG == "or":
            vol = (vol.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        elif BIN_AGG == "mean":
            vol = vol.mean(axis=2, keepdims=True)
        else:
            raise ValueError("BIN_AGG must be 'or' or 'mean'")

    if USE_SPATIAL_POOLING:
        ph = max(1, H // POOL_FACTOR)
        pw = max(1, W // POOL_FACTOR)
        H2 = (H // ph) * ph
        W2 = (W // pw) * pw
        vol = vol[:H2, :W2]
        Bp = vol.shape[2]
        vol = vol.reshape(H2 // ph, ph, W2 // pw, pw, Bp, 2).mean(axis=(1,3))

    return vol.ravel().astype(np.float32)

# Case-insensitive train/test dir names that we support
TRAIN_DIR_CANDIDATES = ["train", "Train", "TRAIN", "Train_Set", "TRAIN_SET"]
TEST_DIR_CANDIDATES  = ["test", "Test", "TEST", "Test_Set", "TEST_SET", "val", "Val", "VAL"]

def find_split_dirs(base: str) -> Tuple[Optional[str], Optional[str]]:
    train_dir = None
    test_dir = None
    for d in TRAIN_DIR_CANDIDATES:
        p = os.path.join(base, d)
        if os.path.isdir(p):
            train_dir = p
            break
    for d in TEST_DIR_CANDIDATES:
        p = os.path.join(base, d)
        if os.path.isdir(p):
            test_dir = p
            break
    return train_dir, test_dir

def list_classes_from_subdirs(root: str) -> Dict[str, int]:
    class_dirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
    classes = [os.path.basename(d) for d in class_dirs]
    if not classes:
        return {}
    return {c: i for i, c in enumerate(classes)}

def infer_label_from_filename(dataset_key: str, path: str) -> Optional[int]:
    """Fallback heuristics; only used when there are no class subfolders."""
    name = os.path.basename(path).lower()
    parts = re.split(r"[_\-\.\s/]+", path.lower())

    if dataset_key == "nmnist":
        m = re.search(r"(?:^|[_\-\s])([0-9])(?:[_\-\s]|\.|$)", name)
        if m: return int(m.group(1))
        for p in parts:
            if p.isdigit() and len(p) == 1:
                return int(p)

    if dataset_key == "ncars":
        if any(k in parts for k in ["car", "cars", "pos", "positive", "vehicle"]): return 1
        if any(k in parts for k in ["background", "bg", "neg", "negative", "noncar"]): return 0
        m = re.search(r"label([01])", name)
        if m: return int(m.group(1))

    if dataset_key == "dvs-gesture":
        m = re.search(r"label([0-9]{1,2})", name)
        if m:
            return int(m.group(1))
        for p in parts:
            if p.isdigit():
                v = int(p)
                if 0 <= v <= 10:
                    return v

    if dataset_key == "eck+":
        EMOS = {"anger":0,"angry":0,"disgust":1,"fear":2,"happy":3,"happiness":3,"sad":4,"sadness":4,"surprise":5,"neutral":6}
        for k, idx in EMOS.items():
            if k in parts or k in name:
                return idx
        m = re.search(r"label([0-6])", name)
        if m: return int(m.group(1))

    m = re.search(r"label([0-9]+)", name)
    if m: return int(m.group(1))

    return None

def collect_pairs_from_split(split_root: str, dataset_key: str) -> List[Tuple[str,int]]:
    """Return list of (path,label) from split_root.
       Prefer class subfolders; fallback to filename inference.
    """
    pairs: List[Tuple[str,int]] = []
    cls_map = list_classes_from_subdirs(split_root)
    if cls_map:
        for cname, cid in cls_map.items():
            npys = sorted(glob.glob(os.path.join(split_root, cname, "*.npy")))
            pairs.extend([(p, cid) for p in npys])
    else:
        # no class folders; scan npys and infer labels
        npys = sorted(glob.glob(os.path.join(split_root, "**", "*.npy"), recursive=True))
        for p in npys:
            lbl = infer_label_from_filename(dataset_key, p)
            if lbl is not None:
                pairs.append((p, lbl))
    return pairs

# ----------------------
# Dataset class
# ----------------------

class EventTQRDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], pca_model=None):
        self.samples = samples
        self.pca = pca_model

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = load_npy(path)
        feat = tqr_to_feature_vector(arr)
        if self.pca is not None:
            feat = self.pca.transform(feat[None, :]).squeeze(0).astype(np.float32)
        x = torch.from_numpy(feat)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# ----------------------
# PCA
# ----------------------

def fit_pca(samples: List[Tuple[str,int]], k: int, batch: int = 256) -> IncrementalPCA:
    if IncrementalPCA is None:
        raise RuntimeError("scikit-learn not available. Install scikit-learn.")
    ipca = IncrementalPCA(n_components=k, batch_size=batch)
    buf = []
    for i, (p, _) in enumerate(samples):
        arr = load_npy(p)
        feat = tqr_to_feature_vector(arr)[None, :]
        buf.append(feat)
        if len(buf) >= batch:
            X = np.concatenate(buf, axis=0)
            ipca.partial_fit(X)
            buf = []
    if buf:
        X = np.concatenate(buf, axis=0)
        ipca.partial_fit(X)
    return ipca

# ----------------------
# Quantum model
# ----------------------

class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.qlayer = qml.qnn.TorchLayer(self.circuit, weight_shapes)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, max(16, n_qubits)),
            nn.ReLU(),
            nn.Linear(max(16, n_qubits), n_classes)
        )

    def forward(self, x):
        qout = self.qlayer(x)
        return self.head(qout)

# ----------------------
# Training loop
# ----------------------

def train_one(dataset_key: str, data_root: str, rep: str, epochs: int, batch_size: int):
    subdir, n_qubits, n_classes = DATASETS[dataset_key]
    # representation root (e.g., tqr_tensor)
    rep_root = os.path.join(data_root, subdir, rep)
    if not os.path.isdir(rep_root):
        raise RuntimeError(f"Representation folder not found: {rep_root}")

    # find split dirs
    train_dir, test_dir = find_split_dirs(rep_root)
    if train_dir is None:
        raise RuntimeError(f"Could not find a train split in {rep_root}. Expected one of: {TRAIN_DIR_CANDIDATES}")
    if test_dir is None:
        print(f"[WARN] No explicit test/val split found in {rep_root}; will split train 80/20.")
        # collect all from train_dir and then split
        all_pairs = collect_pairs_from_split(train_dir, dataset_key)
        if not all_pairs:
            raise RuntimeError(f"No samples found under {train_dir}")
        n = len(all_pairs)
        n_train = int(0.8 * n)
        pairs_train, pairs_val = all_pairs[:n_train], all_pairs[n_train:]
    else:
        pairs_train = collect_pairs_from_split(train_dir, dataset_key)
        pairs_val   = collect_pairs_from_split(test_dir,  dataset_key)
        if not pairs_train:
            raise RuntimeError(f"No training samples in {train_dir}")
        if not pairs_val:
            raise RuntimeError(f"No validation/test samples in {test_dir}")

    print(f"=== Dataset: {dataset_key} | rep={rep} ===")
    print(f"-> Train: {len(pairs_train)} | Val/Test: {len(pairs_val)} | qubits={n_qubits} | classes={n_classes}")

    # Fit PCA on train
    print("-> Fitting PCA on train split...")
    pca = fit_pca(pairs_train, k=n_qubits, batch=256)

    ds_train = EventTQRDataset(pairs_train, pca_model=pca)
    ds_val   = EventTQRDataset(pairs_val,   pca_model=pca)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = QuantumClassifier(n_qubits=n_qubits, n_classes=n_classes, n_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0; tot_correct = 0; tot_seen = 0
        for xb, yb in dl_train:
            xb = xb.to(device); yb = yb.to(device)
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

        model.eval()
        v_loss = 0.0; v_correct = 0; v_seen = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += float(loss) * xb.size(0)
                v_correct += (logits.argmax(dim=1) == yb).sum().item()
                v_seen += xb.size(0)
        val_loss = v_loss / max(1, v_seen)
        val_acc = v_correct / max(1, v_seen)

        best_val = max(best_val, val_acc)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} | best {best_val:.3f}")

    print(f"[{dataset_key}] Best Val Acc = {best_val:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum Training on Event-based Datasets (TQR)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "nmnist", "ncars", "dvs-gesture", "eck+"],
                        help="Dataset to train (default: all)")
    parser.add_argument("--data_root", type=str, default="/app/input",
                        help="Root folder containing dataset subfolders (default: /app/input)")
    parser.add_argument("--rep", type=str, default="tqr_tensor",
                        help="Representation subfolder to use (default: tqr_tensor)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    if args.dataset == "all":
        keys = ["nmnist", "ncars", "dvs-gesture", "eck+"]
    else:
        keys = [args.dataset]

    for key in keys:
        try:
            train_one(key, data_root=args.data_root, rep=args.rep, epochs=args.epochs, batch_size=args.batch_size)
        except Exception as e:
            print(f"[ERROR] {key}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
