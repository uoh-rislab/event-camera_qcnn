# Hybrid Quantum-Classical Training on Event-based Datasets (TQR)

This repo provides a **hybrid training pipeline** (Quantum + Classical) for event-based datasets using the **Temporal Quantum Representation (TQR)**.

Supported datasets (processed into `.npy` TQR tensors):
- **N-MNIST** (10 classes, 34×34, ATIS)
- **N-CARS** (2 classes, 120×120, ATIS)
- **DVS Gesture** (11 classes, 128×128, DVS128)
- **e-CK+** (7 classes, 346×260, DAVIS346)

---

## 🔧 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Structure
Expected directory format:
```
datasets/processed_data/recognition/
  ├── nmnist_rep_100ms/
  │     ├── 0/  (class folder with .npy TQR tensors)
  │     ├── 1/
  │     └── ...
  ├── ncars_rep_100ms/
  ├── dvsgesture_rep_100ms/
  └── eck+_rep_100ms/
```

Each `.npy` file should contain a tensor `(H, W, BINS, 2)` with binary values:
- `[..., 0]` → negative polarity
- `[..., 1]` → positive polarity

---

## ▶️ Training
Run training for all datasets:
```bash
python train.py
```

Or train a single dataset:
```bash
python train.py --dataset nmnist
```

---

## ⚙️ Pipeline Overview
1. **Load TQR** tensors (`.npy`)
2. **Preprocessing**: optional spatial pooling, flatten
3. **PCA reduction** → target dimension = number of qubits  
   - N-MNIST → 8  
   - N-CARS → 8  
   - DVS Gesture → 12  
   - e-CK+ → 16  
4. **Quantum embedding**: AngleEmbedding + StronglyEntanglingLayers
5. **Classical head**: Linear + ReLU + Linear → Softmax

---

## 📊 Output
- Training/validation accuracy per epoch is printed.
- Best validation accuracy is reported per dataset.

---

## 🧪 Example
```bash
python train.py --dataset ncars --epochs 10 --batch_size 32
```
```
=== Dataset: ncars | qubits=8 | classes=2 ===
Epoch 01 | train loss 0.6932 acc 0.51 | val loss 0.6920 acc 0.55 | best 0.55
...
```
