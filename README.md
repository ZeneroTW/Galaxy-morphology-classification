# 🌌 Galaxy Morphology Classification — ResNet50 + Transfer Learning

> 🇷🇺 [Читать на русском](README_galaxy.ru.md)

A deep learning project for automatic classification of galaxy morphological types from images. Uses the **Galaxy10 SDSS** dataset (21,785 galaxy images across 10 morphology classes) and a two-phase transfer learning strategy based on **ResNet50** pretrained on ImageNet.

Achieved **84.00% test accuracy** on a 10-class imbalanced dataset.

---

## Dataset

**Galaxy10 SDSS** — an open astronomical dataset designed as an ML benchmark, built from SDSS survey images with labels from Galaxy Zoo volunteers (consensus threshold >55%).

| Property | Value |
|---|---|
| Total images | 21,785 |
| Image size | 69×69 px, 3 channels (g, r, i SDSS bands) |
| Classes | 10 morphological types |
| Format | HDF5 (.h5), ~210 MB |
| Balance | Imbalanced (Round Smooth dominates) |

### Classes

| # | Name |
|---|---|
| 0 | Disturbed Galaxies |
| 1 | Merging Galaxies |
| 2 | Round Smooth Galaxies |
| 3 | In-between Round Smooth |
| 4 | Cigar Shaped Smooth |
| 5 | Barred Spiral Galaxies |
| 6 | Unbarred Tight Spiral |
| 7 | Unbarred Loose Spiral |
| 8 | Edge-on without Bulge |
| 9 | Edge-on with Bulge |

---

## Architecture

**ResNet50** pretrained on ImageNet, with a custom classification head:

```
ResNet50 backbone (frozen in Phase 1)
    └── fc: Linear(2048 → 256) → ReLU → Dropout(0.5) → Linear(256 → 10)
```

ResNet50 was chosen over alternatives (VGG16, EfficientNetB0, MobileNetV2) for its optimal balance of depth (50 layers with residual connections), parameter count (25M), and ImageNet accuracy (76.1% Top-1).

---

## Training Strategy

Two-phase transfer learning:

**Phase 1 — Head training (10 epochs, LR=1e-3)**
- Entire backbone frozen
- Only the custom head (~0.5M params) is trained
- Val Accuracy: **53.88%**

**Phase 2 — Fine-tuning layer4 (15 epochs, LR=1e-4)**
- layer4 (~14.3M params) and head unfrozen; layer1–3 remain frozen
- CosineAnnealingLR scheduler
- Val Accuracy: **81.16%**
- Best checkpoint saved as `resnet50_best.pth`

**Final Test Accuracy: 84.00%**

---

## Results

| Metric | Value |
|---|---|
| Val Accuracy — Phase 1 | 53.88% |
| Val Accuracy — Phase 2 | 81.16% |
| **Test Accuracy (final)** | **84.00%** |
| Test Loss | ≈0.50 |

Highest per-class accuracy: Edge-on galaxies (class 8, 9) and Cigar Shaped (class 4) — geometrically distinctive shapes.

Most confused pairs: Round Smooth ↔ In-between Round, Barred Spiral ↔ Unbarred Tight Spiral — morphologically similar classes, difficult even for human annotators.

---

## Tech Stack

- **Python 3**
- **PyTorch + torchvision** — model, training, transforms
- **h5py** — HDF5 dataset loading
- **scikit-learn** — train/val/test split, metrics
- **Matplotlib + Seaborn** — visualizations
- **Yandex DataSphere** — GPU compute environment

---

## Project Structure

```
├── galaxy10_resnet50.ipynb    # Main notebook
├── resnet50_best.pth          # Best model checkpoint (Phase 2)
├── resnet50_phase1.pth        # Phase 1 checkpoint
├── galaxy_samples.png         # Sample images from each class
├── class_distribution.png     # Class imbalance chart
├── confusion_matrix.png       # Confusion matrix on test set
├── predictions.png            # Sample predictions (green=correct, red=wrong)
├── README.md                  # This file (English)
└── README.ru.md               # Russian version
```

> **Note:** `Galaxy10.h5` (~210 MB) is not included — it is downloaded automatically when running the notebook.

---

## Getting Started

```bash
# Install dependencies
pip install torch torchvision h5py scikit-learn matplotlib seaborn

# Run the notebook top to bottom
# Galaxy10.h5 will be downloaded automatically on first run
```

Recommended: run on GPU (CUDA). Training on CPU is possible but significantly slower.

---

*Academic project (Practical Assignment #1) — RTU MIREA, Institute of Artificial Intelligence.  
Course: AI Systems and Big Data. Supervisor: K.E. Medvedev.*
