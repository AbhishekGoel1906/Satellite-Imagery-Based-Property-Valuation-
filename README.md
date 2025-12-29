# 🏠 Spatial Analysis & Multimodal Real Estate Price Prediction

This project implements an end-to-end **machine learning pipeline** for predicting real estate prices using **tabular property data combined with satellite image embeddings**.  
It includes data preprocessing, feature engineering, CNN-based embedding extraction, baseline models, multimodal fusion, and evaluation.

The repository is structured for **easy setup, reproducibility, and scalability**, following industry best practices.

---

## 📌 Project Highlights
- Clean and reproducible ML pipeline
- Strong tabular baseline models
- Satellite imagery embeddings using CNNs (ResNet)
- Multimodal feature fusion
- Cross-validated evaluation
- Designed for **local execution or Google Colab**

---

## 📂 Repository Structure
```
CDC_Project/
├── data/
│   ├── raw/                         # Original CSV datasets
│   ├── processed/                   # Cleaned & feature-engineered data
│   └── images/
│       ├── train/                   # Training satellite images
│       └── test/                    # Testing satellite images
│
├── train_image_embeddings.parquet   # Precomputed satellite embeddings (train)
├── test_image_embeddings.parquet    # Precomputed satellite embeddings (test)
│
├── Preprocessing.ipynb              # Tabular preprocessing + geo features
├── model_training.ipynb             # Model training with tabular + image features
├── data_fetcher.py                  # Image embedding loader / dataset builder
├── models/                          # Saved trained models
├── results/                         # Metrics & plots
├── requirements.txt
└── README.md

```
---

## 📦 Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Key libraries used:
- numpy, pandas
- scikit-learn
- matplotlib, seaborn
- torch, torchvision
- opencv-python
- tqdm

---

## 🚀 Project Setup (Step-by-Step)

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

---

### 2️⃣ Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 📊 Data Setup

### Tabular Dataset
Place the main CSV file inside:
```
data/raw/
```

Example:
```
data/raw/real_estate_data.csv
```

The dataset should contain:
- Latitude & Longitude
- Property attributes (area, rooms, age, etc.)
- Target variable (price)

---

### Satellite Images & Embeddings

You may already have images and embeddings stored on **Google Drive**.

#### Option A: Google Colab (Recommended for Large Data)
```python
from google.colab import drive
drive.mount('/content/drive')
```

Update paths inside the notebooks to point to your Drive folders.

#### Option B: Local Execution
Place files manually:
```
data/images/
data/embeddings/
```

⚠️ **Do NOT push images or embeddings to GitHub.**

---

## 🧪 Running the Pipeline

### Step 1: Data Preprocessing
```bash
jupyter notebook Preprocessing.ipynb
```

This step:
- Cleans missing values
- Performs feature engineering
- Applies log transformations
- Saves processed data to `data/processed/`

---

### Step 2: Model Training & Evaluation
```bash
jupyter notebook model_training.ipynb
```

This step:
- Trains tabular baseline models
- Performs multimodal fusion with embeddings
- Uses cross-validation
- Evaluates using RMSE and R²

Outputs:
```
models/    → saved trained models
results/   → metrics, plots, evaluation results
```

---

## 📈 Evaluation Metrics
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-Validation Mean & Standard Deviation
- Model comparison visualizations

---

## 🔁 Reproducibility
- Fixed random seeds
- Deterministic data splits
- Saved embeddings and trained models
- Same preprocessing pipeline for all runs

---

## 🛑 Common Pitfalls
- Ensure ID columns match between tabular data and embeddings
- Apply log-transforms consistently
- Avoid committing large files to GitHub

---

## 🔮 Future Work
- Transformer-based image encoders
- CNN fine-tuning on domain-specific data
- Hyperparameter optimization
- Real-time inference pipeline

---

## 👤 Author
**Abhishek Goel**  
Machine Learning | Data Science

---

## 📄 License
This project is licensed under the **MIT License**.

---

⭐ If you find this project useful, consider giving it a star!
