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
├── requirements.txt
└── README.md

```
---

## 📦 Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Key libraries used in this project:
- numpy, pandas (data handling)
- scikit-learn (modeling & evaluation)
- matplotlib, seaborn (visualization)
- torch, torchvision (CNN & embeddings)
- opencv-python (image processing)
- tqdm (progress tracking)

---

## 🚀 Project Setup (Step-by-Step)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AbhishekGoel1906/Satellite-Imagery-Based-Property-Valuation.git
cd Satellite-Imagery-Based-Property-Valuation
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

Place the real estate dataset CSV file inside:
```
data/raw/
```

Example:
```
data/raw/house_data.csv
```

### Dataset Description

Each row represents a residential property with the following attributes:

| Column Name | Description |
|------------|-------------|
| `id` | Unique property identifier |
| `date` | Date of sale |
| `price` | Sale price (target variable) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Living area in square feet |
| `sqft_lot` | Lot size in square feet |
| `floors` | Number of floors |
| `waterfront` | Waterfront presence (0 = No, 1 = Yes) |
| `view` | Quality of view (ordinal) |
| `condition` | Overall condition rating |
| `grade` | Construction and design grade |
| `sqft_above` | Square feet above ground |
| `sqft_basement` | Square feet of basement |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year of renovation (0 if none) |
| `zipcode` | Postal code |
| `lat` | Latitude |
| `long` | Longitude |
| `sqft_living15` | Living area of nearby homes |
| `sqft_lot15` | Lot size of nearby homes |

The **target variable** for prediction is:
```
price
```

---

## 🛰️ Satellite Images & Embeddings

Satellite imagery is used to capture **neighborhood-level and environmental context**.

You may already have satellite images and CNN embeddings stored on **Google Drive**.

### Option A: Google Colab (Recommended for Large Image Data)
```python
from google.colab import drive
drive.mount('/content/drive')
```

Update image and embedding paths inside the notebooks to point to your Drive directories.

---

### Option B: Local Execution

If running locally, place files as follows:
```
data/images/        # Satellite images (e.g., <id>.jpg)
data/embeddings/    # CNN embeddings (.npy or .parquet)
```

⚠️ **Do NOT commit satellite images or embeddings to GitHub.**

---

## 🧪 Running the Pipeline

### Step 1: Data Preprocessing
```bash
jupyter notebook Preprocessing.ipynb
```

This step performs:
- Missing value handling
- Feature engineering (ratios, age, log transforms)
- Geographic feature processing (lat-long usage)
- Saving cleaned data to `data/processed/`

---

### Step 2: Model Training & Evaluation
```bash
jupyter notebook model_training.ipynb
```

This step:
- Trains tabular baseline regression models
- Integrates satellite image embeddings for multimodal learning
- Uses cross-validation for robustness
- Evaluates models using RMSE and R²

Outputs generated:
```
models/    → saved trained models
results/   → evaluation metrics, plots, comparisons
```

---

## 📈 Evaluation Metrics

Models are evaluated using:
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation mean and standard deviation
- Tabular vs Multimodal performance comparison

---

## 🔁 Reproducibility

To ensure reproducibility:
- Fixed random seeds are used
- Deterministic train-validation splits
- Saved CNN embeddings and trained models
- Identical preprocessing pipeline across runs

---

## 🛑 Common Pitfalls

- Ensure property `id` matches between tabular data and satellite images
- Apply log transformation to `price` consistently during training and evaluation
- Avoid committing large binary files (images, embeddings) to GitHub

---

## 🔮 Future Work

- Fine-tuning CNNs on real estate imagery
- Transformer-based vision models
- Hyperparameter optimization
- Deployment-ready inference pipeline

---

## 👤 Author

**Abhishek Goel**  
Machine Learning | Data Science

---

## 📄 License

This project is licensed under the **MIT License**.

---

⭐ If you find this project useful, consider giving it a star!
