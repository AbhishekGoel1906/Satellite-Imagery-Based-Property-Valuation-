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
- Designed for **local execution or Google Colab**
- Comprehensive project report generation (PDF and Word)

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.8+** (Python 3.9 or higher recommended)
- **pip** (Python package manager)
- **Jupyter Notebook** or **JupyterLab** (for running notebooks)
- **Git** (for cloning the repository)
- **Google Maps API Key** (for fetching satellite images - optional, if using pre-computed embeddings)

---

## 📂 Repository Structure

```
Satellite-Imagery-Based-Property-Valuation/
├── data/
│   ├── train.xlsx                    # Training dataset (place your file here)
│   ├── test.xlsx                     # Test dataset (place your file here)
│   └── images/
│       ├── train/                    # Training satellite images
│       └── test/                     # Testing satellite images
│
├── train_image_embeddings.parquet   # Precomputed satellite embeddings (train)
├── test_image_embeddings.parquet    # Precomputed satellite embeddings (test)
│
├── Preprocessing.ipynb              # Tabular preprocessing + geo features
├── model_training.ipynb             # Model training with tabular + image features
├── data_fetcher.py                  # Image embedding loader / dataset builder
├── generate_report.py                # Generate PDF project report
├── generate_report_docx.py          # Generate Word document project report
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/AbhishekGoel1906/Satellite-Imagery-Based-Property-Valuation.git
cd Satellite-Imagery-Based-Property-Valuation
```

---

### Step 2: Set Up Python Environment

#### Option A: Using Virtual Environment (Recommended)

**For macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**For Windows:**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Option B: Using Conda (Alternative)

```bash
# Create conda environment
conda create -n property_valuation python=3.9
conda activate property_valuation
```

---

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**Key libraries installed:**
- `pandas>=2.0` - Data manipulation and analysis
- `numpy>=1.23` - Numerical computing
- `scikit-learn>=1.3` - Machine learning algorithms
- `xgboost>=1.7` - Gradient boosting framework
- `torch>=2.0`, `torchvision>=0.15` - Deep learning (CNN embeddings)
- `matplotlib>=3.7`, `seaborn>=0.13` - Data visualization
- `geopy>=2.4` - Geospatial calculations
- `python-docx` - Word document generation (for reports)
- `openpyxl` or `xlrd` - Excel file reading

**Note:** If you encounter issues installing PyTorch, visit [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific installation instructions.

---

### Step 4: Prepare Your Data

1. **Place your training and test datasets** in the `data/` directory:
   ```
   data/
   ├── train.xlsx    # Your training dataset
   └── test.xlsx     # Your test dataset
   ```

2. **Dataset Format:**
   - Files should be in Excel format (`.xlsx`)
   - Training file must include a `price` column (target variable)
   - Both files should have matching column names
   - Required columns: `id`, `lat`, `long`, and other property attributes

3. **Verify dataset structure:**
   ```python
   import pandas as pd
   train = pd.read_excel("data/train.xlsx")
   print(train.head())
   print(train.shape)
   ```

---

## 🧪 Running the Pipeline

### Step 1: Data Preprocessing

Open and run the preprocessing notebook:

```bash
jupyter notebook Preprocessing.ipynb
```

**What this notebook does:**
- Loads and explores the training dataset
- Performs exploratory data analysis (EDA)
- Creates feature engineering:
  - Distance from city center calculation
  - House age computation
  - Property ratios (sqft_ratio, bath_per_bed)
- Visualizes price distributions and spatial patterns
- Prepares data for modeling

**How to run:**
1. Open `Preprocessing.ipynb` in Jupyter
2. Run all cells sequentially (Cell → Run All)
3. Review the visualizations and statistics
4. Ensure all cells execute without errors

**Expected outputs:**
- EDA visualizations (price distributions, geospatial plots)
- Engineered features added to the dataset
- Data ready for model training

---

### Step 2: Satellite Image Processing (Optional)

If you want to fetch satellite images yourself (instead of using pre-computed embeddings):

#### Option A: Using Google Colab (Recommended)

1. Upload `data_fetcher.py` to Google Colab
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Set your Google Maps API key:
   ```python
   import os
   os.environ["GOOGLE_MAPS_API_KEY"] = "YOUR_API_KEY_HERE"
   ```
4. Run the data fetcher:
   ```python
   from data_fetcher import fetch_images_from_dataframe
   import pandas as pd
   
   train = pd.read_excel("data/train.xlsx")
   fetch_images_from_dataframe(train, split="train")
   ```

#### Option B: Local Execution

1. Set up Google Maps API key:
   ```bash
   export GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
   ```
   Or add it to your environment variables.

2. Run the fetcher:
   ```python
   python data_fetcher.py
   ```

**Note:** If you have pre-computed embeddings (`train_image_embeddings.parquet` and `test_image_embeddings.parquet`), you can skip this step.

---

### Step 3: Model Training & Evaluation

Open and run the model training notebook:

```bash
jupyter notebook model_training.ipynb
```

**What this notebook does:**
- Loads preprocessed tabular data
- Loads satellite image embeddings (if available)
- Trains three baseline models:
  1. **Location-Only Model** (3 features: lat, long, dist_from_center)
  2. **Rich Tabular Model** (19 engineered features)
  3. **Multimodal Model** (tabular + image embeddings)
- Evaluates models using RMSE and R² metrics
- Compares model performance
- Generates predictions for test set

**How to run:**
1. Open `model_training.ipynb` in Jupyter
2. Ensure data files are in place:
   - `data/train.xlsx`
   - `data/test.xlsx`
   - `train_image_embeddings.parquet` (if using images)
   - `test_image_embeddings.parquet` (if using images)
3. Run all cells sequentially
4. Review model performance metrics

**Expected outputs:**
- Model performance comparison table
- RMSE and R² scores for each model
- Predictions CSV file (`predictions.csv` or `rich_tabular_test_predictions.csv`)

**Model Performance (Expected Results):**
- Location Only: RMSE ~$234,884 | R² ~0.56
- Rich Tabular: RMSE ~$115,729 | R² ~0.89
- Tabular + Images: RMSE ~$118,321 | R² ~0.89

---

## 📈 Evaluation Metrics

Models are evaluated using:

- **RMSE (Root Mean Squared Error)**: Average prediction error in dollars
- **R² Score**: Proportion of variance explained (0-1 scale)
- **Cross-Validated RMSE**: Log-scale RMSE for model comparison

**Interpretation:**
- Lower RMSE = Better predictions
- Higher R² = More variance explained
- R² > 0.8 indicates strong model performance

---

## 📊 Dataset Description

Each row represents a residential property with the following attributes:

| Column Name | Description | Type |
|------------|-------------|------|
| `id` | Unique property identifier | Integer |
| `date` | Date of sale | String |
| `price` | Sale price (target variable) | Integer |
| `bedrooms` | Number of bedrooms | Integer |
| `bathrooms` | Number of bathrooms | Float |
| `sqft_living` | Living area in square feet | Integer |
| `sqft_lot` | Lot size in square feet | Integer |
| `floors` | Number of floors | Float |
| `waterfront` | Waterfront presence (0 = No, 1 = Yes) | Integer |
| `view` | Quality of view (ordinal) | Integer |
| `condition` | Overall condition rating | Integer |
| `grade` | Construction and design grade | Integer |
| `sqft_above` | Square feet above ground | Integer |
| `sqft_basement` | Square feet of basement | Integer |
| `yr_built` | Year the house was built | Integer |
| `yr_renovated` | Year of renovation (0 if none) | Integer |
| `zipcode` | Postal code | Integer |
| `lat` | Latitude | Float |
| `long` | Longitude | Float |
| `sqft_living15` | Living area of nearby homes | Integer |
| `sqft_lot15` | Lot size of nearby homes | Integer |

---

## 🛰️ Satellite Images & Embeddings

### Using Pre-computed Embeddings (Recommended)

If you have `train_image_embeddings.parquet` and `test_image_embeddings.parquet` files, simply place them in the project root directory. The notebooks will automatically load them.

### Fetching Images Yourself

**Requirements:**
- Google Maps Static API key ([Get one here](https://developers.google.com/maps/documentation/maps-static/get-api-key))
- API quota sufficient for your dataset size

**Image Specifications:**
- Resolution: 224×224 pixels
- Map type: Satellite
- Zoom level: 18
- Format: PNG

**Storage:**
- Images are saved as `<id>.png` in `data/images/train/` and `data/images/test/`
- Embeddings are extracted using ResNet18 and saved as Parquet files

---

## 🔁 Reproducibility

To ensure reproducible results:

- **Fixed random seeds**: `random_state=42` used throughout
- **Deterministic splits**: Same train-validation split every run
- **Version control**: Track code changes with Git
- **Environment**: Use virtual environment and `requirements.txt`

**Reproducing results:**
```bash
# Set random seed (already in notebooks)
import numpy as np
import random
np.random.seed(42)
random.seed(42)

# Run notebooks in order
# 1. Preprocessing.ipynb
# 2. model_training.ipynb
```

---

## 🛑 Troubleshooting

### Common Issues and Solutions

#### 1. **ModuleNotFoundError**
```bash
# Solution: Install missing package
pip install <package_name>

# Or reinstall all dependencies
pip install -r requirements.txt
```

#### 2. **FileNotFoundError: data/train.xlsx**
```bash
# Solution: Ensure data files are in the correct location
mkdir -p data
# Place train.xlsx and test.xlsx in data/ directory
```

#### 3. **Google Maps API Errors**
- Verify your API key is correct
- Check API quota/billing status
- Ensure API is enabled for Static Maps service

#### 4. **Memory Errors (Large Datasets)**
- Use Google Colab for large datasets
- Process data in batches
- Reduce image resolution if needed

#### 5. **Jupyter Notebook Not Starting**
```bash
# Install/upgrade Jupyter
pip install --upgrade jupyter notebook

# Start Jupyter
jupyter notebook
```

#### 6. **PyTorch Installation Issues**
Visit [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific installation:
```bash
# Example for CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 7. **Excel File Reading Errors**
```bash
# Install openpyxl for .xlsx files
pip install openpyxl

# Or xlrd for older .xls files
pip install xlrd
```

---

## 📝 Output Files

After running the pipeline, you'll find:

- **`predictions.csv`** or **`rich_tabular_test_predictions.csv`**: Model predictions for test set
- **`Project_Report.pdf`**: Comprehensive PDF report (if generated)
- **`Updated_Report.docx`**: Comprehensive Word report (if generated)
- **Visualizations**: Generated in notebook outputs

---

## 🔮 Future Work

- Fine-tuning CNNs on real estate imagery
- Transformer-based vision models (ViT, CLIP)
- Hyperparameter optimization
- Attention mechanisms for feature fusion
- Deployment-ready inference pipeline
- Explainability analysis (Grad-CAM)

---

## 📚 Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 👤 Author

**Abhishek Goel**  
Machine Learning | Data Science

---


---

## ⚠️ Important Notes

- Ensure property `id` matches between tabular data and satellite images
- Apply log transformation to `price` consistently during training and evaluation
- Avoid committing large binary files (images, embeddings) to GitHub
- Keep your Google Maps API key secure and never commit it to version control
- Use `.env` files or environment variables for sensitive credentials

---

