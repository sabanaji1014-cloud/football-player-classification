# ⚽ Football Image Classification
### PCA + KNN + Logistic Regression

This repository contains a small machine learning project for **binary image classification** of football players from two teams:

- Manchester United
- Chelsea

The project demonstrates a complete machine learning workflow including:

- Image preprocessing
- Dimensionality reduction using **Principal Component Analysis (PCA)**
- Classification using:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Logistic Regression with **L1 (Lasso)** regularization
  - Logistic Regression with **L2 (Ridge)** regularization
- Model comparison and evaluation
- Error analysis

---

# 📂 Dataset

The dataset consists of images of football players.

Each image is processed as follows:

- Converted to **grayscale**
- Resized to **64 × 64**
- Flattened into a feature vector

### Labeling rule

File names determine the class:

- Files starting with **c** → Chelsea
- Other files → Manchester United

### Supported formats

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

---

## 🗂 Project Structure

```text
football-image-classification
│
├── data/        # image dataset
├── docs/        # documentation
├── report/      # report files
├── notebooks/   # optional notebooks
└── src/         # runnable scripts
```

## ⚙️ Requirements

R version **4.0 or higher**

Install required packages in R:

```r
install.packages(c("imager","class","glmnet","rmarkdown","knitr"))
```

---

# ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/sabanaji1014-cloud/football-player-classification.git
cd football-player-classification
```

### 2. Run the analysis

```bash
Rscript src/run_pipeline.R
```

### 3. Render the report (optional)

```bash
R -e "rmarkdown::render('report/football_classifier_report.Rmd')"
```

---

# 🧠 Method Overview

The project follows these steps:

1. Load and preprocess images  
2. Apply PCA for dimensionality reduction  
3. Train multiple classifiers  
4. Evaluate models using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

5. Compare model performance and analyze errors

---

# 👩‍💻 Author

**Saba Naji**

