
Football Image Classification (PCA + KNN + Logistic Regression)

This repository contains a small machine learning project for binary image classification of football players from two teams:

Manchester United

Chelsea

The project demonstrates a complete machine learning workflow including:

Image preprocessing

Dimensionality reduction using Principal Component Analysis (PCA)

Classification using:

K-Nearest Neighbors (KNN)

Logistic Regression

Logistic Regression with L1 (Lasso) regularization

Logistic Regression with L2 (Ridge) regularization

Model comparison and evaluation

Error analysis

Dataset

The dataset consists of images of football players.

Images are processed as follows:

Converted to grayscale

Resized to 64 × 64

Flattened into feature vectors

Labeling rule

File names determine the class:

Files starting with c → Chelsea

Other files → Manchester United

Supported formats:

jpg

jpeg

png

bmp

Project Structure
football-image-classification
│
├── data/        # image dataset
├── docs/        # documentation
├── report/      # report files
├── notebooks/   # optional notebooks
└── src/         # runnable scripts
Requirements

R version 4.0 or higher

Required packages:

imager
class
glmnet
rmarkdown
knitr

Install them in R:

install.packages(c("imager","class","glmnet","rmarkdown","knitr"))
How to Run the Project

Clone the repository:

git clone https://github.com/sabanaji1014-cloud/football-player-classification.git
cd football-player-classification

Run the analysis:

Rscript src/run_pipeline.R

Or render the report:

R -e "rmarkdown::render('report/football_classifier_report.Rmd')"
Method Overview

Load and preprocess images

Apply PCA to reduce dimensionality

Train multiple classifiers

Evaluate using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Compare model performance and analyze errors

Author

Saba Naji

Statistical Machine Learning Project