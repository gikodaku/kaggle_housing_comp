# Kaggle Competitions

This repository contains my Kaggle competition projects, organized by challenge.

## Current Project
- House Prices: Advanced Regression Techniques

## Repository Structure
```text
kaggle-competitions/
├── README.md
├── .gitignore
├── requirements.txt
└── competitions/
    └── house-prices-advanced-regression-techniques/
        ├── README.md
        ├── data/
        │   ├── raw/
        │   └── processed/
        ├── notebooks/
        │   ├── .gitkeep
        │   └── 01_eda.ipynb
        ├── src/
        │   ├── preprocess.py
        │   ├── model.py
        │   └── train.py
        └── submissions/
            └── .gitkeep
```

## Project Highlight
- Competition: House Prices - Advanced Regression Techniques
- Best Score: 0.14340
- Metric: RMSE on the log-transformed target
- Goal: Predict house sale prices using tabular regression models and feature engineering

## Notes
- Raw Kaggle datasets are stored in `competitions/house-prices-advanced-regression-techniques/data/raw/`
- Processed data is saved in `competitions/house-prices-advanced-regression-techniques/data/processed/`
- Final prediction files are saved in `competitions/house-prices-advanced-regression-techniques/submissions/`
- Reusable logic lives in `competitions/house-prices-advanced-regression-techniques/src/`

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- Seaborn
- Plotly
