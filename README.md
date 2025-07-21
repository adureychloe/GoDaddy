# GoDaddy - Microbusiness Density Forecasting

This project is a solution for the Kaggle competition "GoDaddy - Microbusiness Density Forecasting". It includes feature engineering, model stacking, prediction, and submission generation. The original notebook has been refactored into reproducible Python scripts.

## Competition Background

The goal of this competition is to predict monthly microbusiness density in a given area using U.S. county-level data. Accurate models will help policymakers gain visibility into microbusinesses, enabling new policies and programs to improve the success and impact of these smallest of businesses.

Submissions are evaluated on SMAPE (Symmetric Mean Absolute Percentage Error) between forecasts and actual values.

## Project Structure

```
GoDaddy-Microbusiness-Forecasting/
│
├── data/                # Data folder (download competition data here)
│   └── README.md        # Data download instructions
│
├── src/                 # Source code
│   ├── main.py          # Main pipeline script
│   └── utils.py         # Utility functions
│
├── godaddy-advance.ipynb # Original notebook
│
├── requirements.txt     # Dependencies
│
└── README.md            # Project documentation
```

## Data Preparation

Download the following files from the [Kaggle competition page](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/data) and place them in the `data/` directory:

- train.csv
- test.csv
- census_starter.csv
- revealed_test.csv
- sample_submission.csv
- usa-counties-coordinates/cfips_location.csv
- us-indicator/co-est2021-alldata.csv
- census-data-for-godaddy/ACSST5Y2020.S0101-Data.csv
- census-data-for-godaddy/ACSST5Y2021.S0101-Data.csv

## Dependencies

See `requirements.txt`. Main dependencies:
- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- catboost
- matplotlib
- seaborn
- tqdm
- plotly

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure data is placed in the `data/` directory.
2. Run the main script:
```bash
python src/main.py
```
3. The result will be saved as `submission.csv` in the project root.

## Main Pipeline

1. **Data Loading & Preprocessing**  
   - Merge train, test, and auxiliary data. Handle missing values and outliers.
2. **Feature Engineering**  
   - Create time-series, demographic, and geographic features (e.g., rotated coordinates).
3. **Model Training & Stacking**  
   - Use XGBoost, LightGBM, CatBoost, and stacking for ensemble prediction.
4. **Prediction & Postprocessing**  
   - Generate predictions and output submission file in the required format.
5. **Evaluation**  
   - Use SMAPE as the main evaluation metric.

## Code Overview

- `src/main.py`: Main pipeline with detailed English comments.
- `src/utils.py`: Utility functions such as SMAPE.
- `godaddy-advance.ipynb`: Original notebook for reference.

## Acknowledgements

This project is inspired by top Kaggle solutions and the Kaggle community. 