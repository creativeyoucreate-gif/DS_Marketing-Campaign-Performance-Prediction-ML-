# Multi-Brand Marketing Campaign Performance Analysis and Prediction

An end-to-end machine learning project that cleans campaign data from three beauty brands (Nykaa, Purplle, Tira), analyses what drives profit, predicts campaign **Revenue** (regression) and **Profit / Loss** (classification), and serves the predictions in a Streamlit app.

## Dataset

- 3 raw CSV files (`nykaa.csv`, `purplle.csv`, `tira.csv`), combined into 166,665 campaigns.
- Columns include campaign type, target audience, duration, channels used, impressions, clicks, leads, conversions, revenue, acquisition cost (cost per conversion), ROI, language, engagement score, customer segment and date.

## Project structure

```
marketing_campaign/
├── data/
│   ├── raw/                  original brand CSV files (never edited)
│   └── processed/            cleaned, feature-engineered and model-ready CSVs
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_prep.ipynb
│   ├── 05_regression.ipynb
│   └── 06_classification.ipynb
├── models/                   saved models and column lists used by the app
├── app.py                    Streamlit app
├── requirements.txt
└── README.md
```

## Workflow

1. **Cleaning** - combine the three files and add a Brand column. Fill missing values using the ROI formula where possible, otherwise the median (numbers) or "Unknown" (text). Create Year and Month from the date.
2. **Feature engineering** - multi-label encode `Channel_Used` into 6 channel columns plus `Channel_Count`. Create `Profit_Flag` (1 if ROI > 0, else 0).
3. **EDA** - brand, channel, campaign type, audience and time analysis, top and low campaigns, correlations.
4. **Model preparation** - drop ID and label columns, encode text columns, and keep Revenue, ROI and Profit_Flag out of the inputs to avoid data leakage.
5. **Regression** - predict Revenue with Linear Regression, Decision Tree and Random Forest.
6. **Classification** - predict Profit / Loss with Logistic Regression, Decision Tree and Random Forest.
7. **App** - Streamlit app that takes campaign details and shows the predicted Revenue and Profit / Loss.

## Results

| Task | Best score | Metrics checked |
|---|---|---|
| Revenue prediction | R² about 0.76 | MSE, RMSE, MAE, R² |
| Profit / Loss prediction | Accuracy about 0.91, F1 about 0.94 | Accuracy, Precision, Recall, F1 |

All three models in each task score almost the same, and the Random Forest is saved for the app. Always guessing "Profit" gives 0.76 accuracy, so 0.91 is a real improvement.

The project brief suggests a target of 0.95. It was not reached because the remaining variation is the revenue earned per conversion, which none of the allowed input columns predict. Reaching 0.95 would need ROI or Revenue as an input, which is data leakage, so the real scores are reported.

## Key insights

- About 76% of campaigns are profitable and 24% make a loss.
- Brand, channel, campaign type, customer segment, language and month have almost no effect on profitability (profit share is about 0.76 in every group).
- Revenue is most closely linked to Conversions (0.87), then Leads (0.78) and Clicks (0.69).
- Loss campaigns have a much higher cost per conversion (about 966 vs 190) and far fewer conversions (about 318 vs 1,249).
- Recommendation: focus on lowering the cost per conversion and increasing conversion volume.

## How to run

```bash
# 1. Create and activate a virtual environment, then install the libraries
pip install -r requirements.txt

# 2. Run the notebooks in order (01 to 06) to create the data files and models

# 3. Start the app from the project folder
streamlit run app.py
```

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit
