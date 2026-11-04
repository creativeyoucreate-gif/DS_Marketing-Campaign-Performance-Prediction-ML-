# Multi-Brand Marketing Campaign Performance Analysis and Prediction

A beginner-friendly machine learning project. It uses campaign data from three brands
(**Nykaa, Purplle, Tira**) to:

1. Clean the raw data
2. Explore it (EDA)
3. Predict **Revenue** (regression)
4. Predict **Profit or Loss** (classification)
5. Show the predictions in a **Streamlit** app

---

## 1. Project folder

```
project/
├── app.py                      <- Streamlit app (run this)
├── requirements.txt            <- Python libraries to install
├── README.md                   <- This file
├── data/
│   ├── raw/                    <- nykaa.csv, purplle.csv, tira.csv (original files)
│   └── processed/
│       ├── campaigns_clean.csv        <- made by notebook 01
│       ├── campaigns_features.csv     <- made by notebook 02
│       └── campaigns_model_ready.csv  <- made by notebook 04
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_prep.ipynb
│   ├── 05_regression.ipynb
│   └── 06_classification.ipynb
└── models/
    ├── revenue_model.pkl       <- saved regression model
    ├── profit_model.pkl        <- saved classification model
    ├── model_columns.pkl       <- the 37 input column names, in order
    └── dropdown_options.pkl    <- choices shown in the app dropdowns
```

---

## 2. How to run (Windows + VS Code)

Open a terminal in the project folder, then run these one at a time:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Run the notebooks in order (01 to 06).** Each one saves a file the next one needs.
In VS Code, pick the `venv` kernel, then click **Run All**.

**Start the app** (from the project folder, not from inside `notebooks`):

```
streamlit run app.py
```

A browser tab opens. Fill in the campaign details on the left and click **Predict**.

---

## 3. What each notebook does

| Notebook | What it does | Saves |
|---|---|---|
| 01 Cleaning | Joins the 3 brand files, checks duplicates, fixes dates, checks ROI, fills missing values | `campaigns_clean.csv` |
| 02 Feature engineering | Multi-label encoding of `Channel_Used` (6 channel 0/1 columns + `Channel_Count`), creates `Profit_Flag` | `campaigns_features.csv` |
| 03 EDA | Charts and tables: brands, channels, campaign types, correlations, time trends | (charts only) |
| 04 Model prep | Turns text columns into 0/1 columns, removes `ROI` and other columns that would leak the answer | `campaigns_model_ready.csv`, `model_columns.pkl`, `dropdown_options.pkl` |
| 05 Regression | Trains Linear Regression, Decision Tree, Random Forest to predict Revenue | `revenue_model.pkl` |
| 06 Classification | Trains Logistic Regression, Decision Tree, Random Forest to predict Profit/Loss | `profit_model.pkl` |

---

## 4. Key ideas in simple words

- **Total cost** = Acquisition_Cost x Conversions (Acquisition_Cost is the cost per conversion)
- **ROI** = (Revenue - Total cost) / Total cost
- **Profit_Flag**: 1 if ROI is above 0 (Profit), otherwise 0 (Loss)
- **Multi-label encoding**: one campaign can use several channels, so each channel gets its own 0/1 column
- **Data leakage**: using information the model would not know in real life. `ROI` is calculated from Revenue, so it is never used as an input
- **Train / test split**: 80% of rows teach the model, 20% are kept hidden to test it fairly

---

## 5. Results

Tested on 33,333 rows the models had never seen.

**Regression (predict Revenue)**

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Linear Regression | 237,802 | 157,058 | 0.764 |
| Decision Tree (max_depth=5) | 238,451 | 157,999 | 0.763 |
| Random Forest (saved model) | 237,925 | 157,125 | 0.764 |

**Classification (predict Profit or Loss)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.910 | 0.925 | 0.961 | 0.942 |
| Decision Tree (max_depth=3) | 0.911 | 0.931 | 0.953 | 0.942 |
| Random Forest (saved model) | 0.910 | 0.922 | 0.963 | 0.942 |

- Guessing "Profit" every time gives 0.761 accuracy, so 0.91 is a real improvement.
- **The project brief asks for R2 and accuracy of 0.95. We did not reach it, and that is honest, not a mistake.** The remaining unexplained part is the revenue earned per conversion, and none of the allowed inputs predict it. Reaching 0.95 would need `ROI` or `Revenue` as an input, which is data leakage. We kept the honest scores.
- All three models score almost the same, so we saved the Random Forest. It needs no scaler.

---

## 6. EDA findings

1. About **76%** of campaigns are profitable and **24%** make a loss.
2. The three brands are almost identical (profit share about 0.76 each).
3. Channels, campaign types, segments, languages, months and years show almost no difference, so they are weak predictors.
4. The strongest link with Revenue is **Conversions (0.87)**, then Leads (0.78) and Clicks (0.69).
5. Loss campaigns have a much higher cost per conversion (about 966 vs 190) and far fewer conversions (about 318 vs 1,249).
6. **Recommendation:** lower the cost per conversion and raise conversion volume, instead of switching brand or channel.

---

## 7. Streamlit app

Inputs: brand, campaign type, audience, language, segment, channels, duration, impressions,
clicks, leads, conversions, acquisition cost, engagement score, year, month.

Outputs:
- Predicted Revenue
- Predicted PROFIT or LOSS with the chance of profit
- A bar chart of Total cost vs Predicted revenue

The app builds the same 37 columns the models were trained on, in the same order.

---

## 8. Known limits

- Many rows have `Conversions = 775`. These were filled with the median because the data was missing, so the EDA rankings leave them out.
- Missing values were filled using the formulas when possible, otherwise the median.
- The data has almost no pattern by brand, channel or month, so the app mainly reacts to conversions, leads, clicks and cost.

---

## 9. Libraries

pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, streamlit, joblib, ipykernel
