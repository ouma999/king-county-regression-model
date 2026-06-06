# King County, Washington — Multiple Regression Mass Appraisal Model

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![QGIS](https://img.shields.io/badge/QGIS-3.x-green?logo=qgis&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete end-to-end **Multiple Regression Mass Appraisal Model** built on 21,613 real King County, Washington home sales. This project replicates the statistical modeling workflow used by County Appraiser offices to build Computer Assisted Mass Appraisal (CAMA) value models — from raw data through feature engineering, regression modeling, residual analysis, and QGIS spatial visualization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [Part 1 — Python Analysis (Google Colab)](#part-1--python-analysis-google-colab)
  - [Part 2 — QGIS Spatial Residual Mapping](#part-2--qgis-spatial-residual-mapping)
- [Coefficient Interpretation](#coefficient-interpretation)
- [Residual Color Guide](#residual-color-guide)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Author](#author)

---

## Project Overview

Mass appraisal requires valuing thousands of properties simultaneously using statistical models rather than individual appraisals. This project builds that exact model — a **Multiple Regression Analysis (MRA)** that predicts residential property values from 15 property characteristics including size, quality, age, location, and amenity features.

The workflow mirrors what a **Statistical Data Analyst** at a County Appraiser's office does daily:

1. Load and clean property sales data
2. Engineer features that reflect appraisal theory (depreciation, condition, location)
3. Build a simple regression model (one variable) to establish baseline
4. Build a full multiple regression model (15 variables)
5. Analyze residuals to identify where and why the model errs
6. Map residuals spatially in QGIS to detect neighborhood adjustment needs

---

## Key Results

| Metric | Simple Model (sqft only) | Multiple Regression Model | Improvement |
|---|---|---|---|
| Features Used | 1 | 15 | +14 variables |
| R² | 0.4929 | **0.6947** | +20.2% |
| RMSE | $261,441 | **$214,828** | -$46,613 |
| Price Variation Explained | 49.3% | **69.5%** | Significantly better |

**The model explains 69.5% of all price variation** across 21,613 King County home sales — a strong result for a linear regression model on residential real estate data. The remaining 30.5% reflects factors not captured in the dataset: interior finishes, micro-location nuances, buyer motivation, and property-specific features.

---

## Dataset

**Source:** King County House Sales dataset — Kaggle  
**Link:** `kaggle.com/datasets/harlfoxem/housesalesprediction`  
**Coverage:** 21,613 home sales in King County, Washington (May 2014 – May 2015)

### Original Columns

| Column | Type | Description |
|---|---|---|
| `price` | float | Sale price — dependent variable |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms |
| `sqft_living` | int | Interior square footage |
| `sqft_lot` | int | Lot square footage |
| `floors` | float | Number of floors |
| `waterfront` | int | Waterfront property (0/1) |
| `view` | int | View quality rating (0–4) |
| `condition` | int | Property condition (1–5) |
| `grade` | int | King County building grade (1–13) |
| `sqft_above` | int | Above-ground square footage |
| `sqft_basement` | int | Basement square footage |
| `yr_built` | int | Year built |
| `yr_renovated` | int | Year renovated (0 if never) |
| `zipcode` | int | ZIP code |
| `lat` | float | Latitude coordinate |
| `long` | float | Longitude coordinate |
| `sqft_living15` | int | Avg sqft of 15 nearest neighbors |
| `sqft_lot15` | int | Avg lot sqft of 15 nearest neighbors |

### Engineered Features (Created in Step 3)

| New Column | Formula | Appraisal Rationale |
|---|---|---|
| `age` | `2015 - yr_built` | Age relates directly to physical depreciation |
| `was_renovated` | `1 if yr_renovated > 0` | Renovation premium — adds market value |
| `price_per_sqft` | `price / sqft_living` | Standard appraisal benchmark metric |
| `has_basement` | `1 if sqft_basement > 0` | Basement as additional living space |

---

## Project Structure

```
king-county-regression-model/
│
├── README.md                          ← This file
│
├── data/
│   ├── kc_house_data.csv              ← Original Kaggle dataset (21,613 rows)
│   └── kc_final.csv                   ← GIS-ready residuals export for QGIS
│
├── notebooks/
│   └── king_county_regression.ipynb   ← Full Google Colab notebook
│
└── outputs/
    ├── step4_simple_regression.png    ← Price vs sqft scatter plot
    ├── step6_residuals.png            ← Actual vs predicted + residual distribution
    └── kc_residual_map.png            ← QGIS spatial residual heat map
```

---

## Step-by-Step Workflow

### Part 1 — Python Analysis (Google Colab)

---

#### Step 1 — Load the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("kc_house_data.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))
```

**What this does:** Loads 21,613 real King County home sales. Confirms shape `(21613, 21)` — 21 columns including price, size, quality, location, and coordinates.

---

#### Step 2 — Explore and Understand the Data

```python
print(df.dtypes)
print("Missing values:\n", df.isnull().sum())
print(f"Cheapest home  : ${df['price'].min():,.0f}")
print(f"Most expensive : ${df['price'].max():,.0f}")
print(f"Median price   : ${df['price'].median():,.0f}")
print(f"Avg sqft       : {df['sqft_living'].mean():,.0f}")
print(f"Waterfront homes: {df['waterfront'].sum():,}")
print(df['grade'].value_counts().sort_index())
```

**Key findings:**
- Zero missing values across all 21 columns — no cleaning required
- Price range: $75,000 to $7,700,000
- Median price: $450,000
- Average interior size: 2,080 sqft
- 163 waterfront properties in the dataset

---

#### Step 3 — Feature Engineering

```python
df["age"]           = 2015 - df["yr_built"]
df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
df["price_per_sqft"]= df["price"] / df["sqft_living"]
df["has_basement"]  = (df["sqft_basement"] > 0).astype(int)

print(f"Age range      : {df['age'].min()} to {df['age'].max()} years")
print(f"Ever renovated : {df['was_renovated'].sum():,} homes ({df['was_renovated'].mean()*100:.1f}%)")
print(f"Has basement   : {df['has_basement'].sum():,} homes ({df['has_basement'].mean()*100:.1f}%)")
print(f"Median $/sqft  : ${df['price_per_sqft'].median():,.0f}")
```

**Key findings:**
- Average property age: 44 years
- Only 4.2% of homes ever renovated
- 39.3% have basements
- Median price per sqft: $245

**Why feature engineering matters:** Raw `yr_built = 1955` means nothing to a regression model. Transformed to `age = 60`, it directly maps to physical depreciation theory — older homes lose value at a measurable rate per year.

---

#### Step 4 — Simple Regression (sqft only)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X_simple = df[["sqft_living"]]
y        = df["price"]

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

slope     = model_simple.coef_[0]
intercept = model_simple.intercept_
r2        = r2_score(y, model_simple.predict(X_simple))
rmse      = np.sqrt(mean_squared_error(y, model_simple.predict(X_simple)))

print(f"Formula: Price = ${intercept:,.0f} + ${slope:,.2f} × sqft_living")
print(f"R²   : {r2:.4f} ({r2*100:.1f}% explained)")
print(f"RMSE : ${rmse:,.0f}")
```

**Results:**
- Formula: `Price = -$43,581 + $280.62 × sqft_living`
- R² = 0.4929 — sqft explains 49.3% of price variation
- RMSE = $261,441 — average error of $261K
- Each additional sqft adds $280.62 of value

**Interpretation:** Square footage is the single most important driver of value — but it only explains half the story. The scatter plot shows wide spread above the regression line — proof that grade, location, waterfront, and condition drive the rest.

---

#### Step 5 — Multiple Regression Analysis (15 variables)

```python
from sklearn.model_selection import train_test_split

features = [
    "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "floors",
    "waterfront", "view", "condition", "grade", "age",
    "was_renovated", "has_basement", "sqft_living15", "lat", "long",
]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
r2_test     = r2_score(y_test, y_pred_test)
rmse        = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"R² (Test)  : {r2_test:.4f} ({r2_test*100:.1f}% explained)")
print(f"RMSE (Test): ${rmse:,.0f}")
```

**Results:**
- R² jumped from 0.4929 → **0.6947** (+20.2%)
- RMSE dropped from $261,441 → **$214,828** (-$46,613)
- Training set: 17,290 sales | Test set: 4,323 sales
- No overfitting — train and test R² are identical at 0.6947

---

#### Step 6 — Residual Analysis and QGIS Export

```python
df_test = X_test.copy()
df_test["actual_price"]    = y_test.values
df_test["predicted_price"] = y_pred_test
df_test["residual"]        = df_test["actual_price"] - df_test["predicted_price"]
df_test["residual_pct"]    = (df_test["residual"] / df_test["actual_price"]) * 100

print(f"Within 10% accurate: {(df_test['residual_pct'].abs() <= 10).mean()*100:.1f}%")
print(f"Within 20% accurate: {(df_test['residual_pct'].abs() <= 20).mean()*100:.1f}%")

# Export for QGIS
output = []
for i in range(len(X_test)):
    output.append({
        "lat":       round(float(X_test.iloc[i]["lat"]), 4),
        "long":      round(float(X_test.iloc[i]["long"]), 4),
        "actual":    int(y_test.iloc[i]),
        "predicted": int(y_pred_test[i]),
        "residual":  int(y_test.iloc[i] - y_pred_test[i]),
        "resid_pct": round(float((y_test.iloc[i] - y_pred_test[i]) / y_test.iloc[i] * 100), 1),
        "sqft":      int(X_test.iloc[i]["sqft_living"]),
        "grade":     int(X_test.iloc[i]["grade"]),
        "age":       int(X_test.iloc[i]["age"]),
    })

pd.DataFrame(output).to_csv("kc_final.csv", index=False)
print("✅ GIS file saved: kc_final.csv")
```

**Residual = Actual Sale Price − Model's Predicted Price**
- Positive residual → model UNDER-predicted → property worth more than model said
- Negative residual → model OVER-predicted → property worth less than model said

---

### Part 2 — QGIS Spatial Residual Mapping

After completing the Python analysis, `kc_final.csv` was loaded into QGIS to build a spatial residual heat map of King County, Washington.

---

#### QGIS Step 1 — Open QGIS and Load Base Map

1. Open QGIS
2. On the welcome screen click **OpenStreetMap Basemap** (EPSG:3857)
3. A world map loads as the background layer

---

#### QGIS Step 2 — Add the CSV as a Point Layer

1. Mac menu bar → **Layer → Add Layer → Add Delimited Text Layer...**
2. Click **...** → navigate to `kc_final.csv`
3. Expand **► Record and Fields Options** → tick **Detect field types** ✅
4. Expand **► Geometry Definition** and set:

| Setting | Value |
|---|---|
| X field | `long` |
| Y field | `lat` |
| Geometry CRS | EPSG:4326 — WGS 84 |

5. Click **Add** then **Close**
6. 4,323 point features appear across King County, Washington

> **Important:** Always tick **Detect field types** so QGIS correctly identifies numeric columns. Without this, the Symbology Value dropdown will be empty and graduated coloring won't work.

---

#### QGIS Step 3 — Apply Graduated Color by Residual

1. Mac menu bar → **Layer → Layer Properties**
2. Click **Symbology** in the left sidebar
3. Change top dropdown from **Single Symbol** → **Graduated**
4. Set **Value** = `resid_pct`
5. Set **Color ramp** = **RdYlGn** → click **Invert Color Ramp**
6. Set **Mode** = **Equal Count (Quantile)**
7. Set **Classes** = **7**
8. Click **Classify**
9. Click **OK**

---

#### QGIS Step 4 — Zoom to King County

Right-click layer → **Zoom to Layer** → QGIS zooms to the Seattle/King County area

---

#### QGIS Step 5 — Inspect Individual Properties

Click the **Identify Features** tool → click any dot → panel shows:
- Actual sale price
- Model's predicted price
- Residual dollar amount
- Residual percentage
- Square footage, grade, age

---

## Coefficient Interpretation

The model learned these dollar relationships from the data:

| Feature | Coefficient | Meaning |
|---|---|---|
| `waterfront` | +$566,789 | Waterfront adds over half a million dollars |
| `lat` | +$557,545 | Moving north in KC adds premium value |
| `long` | -$106,325 | East-west location gradient |
| `grade` | +$97,317 | Each building grade jump adds $97K |
| `view` | +$49,329 | Each view level adds $49K |
| `bathrooms` | +$46,645 | Each bathroom adds $47K |
| `was_renovated` | +$42,029 | Renovation adds $42K |
| `bedrooms` | -$32,649 | More bedrooms = smaller rooms = less value (multicollinearity) |
| `condition` | +$27,665 | Each condition level adds $28K |
| `has_basement` | -$23,617 | Already captured in sqft — multicollinearity effect |
| `sqft_living` | +$170 | Each sqft adds $170 (adjusted for other variables) |
| `age` | +$2,552 | Minor positive — older homes have character premium |

---

## Residual Color Guide

| Color | Residual % | Meaning |
|---|---|---|
| 🟢 Dark Green | > +20% | Model severely UNDER-predicted — hidden value factors |
| 🟩 Light Green | +10% to +20% | Model moderately under-predicted |
| 🟡 Yellow-Green | 0% to +10% | Slightly under-predicted |
| 🟡 Yellow | -5% to +5% | ACCURATE — model nailed it |
| 🟠 Orange | -10% to -20% | Model moderately over-predicted |
| 🔴 Light Red | -20% to -30% | Model significantly over-predicted |
| 🔴 Dark Red | < -30% | Model severely over-predicted |

**Spatial patterns on the map:**
- Red clusters → neighborhood needs a negative adjustment factor in the model
- Green clusters → neighborhood commands a premium the model misses
- Random mixed colors → model errors are unbiased — no systematic geographic error

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core analysis language |
| pandas | Data loading, cleaning, manipulation |
| numpy | Numerical calculations |
| scikit-learn | LinearRegression, train_test_split, R², RMSE |
| matplotlib | Charts — scatter plots, residual distribution |
| Google Colab | Cloud notebook environment |
| QGIS 3.x | GIS spatial residual heat map |
| Kaggle | Source of real King County sales data |

---

## How to Run

### Option A — Google Colab (Recommended)

1. Go to `kaggle.com/datasets/harlfoxem/housesalesprediction`
2. Download and unzip → get `kc_house_data.csv`
3. Upload to Colab using the Files panel
4. Open `notebooks/king_county_regression.ipynb`
5. Run all cells in order (Steps 1–6)
6. Download `kc_final.csv` for QGIS

### Option B — Local Python

```bash
git clone https://github.com/ouma999/king-county-regression-model.git
cd king-county-regression-model
pip install pandas numpy scikit-learn matplotlib
python notebooks/king_county_regression.py
```

### Load Residuals into QGIS

```
1. Open QGIS → OpenStreetMap Basemap
2. Layer → Add Layer → Add Delimited Text Layer
3. Select: data/kc_final.csv
4. Record and Fields Options → tick Detect field types ✅
5. X field = long  |  Y field = lat  |  CRS = EPSG:4326
6. Add → Close
7. Layer Properties → Symbology → Graduated
8. Value = resid_pct → RdYlGn (inverted) → Quantile → 7 classes → Classify → OK
```

---

## Resume Description

> **King County, WA — Multiple Regression Mass Appraisal Model** | Python · scikit-learn · QGIS · pandas
> Built a 15-variable multiple regression model on 21,613 real King County Washington home sales achieving R² of 0.6947 — a 20% improvement over a single-variable baseline. Engineered appraisal-specific features including age-based depreciation, renovation premium, and location gradients. Exported model residuals to QGIS and produced a spatial heat map revealing geographic patterns of model over and under-prediction across the Seattle metro area — directly replicating the CAMA model validation workflow used by County Appraiser offices.

---

## Author

**Polex Ouma**  
Data Engineer & Analytics Professional  
Helix Fellow | Statistical Data Analyst Candidate

- GitHub: [github.com/ouma999](https://github.com/ouma999)
- Skills: Python · SQL · scikit-learn · QGIS · Mass Appraisal Analytics · pandas · Airflow

---

*Dataset: King County House Sales — Kaggle (harlfoxem). Real residential sales May 2014 – May 2015. Analysis methodology follows IAAO Standard on Mass Appraisal of Real Property.*
