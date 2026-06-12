# King County, Washington — Multiple Regression Mass Appraisal Model

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![QGIS](https://img.shields.io/badge/QGIS-3.x-green?logo=qgis&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete end-to-end **Multiple Regression Mass Appraisal Model** built on 21,613 real King County, Washington residential home sales, aligned with IAAO standards for mass appraisal methodology.

---

## Project Overview

This project demonstrates the full workflow of a professional mass appraisal model — from raw sales data through statistical modeling, diagnostic testing, and spatial residual analysis using GIS. It mirrors the methodology used by county assessors and CAMA system analysts in production environments.

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | King County, WA House Sales Dataset |
| Records | 21,613 residential transactions |
| Features | 15 property characteristics |
| Target Variable | Sale Price (USD) |

**Key variables:** sqft_living, bedrooms, bathrooms, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long

---

## Methodology

1. **Data Preprocessing** — handled missing values, removed outliers, engineered features
2. **Exploratory Data Analysis** — correlation analysis, distribution plots, price heatmaps
3. **Baseline Model** — simple linear regression on sqft_living for benchmark comparison
4. **Multiple Regression Model** — OLS regression with 15 variables using scikit-learn
5. **Model Diagnostics** — residual plots, Q-Q plots, heteroscedasticity checks
6. **Spatial Residual Mapping** — residuals exported and visualized in QGIS to identify geographic bias patterns

---

## Results

| Metric | Baseline Model | Multiple Regression Model |
|---|---|---|
| R² | 0.4929 | **0.6947** |
| RMSE | $261,441 | **$214,828** |

The multiple regression model improved R² by **+20.1 percentage points** and reduced prediction error by **$46,613 per property** compared to the single-variable baseline.

---

## Spatial Analysis (QGIS)

Residuals were mapped across King County to identify spatial patterns in model performance:

- **Red clusters** — model over-predicted (sold for less than expected) — concentrated in older South Seattle and Federal Way neighborhoods
- **Green clusters** — model under-predicted (sold for more than expected) — waterfront and premium micro-locations
- **Eastside / Bellevue** — high variability consistent with luxury market heterogeneity

This spatial diagnostic mirrors the locational adjustment factor workflow used in professional CAMA model validation.

---

## IAAO Alignment

This project follows key principles from the **International Association of Assessing Officers (IAAO) Standard on Mass Appraisal**:

- Sales ratio analysis for assessment equity evaluation
- Coefficient of Dispersion (COD) and Price-Related Differential (PRD) as equity metrics
- Residual analysis for model bias detection
- Geographic stratification for locational adjustments

---

## Technologies

| Tool | Purpose |
|---|---|
| Python (pandas, numpy) | Data wrangling and preprocessing |
| scikit-learn | OLS regression modeling |
| matplotlib, seaborn | Visualization and diagnostics |
| QGIS 3.x | Spatial residual heat mapping |
| Google Colab | Development environment |

---

## Repository Structure

```
king-county-regression-model/
├── kc_house_data.csv          # Raw dataset
├── kc_final.csv               # Cleaned dataset
├── regression_model.ipynb     # Full modeling notebook
├── step4_simple_regression.png
├── step6_residuals.png
└── README.md
```

---

## Author

**Polex Ouma Otieno**
M.S. Business Analytics — University of South Dakota
[GitHub](https://github.com/ouma999) | [LinkedIn](https://linkedin.com/in/ouma-pius6822023a)
