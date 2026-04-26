# Heart Disease Prediction

This project predicts whether a patient is likely to have heart disease using machine learning. It follows an end-to-end workflow from data understanding, cleaning, and feature preparation to model training, threshold tuning, prediction, and deployment with Flask.

The project is built for educational decision support, not medical diagnosis. In this workflow, missing a patient who may have heart disease is treated as more harmful than sending a healthy patient for additional follow-up, so recall is a key priority during evaluation.

## Problem Statement

- Target: `target`
- Class `0`: No disease
- Class `1`: Disease
- Problem type: Binary classification
- Goal: Support early screening and reduce the risk of missed heart disease cases

The original dataset uses `num` as the diagnosis column. In this project:

- `num = 0` becomes `target = 0`
- `num > 0` becomes `target = 1`

## Why Recall Matters

In healthcare prediction, a false negative means the model predicts a patient is healthy when the patient may actually have heart disease. That is the most dangerous type of error because it may delay follow-up testing or treatment.

For that reason, this project does not rely on accuracy alone. It compares models and tunes classification thresholds using **F2-score**, which gives more weight to recall than precision.

## Dataset

- Source file: `data/raw/heart_disease_uci.csv`
- Raw shape: `920 rows x 16 columns`
- Cleaned shape: `920 rows x 14 columns`
- Final target distribution:
  - `0`: 411 patients
  - `1`: 509 patients

### Main Features Used

- `age`: age in years
- `sex`: patient sex
- `dataset`: source hospital group
- `cp`: chest pain type
- `trestbps`: resting blood pressure
- `chol`: serum cholesterol
- `fbs`: fasting blood sugar above 120 mg/dL
- `restecg`: resting ECG result
- `thalch`: maximum heart rate achieved
- `exang`: exercise-induced angina
- `oldpeak`: ST depression induced by exercise

## Project Workflow

### 1. Data Understanding and EDA

EDA is documented in:

- `notebooks/01_eda.ipynb`

The notebook explores dataset structure, class balance, feature distributions, and general data understanding before modeling.

### 2. Data Cleaning

Cleaning logic is implemented in:

- `src/data/clean_data_processing.py`

Main cleaning decisions:

- Validate required raw-data columns
- Drop columns with more than `30%` missing values
- Standardize text values in categorical columns
- Create a binary `target` from `num`
- Remove duplicate rows
- Replace invalid zero values in `trestbps`, `chol`, and `oldpeak` with missing values before imputation
- Impute numeric values with median
- Impute categorical values with mode
- Save cleaned output to `data/processed/heart_disease_clean.csv`

Why this matters:

- It keeps the cleaning pipeline reproducible
- It reduces noise from invalid or inconsistent raw values
- It prepares the data in a form that can be reused for modeling and deployment

### 3. Feature Preparation

Feature preparation is implemented in:

- `src/features/build_features.py`

Main feature decisions:

- Drop `id` because it is an identifier, not a predictive feature
- Drop `num` because `target` already stores the binary outcome
- Scale numeric variables with `StandardScaler`
- Encode `dataset` using one-hot encoding
- Encode binary variables with ordinal encoding
- Map ordinal clinical variables such as `cp` and `restecg` into ordered numeric values

Saved artifacts:

- `data/processed/heart_disease_feature_matrix.csv`
- `data/processed/heart_disease_target.csv`
- `models/heart_disease_feature_preparation.pkl`
- `models/heart_disease_feature_metadata.json`

### 4. Modeling

Training logic is implemented in:

- `src/models/heart_disease_train_model.py`

Models compared:

- Logistic Regression
- Random Forest
- XGBoost

Train/test strategy:

- `80/20` split
- `stratify=y`
- `random_state=42`

The training script evaluates thresholds from `0.10` to `0.90` in `0.01` steps and selects the best model-threshold pair using:

1. Highest F2-score
2. Higher recall
3. Higher F1-score
4. Higher precision

This makes the final decision rule more aligned with the screening goal of reducing missed disease cases.

## Final Model Results

According to `models/heart_disease_model_results.json`, the selected deployment setup is:

- Best model: `Logistic Regression`
- Selected threshold: `0.21`

### Final Threshold-Tuned Performance

- Accuracy: `79.89%`
- Precision: `74.07%`
- Recall: `98.04%`
- F1-score: `84.39%`
- F2-score: `92.08%`
- ROC-AUC: `90.61%`

Confusion matrix at the selected threshold:

- True Negatives: `47`
- False Positives: `35`
- False Negatives: `2`
- True Positives: `100`

### Medical Interpretation

This result is strong for screening because the model only missed `2` disease cases in the test split. Its main strength is very high recall.

The tradeoff is a higher number of false positives (`35`). In a real clinical workflow, that may still be acceptable for screening because those patients can be referred for further examination instead of being incorrectly cleared as healthy.

## Prediction Module

Prediction logic is implemented in:

- `src/models/heart_disease_predict_model.py`

It:

- loads the saved model from `models/heart_disease_model.pkl`
- loads the saved threshold from `models/heart_disease_model_results.json`
- validates and normalizes new patient inputs
- returns probability, predicted class, risk label, and an educational-use warning

Example output fields:

- `prediction`
- `label`
- `probability`
- `threshold`
- `risk_level`
- `note`

## Flask App

The demo application is implemented in:

- `app/app.py`

The app:

- accepts patient input through a web form
- runs the trained model on new patient data
- displays prediction, probability, and risk level
- shows saved model recall and precision from the training results

Run the app and open `http://localhost:5000`.

## Project Structure

```text
Heart disease project/
|-- app/
|   |-- app.py
|   |-- static/
|   |   |-- style.css
|   |-- templates/
|       |-- index.html
|-- data/
|   |-- raw/
|   |   |-- heart_disease_uci.csv
|   |-- processed/
|       |-- heart_disease_clean.csv
|       |-- heart_disease_feature_matrix.csv
|       |-- heart_disease_target.csv
|-- models/
|   |-- heart_disease_model.pkl
|   |-- heart_disease_feature_preparation.pkl
|   |-- heart_disease_feature_metadata.json
|   |-- heart_disease_model_results.json
|-- notebooks/
|   |-- 01_eda.ipynb
|-- reports/
|-- src/
|   |-- data/
|   |   |-- clean_data_processing.py
|   |-- features/
|   |   |-- build_features.py
|   |-- models/
|   |   |-- heart_disease_predict_model.py
|   |   |-- heart_disease_train_model.py
|-- README.md
|-- requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

For local training and notebook work, install the full development stack instead:

```bash
pip install -r requirements-dev.txt
```

## How To Run

### 1. Clean the raw data

```bash
python src/data/clean_data_processing.py
```

### 2. Prepare features

```bash
python src/features/build_features.py
```

### 3. Train and save the model

```bash
python src/models/heart_disease_train_model.py
```

This saves:

- `models/heart_disease_model.pkl`
- `models/heart_disease_model_results.json`

### 4. Test a sample prediction

```bash
python src/models/heart_disease_predict_model.py
```

### 5. Run the Flask app

```bash
python app/app.py
```

Then visit:

```text
http://localhost:5000
```

## Dependencies

Runtime:

- flask
- numpy
- pandas
- scikit-learn

Development and training:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter
- xgboost

## Key Learnings

- Medical classification should not be judged by accuracy alone
- Threshold tuning can be as important as model choice
- Reproducible cleaning and feature pipelines make deployment easier
- High recall is valuable for screening, even when it increases false positives
- Saving preprocessing and model artifacts helps keep prediction behavior consistent

## Important Limitation

This project is an educational machine learning application built on structured tabular data. It should not be used for real clinical diagnosis without medical validation, external testing, fairness checks, and review by qualified healthcare professionals.
