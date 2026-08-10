# AML Risk Scoring with XGBoost

An end-to-end machine learning learning project that demonstrates how to move from raw banking data to a trained, evaluated, explainable, and reusable classification pipeline.

> **Important:** This project currently uses the UCI Bank Marketing Dataset. The target variable `y` represents whether a customer subscribed to a term deposit. It is **not** a real AML risk label. The project is designed to teach the machine learning workflow that can later be adapted to real AML risk-scoring data.

---

## Project Goal

The purpose of Chapters 1–3 is to build the complete machine learning development flow:

```text
Raw Banking Data
        ↓
Load Data
        ↓
Explore and Understand Data
        ↓
Create Features and Target
        ↓
Remove Leakage
        ↓
Train / Test Split
        ↓
Identify Numeric and Categorical Features
        ↓
Build Preprocessing
        ↓
Build Machine Learning Pipeline
        ↓
Train Logistic Regression Baseline
        ↓
Train XGBoost
        ↓
Cross Validation
        ↓
Evaluate Models
        ↓
Compare Models
        ↓
Tune Decision Threshold
        ↓
Feature Importance
        ↓
SHAP Explainability
        ↓
Save Complete Pipeline
        ↓
Load Pipeline
        ↓
Predict New Records
```

The main design principle is:

```text
Machine Learning predicts.
Probability becomes a business decision through a threshold.
Explainability helps us understand the model.
The complete Pipeline is saved for future use.
```

---

# 1. Project Structure

After Chapters 1–3, the project contains:

```text
AML-Risk-Scoring-with-XGBoost/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── bank-full.csv
│   └── processed/
│
├── notebooks/
│   ├── Day01_Data_Inspection.ipynb
│   ├── Day02_EDA.ipynb
│   ├── Day03_Train_Test_Split.ipynb
│   ├── Day04_Preprocessing.ipynb
│   ├── Day05_Baseline_Model.ipynb
│   ├── Day06_XGBoost_Fundamentals.ipynb
│   ├── Day07_Train_XGBoost.ipynb
│   ├── Day08_Model_Validation.ipynb
│   └── Day09_Explainability_and_Threshold.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluate.py
│   ├── train_baseline.py
│   ├── train_xgboost.py
│   ├── threshold_tuning.py
│   ├── explain_model.py
│   └── predict.py
│
├── models/
│   ├── baseline_pipeline.pkl
│   └── xgboost_pipeline.pkl
│
└── reports/
    ├── baseline_metrics.json
    ├── xgboost_metrics.json
    ├── model_comparison.csv
    ├── threshold_results.csv
    ├── baseline_confusion_matrix.png
    ├── xgboost_confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.csv
    ├── feature_importance.png
    ├── shap_summary.png
    └── sample_predictions.csv
```

---

# 2. Dataset

This project uses the UCI Bank Marketing Dataset.

Expected file:

```text
data/raw/bank-full.csv
```

The CSV uses a semicolon delimiter:

```python
pd.read_csv(
    DATA_PATH,
    sep=";"
)
```

The original dataset contains approximately:

```text
45,211 rows
17 columns
```

Main fields:

```text
age
job
marital
education
default
balance
housing
loan
contact
day
month
duration
campaign
pdays
previous
poutcome
y
```

Target mapping:

```text
no  → 0
yes → 1
```

---

# 3. Step 1 — Load the Data

The first step is to load the raw CSV into a Pandas DataFrame.

```python
import pandas as pd

df = pd.read_csv(
    "data/raw/bank-full.csv",
    sep=";"
)
```

In the formal project structure, data loading is moved into:

```text
src/data_loader.py
```

and called with:

```python
dataframe = load_bank_data()
```

The loader also validates whether the file exists, whether the dataset is empty, whether the target column exists, and whether duplicated column names are present.

```text
Raw CSV
    ↓
Validated Pandas DataFrame
```

---

# 4. Step 2 — Explore and Understand the Data

Before training a model, inspect the dataset.

```python
df.head()
df.tail()
df.shape
df.columns
df.info()
df.dtypes
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.nunique()
df["y"].value_counts()
```

This stage is called **EDA — Exploratory Data Analysis**.

The purpose is to understand:

```text
How many rows exist?
How many columns exist?
Which fields are numeric?
Which fields are categorical?
Are values missing?
Are records duplicated?
Is the target imbalanced?
Could any feature create Data Leakage?
```

---

# 5. Step 3 — Create Features and Target

Machine learning requires two primary objects:

```text
X = Features
y = Target
```

The target is created using:

```python
TARGET_MAPPING = {
    "no": 0,
    "yes": 1
}

target = (
    dataframe[
        TARGET_COLUMN
    ]
    .map(
        TARGET_MAPPING
    )
)
```

The model inputs are created separately:

```python
features = dataframe.drop(
    columns=columns_to_drop
)
```

Conceptually:

```text
Original DataFrame
        ↓
   ┌─────────────┐
   │             │
   X             y
Features       Target
```

A **Feature** is an input used by the model. The **Target** is the outcome that the model learns to predict.

---

# 6. Step 4 — Remove Data Leakage

This project removes:

```python
LEAKAGE_COLUMNS = [
    "duration"
]
```

Therefore:

```python
columns_to_drop = [
    TARGET_COLUMN,
    *LEAKAGE_COLUMNS
]
```

A model should only use information that would legitimately be available when the prediction is made.

> The model must not learn from information that would only become available after the outcome or decision point.

This becomes especially important when adapting the workflow to a real AML system.

---

# 7. Step 5 — Train / Test Split

The dataset is divided into training and test data:

```python
X_train, X_test, y_train, y_test = (
    train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target
    )
)
```

Conceptually:

```text
                  Full Dataset
                       │
                train_test_split
                  /          \
                 /            \
           Training Data      Test Data
               80%               20%
             /     \           /     \
        X_train  y_train   X_test   y_test
```

- `X_train`: features used to train the model.
- `y_train`: correct answers used during training.
- `X_test`: features not used to train the model.
- `y_test`: correct answers used for final evaluation.

Important parameters:

```python
test_size=0.20
random_state=42
stratify=target
```

---

# 8. Step 6 — Identify Numeric and Categorical Features

The dataset contains different feature types.

Numeric examples:

```text
age
balance
day
campaign
pdays
previous
```

Categorical examples:

```text
job
marital
education
default
housing
loan
contact
month
poutcome
```

Numeric fields:

```python
numeric_features = (
    X_train
    .select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)
```

Categorical fields:

```python
categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)
```

This matters because numeric and categorical fields require different preprocessing.

---

# 9. Step 7 — Build Numeric Preprocessing

Numeric features use:

```python
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
```

Flow:

```text
Numeric Features
      ↓
SimpleImputer
      ↓
StandardScaler
      ↓
Processed Numeric Features
```

---

# 10. Step 8 — Build Categorical Preprocessing

Categorical features use:

```python
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
```

Flow:

```text
Categorical Features
        ↓
SimpleImputer
        ↓
OneHotEncoder
        ↓
Numeric Representation
```

`handle_unknown="ignore"` allows future prediction inputs to contain categories that were not seen during model training.

---

# 11. Step 9 — Combine Preprocessing

Numeric and categorical preprocessing are combined with:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)
```

Architecture:

```text
                       Raw Features
                            │
                   ColumnTransformer
                    /              \
                   /                \
          Numeric Features     Categorical Features
                 │                    │
             Imputer              Imputer
                 │                    │
              Scaler            OneHotEncoder
                  \                  /
                   \                /
                    └──────┬───────┘
                           ↓
                  Processed Features
```

---

# 12. Step 10 — Build the Machine Learning Pipeline

Preprocessing and the classifier are combined into one Pipeline:

```python
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)
```

Architecture:

```text
Raw Input
    ↓
Preprocessor
    ↓
Processed Features
    ↓
Classifier
    ↓
Prediction
```

The model and preprocessing remain connected during training, cross validation, testing, model saving, and future prediction.

---

# 13. Step 11 — Train the Logistic Regression Baseline

The first model is a Logistic Regression baseline:

```python
baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

Training:

```python
baseline_pipeline.fit(
    X_train,
    y_train
)
```

This single call performs:

```text
Fit Numeric Imputer
        ↓
Fit StandardScaler
        ↓
Fit Categorical Imputer
        ↓
Fit OneHotEncoder
        ↓
Transform Training Data
        ↓
Train Logistic Regression
```

The baseline provides a reference point before introducing a more complex model.

---

# 14. Step 12 — Build the XGBoost Model

Chapter 3 introduces XGBoost.

```python
xgboost_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.80,
    colsample_bytree=0.80,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
```

Important parameters include:

```text
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
scale_pos_weight
```

The model is inserted into the same Pipeline design:

```python
xgboost_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            xgboost_model
        )
    ]
)
```

Changing the classifier does not require rebuilding the entire application architecture.

---

# 15. Step 13 — Handle Class Imbalance

Class counts are calculated using:

```python
negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()
```

Then:

```python
scale_pos_weight = (
    negative_count
    / positive_count
)
```

The value is passed to XGBoost:

```python
XGBClassifier(
    scale_pos_weight=scale_pos_weight
)
```

This makes the model pay more attention to the minority positive class.

---

# 16. Step 14 — Cross Validation

The project uses:

```python
cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Then:

```python
cross_validation_results = cross_validate(
    estimator=xgboost_pipeline,
    X=X_train,
    y=y_train,
    cv=cross_validation,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)
```

Important rule:

```text
Cross Validation uses only X_train and y_train.
```

The final Test Set remains untouched until final model evaluation.

Cross Validation helps evaluate model stability, generalization, and possible overfitting.

---

# 17. Step 15 — Train the Final Pipeline

The final Pipeline is trained using:

```python
xgboost_pipeline.fit(
    X_train,
    y_train
)
```

After this step, the fitted object contains all preprocessing components and the fitted XGBoost classifier.

---

# 18. Step 16 — Generate Predictions

Class prediction:

```python
y_pred = xgboost_pipeline.predict(
    X_test
)
```

Probability prediction:

```python
y_probability = (
    xgboost_pipeline
    .predict_proba(
        X_test
    )[:, 1]
)
```

The model fundamentally produces a positive-class probability.

```text
Customer A → 0.12
Customer B → 0.46
Customer C → 0.88
```

---

# 19. Step 17 — Evaluate the Model

The project evaluates:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Average Precision
Confusion Matrix
ROC Curve
Precision-Recall Curve
```

Example:

```python
precision = precision_score(
    y_test,
    y_pred
)
```

```python
recall = recall_score(
    y_test,
    y_pred
)
```

```python
roc_auc = roc_auc_score(
    y_test,
    y_probability
)
```

Important distinction:

```text
Class Predictions → Accuracy / Precision / Recall / F1 / Confusion Matrix
Probability        → ROC-AUC / Average Precision / ROC Curve / PR Curve
```

The project does not rely only on Accuracy because the target is imbalanced.

---

# 20. Step 18 — Compare Logistic Regression and XGBoost

Both models are evaluated using the same Test Set.

```text
                Same X_test / y_test
                       │
              ┌────────┴─────────┐
              ↓                  ↓
      Logistic Regression      XGBoost
              │                  │
              └────────┬─────────┘
                       ↓
                Compare Metrics
```

Comparison includes Accuracy, Precision, Recall, F1, ROC-AUC, False Positives, and False Negatives.

The goal is to objectively determine whether the more complex model provides meaningful improvement.

---

# 21. Step 19 — Tune the Decision Threshold

A classifier may use a default threshold around `0.50`.

```text
Probability >= 0.50 → Class 1
Probability <  0.50 → Class 0
```

The same behavior can be created manually:

```python
prediction = (
    y_probability
    >= threshold
).astype(int)
```

The project evaluates multiple thresholds and compares Precision, Recall, F1, False Positives, False Negatives, and Predicted Positive Count.

```text
Model
  ↓
Probability
  ↓
Business Threshold
  ↓
Final 0 / 1 Decision
```

> The model produces the probability. The threshold converts that probability into an operational decision.

---

# 22. Step 20 — Feature Importance

The fitted XGBoost model provides:

```python
feature_importances_
```

The fitted preprocessing Pipeline provides:

```python
get_feature_names_out()
```

Together they allow the project to understand which transformed features are most important to the model overall.

Feature Importance is useful for global understanding, but it does not fully explain individual predictions.

---

# 23. Step 21 — SHAP Explainability

The project uses SHAP for more detailed model explanations.

```python
explainer = shap.TreeExplainer(
    fitted_classifier
)
```

Then:

```python
shap_values = explainer(
    transformed_sample
)
```

Two key views are produced:

```text
SHAP Summary Plot
        ↓
Global model behavior
```

```text
SHAP Waterfall Plot
        ↓
Individual prediction explanation
```

> SHAP explains how the model behaves. It does not prove real-world causation.

---

# 24. Step 22 — Save the Complete Pipeline

After training:

```python
joblib.dump(
    xgboost_pipeline,
    XGBOOST_MODEL_PATH
)
```

The project saves the complete Pipeline, not just the XGBoost classifier.

```text
SimpleImputer
+
StandardScaler
+
OneHotEncoder
+
ColumnTransformer
+
XGBoost
```

This is critical because future applications should not manually reproduce preprocessing logic.

---

# 25. Step 23 — Load the Pipeline

The saved Pipeline can later be loaded using:

```python
pipeline = joblib.load(
    XGBOOST_MODEL_PATH
)
```

```text
Train once
    ↓
Save model
    ↓
Load model
    ↓
Predict many times
```

---

# 26. Step 24 — Predict New Records

A new record may look like:

```python
record = {
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 8500,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
```

Convert the input into a DataFrame:

```python
input_dataframe = pd.DataFrame(
    [record]
)
```

Then:

```python
positive_probability = (
    pipeline
    .predict_proba(
        input_dataframe
    )[0, 1]
)
```

The application does not manually call the Imputer, Scaler, or OneHotEncoder because the saved Pipeline already contains those components.

---

# 27. Complete Chapters 1–3 Architecture

```text
                           RAW DATA
                              │
                              ▼
                         Load Data
                              │
                              ▼
                             EDA
                              │
                              ▼
                    Create Features + Target
                              │
                              ▼
                       Remove Leakage
                              │
                              ▼
                     Train / Test Split
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
              X_train                   X_test
              y_train                   y_test
                 │
                 ▼
             Identify Types
             /            \
            /              \
      Numeric            Categorical
         │                    │
      Imputer              Imputer
         │                    │
      Scaler            OneHotEncoder
          \                  /
           \                /
            └───────┬──────┘
                    ▼
             ColumnTransformer
                    │
                    ▼
                 Pipeline
                    │
           ┌────────┴─────────┐
           ▼                  ▼
 Logistic Regression       XGBoost
      Baseline
           │                  │
           └────────┬─────────┘
                    ▼
              Cross Validation
                    │
                    ▼
                 Model Fit
                    │
                    ▼
                 X_test
                    │
                    ▼
              predict_proba()
                    │
                    ▼
                Probability
                    │
           ┌────────┴────────┐
           ▼                 ▼
       Evaluation        Threshold
           │                 │
           ▼                 ▼
       ROC / PR         0 / 1 Decision
       AUC etc.              │
                             ▼
                    Precision / Recall
                             │
                             ▼
                    Feature Importance
                             │
                             ▼
                           SHAP
                             │
                             ▼
                     Model Explanation
                             │
                             ▼
                       joblib.dump()
                             │
                             ▼
                  xgboost_pipeline.pkl
```

---

# 28. Key Concepts to Remember

After Chapters 1–3, you should be able to explain:

```text
Feature
Target
Label
Data Leakage
Train Set
Test Set
Stratification
Numeric Feature
Categorical Feature
SimpleImputer
StandardScaler
OneHotEncoder
ColumnTransformer
Pipeline
Logistic Regression
Baseline Model
XGBoost
Class Imbalance
scale_pos_weight
Cross Validation
StratifiedKFold
Overfitting
Prediction
Probability
Threshold
Accuracy
Precision
Recall
F1
ROC-AUC
Average Precision
False Positive
False Negative
Feature Importance
SHAP
Model Serialization
Inference
```

---

# 29. Code You Should Be Comfortable Writing

## Train/Test Split

```python
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)
```

## Build a Pipeline

```python
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)
```

## Train

```python
pipeline.fit(
    X_train,
    y_train
)
```

## Predict Classes

```python
y_pred = pipeline.predict(
    X_test
)
```

## Predict Probabilities

```python
y_probability = (
    pipeline
    .predict_proba(
        X_test
    )[:, 1]
)
```

## Threshold Conversion

```python
prediction = (
    y_probability
    >= threshold
).astype(int)
```

## Access Pipeline Steps

```python
fitted_preprocessor = (
    pipeline
    .named_steps[
        "preprocessor"
    ]
)
```

```python
fitted_classifier = (
    pipeline
    .named_steps[
        "classifier"
    ]
)
```

## Save Pipeline

```python
joblib.dump(
    pipeline,
    MODEL_PATH
)
```

## Load Pipeline

```python
pipeline = joblib.load(
    MODEL_PATH
)
```

---

# 30. Development vs Production Boundary

Chapters 1–3 are primarily the **Machine Learning Development** phase.

They end with:

```text
xgboost_pipeline.pkl
```

That file becomes the bridge into the production/integration phase.

```text
Chapters 1–3

Data
  ↓
EDA
  ↓
Feature Engineering
  ↓
Preprocessing
  ↓
Model Training
  ↓
Model Validation
  ↓
Threshold Tuning
  ↓
Explainability
  ↓
xgboost_pipeline.pkl
```

The next stage can then begin:

```text
Saved Pipeline
      ↓
Prediction Service
      ↓
FastAPI
      ↓
Prediction API
      ↓
LangGraph Workflow
      ↓
Business User
```

The key architectural principle is:

```text
Machine Learning predicts.
Prediction API exposes the result.
LangGraph orchestrates the workflow.
An explanation layer explains the result.
```

---

# 31. One-Sentence Project Summary

> I built an end-to-end machine-learning pipeline that loads raw banking data, separates features and target, prevents data leakage, preprocesses numeric and categorical variables, trains and validates Logistic Regression and XGBoost models, evaluates probability-based and classification metrics, tunes the decision threshold, explains model behavior with Feature Importance and SHAP, and saves the complete preprocessing-and-model Pipeline for future inference.

---

# 32. Recommended Execution Order

Install dependencies:

```powershell
pip install -r requirements.txt
```

Place the dataset at:

```text
data/raw/bank-full.csv
```

Train the Logistic Regression baseline:

```powershell
python -m src.train_baseline
```

Train XGBoost:

```powershell
python -m src.train_xgboost
```

Tune thresholds:

```powershell
python -m src.threshold_tuning
```

Generate explainability reports:

```powershell
python -m src.explain_model
```

Run a sample prediction:

```powershell
python -m src.predict
```

---

# 33. Final Learning Outcome

After completing Chapters 1–3, you should be able to explain and implement the entire flow:

```text
Raw Data
    ↓
Feature Preparation
    ↓
Preprocessing
    ↓
Model Training
    ↓
Validation
    ↓
Probability
    ↓
Threshold
    ↓
Decision
    ↓
Explanation
    ↓
Saved Pipeline
```

This complete Pipeline is the foundation for exposing the model through an API and integrating it into an AI-driven AML workflow.
