import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

# Note: Missing values are represented as ' ?' in this raw dataset
df = pd.read_csv(
    url, names=columns, na_values=" ?", skipinitialspace=True
)

# 2. Clean & Prepare Features/Target
df["income"] = df["income"].str.strip()
y = (df["income"] == ">50K").astype(int)  # 1 for >50K, 0 for <=50K

# Drop redundant feature 'education' (already captured by education_num) and ID-like weight
X = df.drop(columns=["income", "education", "fnlwgt"])

# Fill missing values with column mode
X = X.fillna(X.mode().iloc[0])

# 3. Setup Preprocessing Pipeline
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 4. Build End-to-End Model Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
        ),
    ]
)

# 5. Split, Train, and Evaluate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
pipeline.fit(X_train, y_train)

# Predictions & Metrics
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")