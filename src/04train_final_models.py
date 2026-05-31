import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    "features/segment_features.csv"
)

X = df.drop(
    "label",
    axis=1
)

y_text = df["label"]

# =========================
# Label Encoding
# =========================

encoder = LabelEncoder()

y = encoder.fit_transform(
    y_text
)

# =========================
# Create Model Folder
# =========================

os.makedirs(
    "models",
    exist_ok=True
)

# =========================
# Save Label Encoder
# =========================

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("Label Encoder Saved")

# =========================
# SVM
# =========================

print("\nTraining SVM...")

svm_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            kernel="rbf",
            C=10,
            probability=True
        )
    )
])

svm_model.fit(
    X,
    y
)

joblib.dump(
    svm_model,
    "models/svm_model.pkl"
)

print("SVM Saved")

# =========================
# Random Forest
# =========================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(

    n_estimators=300,

    random_state=42
)

rf_model.fit(
    X,
    y
)

joblib.dump(
    rf_model,
    "models/rf_model.pkl"
)

print("Random Forest Saved")

# =========================
# XGBoost
# =========================

print("\nTraining XGBoost...")

xgb_model = XGBClassifier(

    objective="multi:softprob",

    num_class=len(
        encoder.classes_
    ),

    n_estimators=300,

    max_depth=6,

    learning_rate=0.1,

    eval_metric="mlogloss",

    random_state=42
)

xgb_model.fit(
    X,
    y
)

joblib.dump(
    xgb_model,
    "models/xgb_model.pkl"
)

print("XGBoost Saved")

print("\n====================")
print("All Models Saved")
print("====================")