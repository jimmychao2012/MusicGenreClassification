import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.mixture import GaussianMixture

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# =========================
# 讀取資料
# =========================

df = pd.read_csv(
    "features/global_features.csv"
)

X = df.drop(
    "label",
    axis=1
)

y = df["label"]

# =========================
# 5 Fold
# =========================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =========================
# Models
# =========================

models = {

    "SVM":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=10
            ))
        ]),

    "KNN":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5
            ))
        ]),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42
        )
}

# =========================
# Evaluation
# =========================

results = []

for model_name, model in models.items():

    print(f"\n{'='*40}")
    print(model_name)
    print(f"{'='*40}")

    fold_acc = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y)
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        acc = accuracy_score(
            y_test,
            y_pred
        )

        fold_acc.append(acc)

        print(
            f"Fold {fold+1}: {acc:.4f}"
        )

    mean_acc = np.mean(
        fold_acc
    )

    std_acc = np.std(
        fold_acc
    )

    results.append([
        model_name,
        mean_acc,
        std_acc
    ])

# =========================
# Result Table
# =========================

result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Mean Accuracy",
        "Std"
    ]
)

result_df = result_df.sort_values(
    by="Mean Accuracy",
    ascending=False
)

print("\n")
print(result_df)

result_df.to_csv(
    "results/model_comparison.csv",
    index=False
)